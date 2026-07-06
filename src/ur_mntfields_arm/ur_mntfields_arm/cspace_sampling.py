from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics, look_at_rotation


def _sample_unit_ball_direction(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=(dim,))
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros((dim,), dtype=np.float64)
    return v / n


def _clearance_to_speed_label(
    clearance_m: float,
    clearance_margin_m: float,
    clearance_offset_m: float,
    label_floor: float = 0.0,
    label_power: float = 1.0,
) -> float:
    margin = max(float(clearance_margin_m), 1.0e-6)
    offset = float(np.clip(clearance_offset_m, 0.0, margin - 1.0e-6))
    alpha = float(np.clip(float(clearance_m), offset, margin) / margin)
    alpha = alpha ** max(1.0e-6, float(label_power))
    floor = float(np.clip(label_floor, 0.0, 1.0))
    return floor + (1.0 - floor) * alpha


def _q0_low_speed_keep_probability(
    clearance_m: np.ndarray | float,
    clearance_margin_m: float,
    clearance_offset_m: float,
) -> np.ndarray | float:
    margin = max(float(clearance_margin_m), 1.0e-6)
    offset = float(np.clip(clearance_offset_m, 0.0, margin - 1.0e-6))
    speed = np.clip(np.asarray(clearance_m, dtype=np.float32), offset, margin) / margin
    prob = np.where(speed <= 0.35, 1.0, np.where(speed <= 0.65, 0.65, 0.25)).astype(np.float32)
    if np.isscalar(clearance_m):
        return float(prob)
    return prob


def _physical_gradient_to_normalized_direction(
    kinematics: UR5Kinematics,
    grad_q: np.ndarray,
) -> np.ndarray:
    grad_q = np.asarray(grad_q, dtype=np.float64)
    q_span = np.asarray(kinematics.joint_max - kinematics.joint_min, dtype=np.float64)
    grad_qn = grad_q * q_span
    n = np.linalg.norm(grad_qn)
    if n < 1e-9:
        return np.zeros((6,), dtype=np.float64)
    return grad_qn / n


def _normalized_contact_point(
    kinematics: UR5Kinematics,
    q: np.ndarray,
    normal_qn: np.ndarray,
    clearance_m: float,
    clearance_margin_m: float,
) -> np.ndarray:
    qn = kinematics.normalize(np.asarray(q, dtype=np.float64)).astype(np.float64)
    normal_qn = np.asarray(normal_qn, dtype=np.float64)
    n = np.linalg.norm(normal_qn)
    if n < 1e-9:
        return qn
    # This is a debug-only proxy for the closest configuration-space point.
    # The physical clearance is in meters, so scale it to the normalized label
    # range instead of moving by meters in joint-radian coordinates.
    step = float(np.clip(clearance_m / max(float(clearance_margin_m), 1.0e-6), 0.0, 1.0))
    return np.clip(qn - step * (normal_qn / n), -0.5, 0.5).astype(np.float64)


def make_cspace_pair_rows_from_q_pairs(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    q0_batch: np.ndarray,
    q1_batch: np.ndarray,
    clearance_margin_m: float,
    clearance_offset_m: float,
    *,
    clearance_label_floor: float = 0.0,
    clearance_label_power: float = 1.0,
    require_offset_clearance: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    q0_arr = np.asarray(q0_batch, dtype=np.float32)
    q1_arr = np.asarray(q1_batch, dtype=np.float32)
    if q0_arr.ndim == 1:
        q0_arr = q0_arr[None, :]
    if q1_arr.ndim == 1:
        q1_arr = q1_arr[None, :]
    count = min(len(q0_arr), len(q1_arr))
    if count <= 0:
        stats = _finalize_sampler_stats(
            {"sampling_mode": "q_pair_rows", "attempts": 0.0, "accepted_pairs": 0.0},
            [],
            [],
            [],
            clearance_margin_m,
        )
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), stats

    q0_arr = q0_arr[:count]
    q1_arr = q1_arr[:count]
    d0_batch, n0_q_batch = checker.clearance_and_normal_batch(q0_arr)
    d1_batch, n1_q_batch = checker.clearance_and_normal_batch(q1_arr)

    raw_rows: list[np.ndarray] = []
    pair_rows: list[np.ndarray] = []
    d0_values: list[float] = []
    d1_values: list[float] = []
    for q0, q1, d0, d1, n0_q, n1_q in zip(q0_arr, q1_arr, d0_batch, d1_batch, n0_q_batch, n1_q_batch):
        if require_offset_clearance and (float(d0) < clearance_offset_m or float(d1) < clearance_offset_m):
            continue
        if np.linalg.norm(n0_q) < 1e-8 or np.linalg.norm(n1_q) < 1e-8:
            continue
        q0n = kinematics.normalize(q0)
        q1n = kinematics.normalize(q1)
        if not (
            np.all(np.isfinite(q0n))
            and np.all(np.isfinite(q1n))
            and np.all(q0n >= -0.5)
            and np.all(q0n <= 0.5)
            and np.all(q1n >= -0.5)
            and np.all(q1n <= 0.5)
        ):
            continue
        y0 = _clearance_to_speed_label(
            float(d0),
            clearance_margin_m,
            clearance_offset_m,
            label_floor=clearance_label_floor,
            label_power=clearance_label_power,
        )
        y1 = _clearance_to_speed_label(
            float(d1),
            clearance_margin_m,
            clearance_offset_m,
            label_floor=clearance_label_floor,
            label_power=clearance_label_power,
        )
        n0 = _physical_gradient_to_normalized_direction(kinematics, n0_q)
        n1 = _physical_gradient_to_normalized_direction(kinematics, n1_q)
        if np.linalg.norm(n0) < 1e-8 or np.linalg.norm(n1) < 1e-8:
            continue
        cp0 = _normalized_contact_point(kinematics, q0, n0, float(d0), clearance_margin_m)
        cp1 = _normalized_contact_point(kinematics, q1, n1, float(d1), clearance_margin_m)
        raw_rows.append(np.concatenate((q0n, cp0), axis=0).astype(np.float32))
        raw_rows.append(np.concatenate((q1n, cp1), axis=0).astype(np.float32))
        pair_rows.append(np.concatenate((q0n, q1n, [y0, y1], n0, n1), axis=0).astype(np.float32))
        d0_values.append(float(d0))
        d1_values.append(float(d1))

    stats = {
        "sampling_mode": "q_pair_rows",
        "attempts": float(count),
        "ik_seed_tries": 0.0,
        "accepted_pairs": float(len(pair_rows)),
        "acceptance_rate": 0.0 if count <= 0 else float(len(pair_rows)) / float(count),
        "accepted_per_seed": 0.0,
        "samples_per_seed": 0.0,
        "refined_q0": 0.0,
        "refined_q1": 0.0,
        "anchor_seed_tries": 0.0,
        "anchor_seed_success": 0.0,
        "roi_seed_tries": 0.0,
        "workspace_seed_tries": 0.0,
        "roi_seed_success": 0.0,
        "workspace_seed_success": 0.0,
    }
    rows = np.asarray(pair_rows, dtype=np.float32) if pair_rows else np.zeros((0, 26), dtype=np.float32)
    stats = _finalize_sampler_stats(stats, rows, d0_values, d1_values, clearance_margin_m)
    raw = np.asarray(raw_rows, dtype=np.float32) if raw_rows else np.zeros((0, 12), dtype=np.float32)
    return raw, rows, stats


def _make_rows_from_labeled_q_pairs(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    q0_batch: np.ndarray,
    q1_batch: np.ndarray,
    d0_batch: np.ndarray,
    d1_batch: np.ndarray,
    n0_q_batch: np.ndarray,
    n1_q_batch: np.ndarray,
    clearance_margin_m: float,
    clearance_offset_m: float,
    *,
    clearance_label_floor: float = 0.0,
    clearance_label_power: float = 1.0,
    require_offset_clearance: bool = False,
    sampling_mode: str = "labeled_q_pair_rows",
    attempts: int | None = None,
    ik_seed_tries: int = 0,
    anchor_seed_tries: int = 0,
    anchor_seed_success: int = 0,
    roi_seed_tries: int = 0,
    workspace_seed_tries: int = 0,
    roi_seed_success: int = 0,
    workspace_seed_success: int = 0,
    samples_per_seed: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    q0_arr = np.asarray(q0_batch, dtype=np.float32)
    q1_arr = np.asarray(q1_batch, dtype=np.float32)
    d0_arr = np.asarray(d0_batch, dtype=np.float32).reshape(-1)
    d1_arr = np.asarray(d1_batch, dtype=np.float32).reshape(-1)
    n0_arr = np.asarray(n0_q_batch, dtype=np.float32)
    n1_arr = np.asarray(n1_q_batch, dtype=np.float32)
    count = min(len(q0_arr), len(q1_arr), len(d0_arr), len(d1_arr), len(n0_arr), len(n1_arr))
    if count <= 0:
        stats = _finalize_sampler_stats(
            {
                "sampling_mode": sampling_mode,
                "attempts": float(0 if attempts is None else attempts),
                "ik_seed_tries": float(ik_seed_tries),
                "accepted_pairs": 0.0,
                "acceptance_rate": 0.0,
                "accepted_per_seed": 0.0,
                "samples_per_seed": float(samples_per_seed),
                "refined_q0": 0.0,
                "refined_q1": 0.0,
                "anchor_seed_tries": float(anchor_seed_tries),
                "anchor_seed_success": float(anchor_seed_success),
                "roi_seed_tries": float(roi_seed_tries),
                "workspace_seed_tries": float(workspace_seed_tries),
                "roi_seed_success": float(roi_seed_success),
                "workspace_seed_success": float(workspace_seed_success),
            },
            [],
            [],
            [],
            clearance_margin_m,
        )
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), stats

    raw_rows: list[np.ndarray] = []
    pair_rows: list[np.ndarray] = []
    d0_values: list[float] = []
    d1_values: list[float] = []
    for q0, q1, d0, d1, n0_q, n1_q in zip(
        q0_arr[:count],
        q1_arr[:count],
        d0_arr[:count],
        d1_arr[:count],
        n0_arr[:count],
        n1_arr[:count],
    ):
        if require_offset_clearance and (float(d0) < clearance_offset_m or float(d1) < clearance_offset_m):
            continue
        if np.linalg.norm(n0_q) < 1e-8 or np.linalg.norm(n1_q) < 1e-8:
            continue
        q0n = kinematics.normalize(q0)
        q1n = kinematics.normalize(q1)
        if not (
            np.all(np.isfinite(q0n))
            and np.all(np.isfinite(q1n))
            and np.all(q0n >= -0.5)
            and np.all(q0n <= 0.5)
            and np.all(q1n >= -0.5)
            and np.all(q1n <= 0.5)
        ):
            continue
        y0 = _clearance_to_speed_label(
            float(d0),
            clearance_margin_m,
            clearance_offset_m,
            label_floor=clearance_label_floor,
            label_power=clearance_label_power,
        )
        y1 = _clearance_to_speed_label(
            float(d1),
            clearance_margin_m,
            clearance_offset_m,
            label_floor=clearance_label_floor,
            label_power=clearance_label_power,
        )
        n0 = _physical_gradient_to_normalized_direction(kinematics, n0_q)
        n1 = _physical_gradient_to_normalized_direction(kinematics, n1_q)
        if np.linalg.norm(n0) < 1e-8 or np.linalg.norm(n1) < 1e-8:
            continue
        cp0 = _normalized_contact_point(kinematics, q0, n0, float(d0), clearance_margin_m)
        cp1 = _normalized_contact_point(kinematics, q1, n1, float(d1), clearance_margin_m)
        raw_rows.append(np.concatenate((q0n, cp0), axis=0).astype(np.float32))
        raw_rows.append(np.concatenate((q1n, cp1), axis=0).astype(np.float32))
        pair_rows.append(np.concatenate((q0n, q1n, [y0, y1], n0, n1), axis=0).astype(np.float32))
        d0_values.append(float(d0))
        d1_values.append(float(d1))

    attempts_f = float(count if attempts is None else attempts)
    stats = {
        "sampling_mode": sampling_mode,
        "attempts": attempts_f,
        "ik_seed_tries": float(ik_seed_tries),
        "accepted_pairs": float(len(pair_rows)),
        "acceptance_rate": 0.0 if attempts_f <= 0 else float(len(pair_rows)) / attempts_f,
        "accepted_per_seed": 0.0 if ik_seed_tries <= 0 else float(len(pair_rows)) / float(ik_seed_tries),
        "samples_per_seed": float(samples_per_seed),
        "refined_q0": 0.0,
        "refined_q1": 0.0,
        "anchor_seed_tries": float(anchor_seed_tries),
        "anchor_seed_success": float(anchor_seed_success),
        "roi_seed_tries": float(roi_seed_tries),
        "workspace_seed_tries": float(workspace_seed_tries),
        "roi_seed_success": float(roi_seed_success),
        "workspace_seed_success": float(workspace_seed_success),
    }
    rows = np.asarray(pair_rows, dtype=np.float32) if pair_rows else np.zeros((0, 26), dtype=np.float32)
    raw = np.asarray(raw_rows, dtype=np.float32) if raw_rows else np.zeros((0, 12), dtype=np.float32)
    return raw, rows, _finalize_sampler_stats(stats, rows, d0_values, d1_values, clearance_margin_m)


def _unit_directions_batch(rng: np.random.Generator, count: int, dim: int = 6) -> np.ndarray:
    dirs = rng.normal(size=(max(0, int(count)), dim)).astype(np.float64)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    bad = norms[:, 0] < 1.0e-9
    if np.any(bad):
        dirs[bad, 0] = 1.0
        norms[bad, 0] = 1.0
    return dirs / np.maximum(norms, 1.0e-9)


def _stratified_indices_by_clearance(
    rng: np.random.Generator,
    d0: np.ndarray,
    count: int,
    clearance_margin_m: float,
    near_boundary_only: bool,
) -> np.ndarray:
    d0 = np.asarray(d0, dtype=np.float32).reshape(-1)
    if len(d0) == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    margin = max(float(clearance_margin_m), 1.0e-6)
    if near_boundary_only:
        bands = (
            np.where(d0 <= 0.35 * margin)[0],
            np.where((d0 > 0.35 * margin) & (d0 <= 0.70 * margin))[0],
            np.where((d0 > 0.70 * margin) & (d0 <= margin))[0],
        )
        quotas = (0.45, 0.35, 0.20)
    else:
        bands = (
            np.where(d0 <= 0.35 * margin)[0],
            np.where((d0 > 0.35 * margin) & (d0 <= margin))[0],
            np.where((d0 > margin) & (d0 <= 2.0 * margin))[0],
            np.where(d0 > 2.0 * margin)[0],
        )
        quotas = (0.18, 0.22, 0.25, 0.35)
    chosen: list[int] = []
    used = np.zeros((len(d0),), dtype=bool)
    for band, frac in zip(bands, quotas):
        if len(chosen) >= count or len(band) == 0:
            continue
        take = min(len(band), max(1, int(round(float(count) * float(frac)))))
        pick = rng.choice(band, size=take, replace=False) if len(band) > take else band
        chosen.extend(np.asarray(pick, dtype=np.int64).tolist())
        used[np.asarray(pick, dtype=np.int64)] = True
    if len(chosen) < count:
        remaining = np.where(~used)[0]
        if len(remaining):
            take = min(count - len(chosen), len(remaining))
            pick = rng.choice(remaining, size=take, replace=False) if len(remaining) > take else remaining
            chosen.extend(np.asarray(pick, dtype=np.int64).tolist())
    if len(chosen) > count:
        chosen = rng.choice(np.asarray(chosen, dtype=np.int64), size=count, replace=False).tolist()
    return np.asarray(chosen, dtype=np.int64)


def _sample_joint_local_stratified_batch(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    num_pairs: int,
    clearance_margin_m: float,
    clearance_offset_m: float,
    rng: np.random.Generator,
    *,
    samples_per_seed: int,
    seed_hint_q: np.ndarray | None,
    anchor_qs: np.ndarray | None,
    anchor_seed_probability: float,
    clearance_label_floor: float,
    clearance_label_power: float,
    sampling_mode: str,
    proposal_batch_size: int,
    near_boundary_only: bool,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target = max(0, int(num_pairs))
    if target <= 0:
        return _make_rows_from_labeled_q_pairs(
            checker,
            kinematics,
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            clearance_margin_m,
            clearance_offset_m,
            sampling_mode=f"{sampling_mode}:{'near' if near_boundary_only else 'broad'}:stratified",
        )

    anchors = np.asarray(anchor_qs, dtype=np.float64) if anchor_qs is not None else np.zeros((0, 6), dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 6:
        anchors = np.zeros((0, 6), dtype=np.float64)
    anchor_qn = np.asarray([kinematics.normalize(q) for q in anchors], dtype=np.float32) if len(anchors) else np.zeros((0, 6), dtype=np.float32)
    seed_hint_qn = (
        kinematics.normalize(np.asarray(seed_hint_q, dtype=np.float64)).astype(np.float32)
        if seed_hint_q is not None
        else None
    )

    q0_pool: list[np.ndarray] = []
    q1_pool: list[np.ndarray] = []
    d0_pool: list[np.ndarray] = []
    d1_pool: list[np.ndarray] = []
    n0_pool: list[np.ndarray] = []
    n1_pool: list[np.ndarray] = []
    attempts = 0
    seed_batches = 0
    anchor_seed_tries = 0
    anchor_seed_success = 0
    workspace_seed_tries = 0
    workspace_seed_success = 0
    projected_q0 = 0

    proposal_batch_size = max(128, int(proposal_batch_size))
    # Near-band samples are sparse, so over-generate once and stratify. This is
    # still cheaper than per-seed rejection because clearance is evaluated in two
    # large batches per round.
    multiplier = 10 if near_boundary_only else 4
    max_attempts = max(target * (18 if near_boundary_only else 8), proposal_batch_size)
    report_every = max(proposal_batch_size, target)
    while sum(len(x) for x in q0_pool) < target and attempts < max_attempts:
        batch = min(max(proposal_batch_size, target * multiplier), max_attempts - attempts)
        if batch <= 0:
            break
        attempts += batch
        seed_batches += max(1, int(np.ceil(batch / max(1, int(samples_per_seed)))))

        q0n = rng.uniform(-0.5, 0.5, size=(batch, 6)).astype(np.float32)
        workspace_seed_tries += batch
        workspace_seed_success += batch

        if seed_hint_qn is not None:
            if near_boundary_only:
                hint_frac = 0.65
            else:
                hint_frac = 0.35
            hint_mask = rng.random(batch) < hint_frac
            if np.any(hint_mask):
                radii = rng.uniform(0.005 if near_boundary_only else 0.0, 0.09 if near_boundary_only else 0.50, size=int(np.count_nonzero(hint_mask)))
                dirs = _unit_directions_batch(rng, int(np.count_nonzero(hint_mask)))
                q0n[hint_mask] = np.clip(seed_hint_qn[None, :] + dirs * radii[:, None], -0.5, 0.5)

        if len(anchor_qn) > 0 and anchor_seed_probability > 0.0:
            anchor_mask = rng.random(batch) < float(np.clip(anchor_seed_probability, 0.0, 1.0))
            anchor_count = int(np.count_nonzero(anchor_mask))
            anchor_seed_tries += anchor_count
            if anchor_count > 0:
                anchor_idx = rng.integers(0, len(anchor_qn), size=anchor_count)
                radii = rng.uniform(0.004, 0.06 if near_boundary_only else 0.25, size=anchor_count)
                dirs = _unit_directions_batch(rng, anchor_count)
                q0n[anchor_mask] = np.clip(anchor_qn[anchor_idx] + dirs * radii[:, None], -0.5, 0.5)
                anchor_seed_success += anchor_count

        q1_radius = rng.uniform(0.015, 0.18 if near_boundary_only else 0.50, size=batch)
        q1n = np.clip(q0n.astype(np.float64) + _unit_directions_batch(rng, batch) * q1_radius[:, None], -0.5, 0.5).astype(np.float32)
        q0 = kinematics.denormalize(q0n).astype(np.float32)
        q1 = kinematics.denormalize(q1n).astype(np.float32)
        d0, n0 = checker.clearance_and_normal_batch(q0)
        d1, n1 = checker.clearance_and_normal_batch(q1)
        if near_boundary_only:
            n0_qn = np.asarray(
                [_physical_gradient_to_normalized_direction(kinematics, n) for n in n0],
                dtype=np.float32,
            )
            n0_qn_norm = np.linalg.norm(n0_qn, axis=1)
            project_ok = np.isfinite(d0) & (n0_qn_norm > 1.0e-8) & (d0 > float(clearance_margin_m))
            if np.any(project_ok):
                target_clearance = rng.uniform(
                    max(float(clearance_offset_m), 1.0e-6),
                    max(float(clearance_margin_m), float(clearance_offset_m) + 1.0e-6),
                    size=int(np.count_nonzero(project_ok)),
                ).astype(np.float32)
                step = np.clip(
                    (d0[project_ok].astype(np.float32) - target_clearance)
                    / max(float(clearance_margin_m), 1.0e-6),
                    0.0,
                    0.45,
                )
                q0n_projected = q0n[project_ok].astype(np.float32) - n0_qn[project_ok] * step[:, None]
                q0n_projected = np.clip(q0n_projected, -0.5, 0.5).astype(np.float32)
                q1_radius_projected = rng.uniform(0.015, 0.18, size=len(q0n_projected))
                q1n_projected = np.clip(
                    q0n_projected.astype(np.float64)
                    + _unit_directions_batch(rng, len(q0n_projected)) * q1_radius_projected[:, None],
                    -0.5,
                    0.5,
                ).astype(np.float32)
                q0_projected = kinematics.denormalize(q0n_projected).astype(np.float32)
                q1_projected = kinematics.denormalize(q1n_projected).astype(np.float32)
                d0_projected, n0_projected = checker.clearance_and_normal_batch(q0_projected)
                d1_projected, n1_projected = checker.clearance_and_normal_batch(q1_projected)
                q0 = np.concatenate((q0, q0_projected), axis=0)
                q1 = np.concatenate((q1, q1_projected), axis=0)
                d0 = np.concatenate((d0, d0_projected), axis=0)
                d1 = np.concatenate((d1, d1_projected), axis=0)
                n0 = np.concatenate((n0, n0_projected), axis=0)
                n1 = np.concatenate((n1, n1_projected), axis=0)
                projected_q0 += int(len(q0_projected))
        normal_ok = (np.linalg.norm(n0, axis=1) > 1.0e-8) & (np.linalg.norm(n1, axis=1) > 1.0e-8)
        offset_ok = d0 >= float(clearance_offset_m)
        finite_ok = np.isfinite(d0) & np.isfinite(d1) & normal_ok & offset_ok
        if near_boundary_only:
            finite_ok &= d0 <= float(clearance_margin_m)
        idx_valid = np.where(finite_ok)[0]
        if len(idx_valid):
            need = target - sum(len(x) for x in q0_pool)
            # Keep a little extra per round so the final stratification can
            # rebalance clearance bands.
            idx_keep_local = _stratified_indices_by_clearance(
                rng,
                d0[idx_valid],
                min(max(need * 2, need), len(idx_valid)),
                clearance_margin_m,
                near_boundary_only,
            )
            idx_keep = idx_valid[idx_keep_local]
            q0_pool.append(q0[idx_keep])
            q1_pool.append(q1[idx_keep])
            d0_pool.append(d0[idx_keep])
            d1_pool.append(d1[idx_keep])
            n0_pool.append(n0[idx_keep])
            n1_pool.append(n1[idx_keep])
        accepted_so_far = min(target, sum(len(x) for x in q0_pool))
        if progress_cb is not None and (attempts % report_every == 0 or accepted_so_far >= target):
            progress_cb(attempts, accepted_so_far, seed_batches)

    if q0_pool:
        q0_all = np.concatenate(q0_pool, axis=0).astype(np.float32)
        q1_all = np.concatenate(q1_pool, axis=0).astype(np.float32)
        d0_all = np.concatenate(d0_pool, axis=0).astype(np.float32)
        d1_all = np.concatenate(d1_pool, axis=0).astype(np.float32)
        n0_all = np.concatenate(n0_pool, axis=0).astype(np.float32)
        n1_all = np.concatenate(n1_pool, axis=0).astype(np.float32)
        final_idx = _stratified_indices_by_clearance(rng, d0_all, min(target, len(q0_all)), clearance_margin_m, near_boundary_only)
        q0_all = q0_all[final_idx]
        q1_all = q1_all[final_idx]
        d0_all = d0_all[final_idx]
        d1_all = d1_all[final_idx]
        n0_all = n0_all[final_idx]
        n1_all = n1_all[final_idx]
    else:
        q0_all = np.zeros((0, 6), dtype=np.float32)
        q1_all = np.zeros((0, 6), dtype=np.float32)
        d0_all = np.zeros((0,), dtype=np.float32)
        d1_all = np.zeros((0,), dtype=np.float32)
        n0_all = np.zeros((0, 6), dtype=np.float32)
        n1_all = np.zeros((0, 6), dtype=np.float32)

    raw, rows, stats = _make_rows_from_labeled_q_pairs(
        checker,
        kinematics,
        q0_all,
        q1_all,
        d0_all,
        d1_all,
        n0_all,
        n1_all,
        clearance_margin_m,
        clearance_offset_m,
        clearance_label_floor=clearance_label_floor,
        clearance_label_power=clearance_label_power,
        sampling_mode=f"{sampling_mode}:{'near' if near_boundary_only else 'broad'}:stratified",
        attempts=attempts,
        ik_seed_tries=seed_batches,
        anchor_seed_tries=anchor_seed_tries,
        anchor_seed_success=anchor_seed_success,
        workspace_seed_tries=workspace_seed_tries,
        workspace_seed_success=workspace_seed_success,
        samples_per_seed=float(samples_per_seed),
    )
    stats["refined_q0"] = float(projected_q0)
    return raw, rows, stats


def _pose_from_position_rpy(position: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = float(np.cos(roll))
    sr = float(np.sin(roll))
    cp = float(np.cos(pitch))
    sp = float(np.sin(pitch))
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = cp * cy
    pose[0, 1] = sr * sp * cy - cr * sy
    pose[0, 2] = cr * sp * cy + sr * sy
    pose[1, 0] = cp * sy
    pose[1, 1] = sr * sp * sy + cr * cy
    pose[1, 2] = cr * sp * sy - sr * cy
    pose[2, 0] = -sp
    pose[2, 1] = sr * cp
    pose[2, 2] = cr * cp
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def _sample_legacy_pair_pose(
    rng: np.random.Generator,
    roi_min: np.ndarray | None = None,
    roi_max: np.ndarray | None = None,
) -> np.ndarray:
    global_min = np.array([0.20, -0.60, -0.10], dtype=np.float64)
    global_max = np.array([1.40, 0.60, 1.10], dtype=np.float64)
    pos_min = global_min
    pos_max = global_max
    if roi_min is not None and roi_max is not None and float(rng.uniform()) < 0.70:
        roi_min = np.asarray(roi_min, dtype=np.float64).reshape(3)
        roi_max = np.asarray(roi_max, dtype=np.float64).reshape(3)
        pad = np.array([0.18, 0.18, 0.18], dtype=np.float64)
        local_min = np.maximum(global_min, roi_min - pad)
        local_max = np.minimum(global_max, roi_max + pad)
        if np.all(local_max > local_min):
            pos_min = local_min
            pos_max = local_max
    pos = rng.uniform(pos_min, pos_max)
    roll = -0.5 * np.pi + (float(rng.uniform()) - 0.5) * 0.6 * np.pi
    pitch = (float(rng.uniform()) - 0.5) * 0.6 * np.pi
    yaw = -0.5 * np.pi + (float(rng.uniform()) - 0.5) * 0.6 * np.pi
    return _pose_from_position_rpy(pos, roll, pitch, yaw)


def _sample_legacy_pair_pose_batch(
    rng: np.random.Generator,
    batch_size: int,
    roi_min: np.ndarray | None = None,
    roi_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((batch_size, 4, 4), dtype=np.float64)
    use_roi = np.zeros((batch_size,), dtype=bool)
    for idx in range(batch_size):
        use_roi[idx] = roi_min is not None and roi_max is not None and float(rng.uniform()) < 0.70
        poses[idx] = _sample_legacy_pair_pose(
            rng,
            roi_min=roi_min if use_roi[idx] else None,
            roi_max=roi_max if use_roi[idx] else None,
        )
    return poses, use_roi


@lru_cache(maxsize=1)
def _get_torch_ik_ur5_cls():
    module_dir = Path("/home/mayank/ur_ws/ntrl-demo/dataprocessing")
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))
    from torch_IK_UR5 import torch_IK_UR5

    return torch_IK_UR5


def _solve_ik_pose_batch_gpu(
    kinematics: UR5Kinematics,
    poses: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape [N,4,4], got {poses.shape}")
    if len(poses) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if not torch.cuda.is_available():
        out = []
        seed = 0.5 * (kinematics.joint_min + kinematics.joint_max)
        for pose in poses:
            q = kinematics.solve_ik(pose.astype(np.float64), seed)
            if q is not None:
                out.append(kinematics.normalize(q).astype(np.float32))
        if not out:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    torch_IK_UR5 = _get_torch_ik_ur5_cls()
    with torch.no_grad():
        pose_t = torch.as_tensor(poses, dtype=torch.float32, device="cuda")
        solver = torch_IK_UR5(int(pose_t.shape[0]))
        solver.setJointLimits(-np.pi, np.pi)
        q_all = solver.solveIK(pose_t)
        q_rad = q_all.reshape(-1, 6)
        q_np = q_rad.detach().cpu().numpy().astype(np.float64)
        q_np = np.asarray([kinematics.clamp(q) for q in q_np], dtype=np.float64)
        qn_np = np.asarray([kinematics.normalize(q) for q in q_np], dtype=np.float32)
        valid = np.all((qn_np >= -0.5) & (qn_np <= 0.5), axis=1)
        qn_np = qn_np[valid]
        if len(qn_np) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return qn_np


def _finalize_sampler_stats(
    stats: dict[str, float],
    pair_rows: list[np.ndarray] | np.ndarray,
    d0_values: list[float],
    d1_values: list[float],
    clearance_margin_m: float,
) -> dict[str, float]:
    out = dict(stats)
    rows = np.asarray(pair_rows, dtype=np.float32) if len(pair_rows) else np.zeros((0, 26), dtype=np.float32)
    d0 = np.asarray(d0_values, dtype=np.float32) if len(d0_values) else np.zeros((0,), dtype=np.float32)
    d1 = np.asarray(d1_values, dtype=np.float32) if len(d1_values) else np.zeros((0,), dtype=np.float32)
    if len(rows):
        speeds = rows[:, 12:14]
        out["speed0_sat_frac"] = float(np.mean(speeds[:, 0] >= 0.999))
        out["speed1_sat_frac"] = float(np.mean(speeds[:, 1] >= 0.999))
        out["speed0_low_frac"] = float(np.mean(speeds[:, 0] <= 0.35))
        out["speed1_low_frac"] = float(np.mean(speeds[:, 1] <= 0.35))
    else:
        out["speed0_sat_frac"] = 0.0
        out["speed1_sat_frac"] = 0.0
        out["speed0_low_frac"] = 0.0
        out["speed1_low_frac"] = 0.0
    if len(d0):
        out["q0_clearance_mean"] = float(np.mean(d0))
        out["q0_clearance_min"] = float(np.min(d0))
        out["q0_clearance_max"] = float(np.max(d0))
        out["q0_near_margin_frac"] = float(np.mean(d0 <= float(clearance_margin_m)))
    else:
        out["q0_clearance_mean"] = 0.0
        out["q0_clearance_min"] = 0.0
        out["q0_clearance_max"] = 0.0
        out["q0_near_margin_frac"] = 0.0
    if len(d1):
        out["q1_clearance_mean"] = float(np.mean(d1))
        out["q1_clearance_min"] = float(np.min(d1))
        out["q1_clearance_max"] = float(np.max(d1))
        out["q1_near_margin_frac"] = float(np.mean(d1 <= float(clearance_margin_m)))
    else:
        out["q1_clearance_mean"] = 0.0
        out["q1_clearance_min"] = 0.0
        out["q1_clearance_max"] = 0.0
        out["q1_near_margin_frac"] = 0.0
    return out


def sample_path_centered_training_batch(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    anchor_qs: np.ndarray,
    num_pairs: int,
    clearance_margin_m: float,
    clearance_offset_m: float,
    rng: np.random.Generator,
    clearance_label_floor: float = 0.0,
    clearance_label_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    anchors = np.asarray(anchor_qs, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 6 or len(anchors) == 0 or num_pairs <= 0:
        stats = {
            "sampling_mode": "path_centered",
            "attempts": 0.0,
            "ik_seed_tries": 0.0,
            "accepted_pairs": 0.0,
            "acceptance_rate": 0.0,
            "accepted_per_seed": 0.0,
            "samples_per_seed": 0.0,
            "refined_q0": 0.0,
            "refined_q1": 0.0,
            "anchor_seed_tries": 0.0,
            "anchor_seed_success": 0.0,
            "roi_seed_tries": 0.0,
            "workspace_seed_tries": 0.0,
            "roi_seed_success": 0.0,
            "workspace_seed_success": 0.0,
        }
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), _finalize_sampler_stats(
            stats, [], [], [], clearance_margin_m
        )

    target = max(0, int(num_pairs))
    anchor_qn = np.asarray([kinematics.normalize(q) for q in anchors], dtype=np.float32)
    q0_pool: list[np.ndarray] = []
    q1_pool: list[np.ndarray] = []
    d0_pool: list[np.ndarray] = []
    d1_pool: list[np.ndarray] = []
    n0_pool: list[np.ndarray] = []
    n1_pool: list[np.ndarray] = []
    attempts = 0
    seed_batches = 0
    projected_q0 = 0
    proposal_batch = max(512, target * 8)
    max_attempts = max(target * 24, proposal_batch)
    while sum(len(x) for x in q0_pool) < target and attempts < max_attempts:
        batch = min(proposal_batch, max_attempts - attempts)
        attempts += batch
        seed_batches += 1

        anchor_idx = rng.integers(0, len(anchor_qn), size=batch)
        # Wider than the old 0.005-0.05 shell: executed paths are usually safe,
        # so strict near-band rejection around anchors is extremely sparse.
        q0_radius = rng.uniform(0.004, 0.16, size=batch)
        q0n = np.clip(
            anchor_qn[anchor_idx].astype(np.float64) + _unit_directions_batch(rng, batch) * q0_radius[:, None],
            -0.5,
            0.5,
        ).astype(np.float32)
        q1_radius = rng.uniform(0.02, 0.35, size=batch)
        q1n = np.clip(
            q0n.astype(np.float64) + _unit_directions_batch(rng, batch) * q1_radius[:, None],
            -0.5,
            0.5,
        ).astype(np.float32)
        q0 = kinematics.denormalize(q0n).astype(np.float32)
        q1 = kinematics.denormalize(q1n).astype(np.float32)
        d0, n0 = checker.clearance_and_normal_batch(q0)
        d1, n1 = checker.clearance_and_normal_batch(q1)

        n0_qn = np.asarray(
            [_physical_gradient_to_normalized_direction(kinematics, n) for n in n0],
            dtype=np.float32,
        )
        n0_qn_norm = np.linalg.norm(n0_qn, axis=1)
        project_ok = np.isfinite(d0) & (n0_qn_norm > 1.0e-8) & (d0 > float(clearance_margin_m))
        if np.any(project_ok):
            target_clearance = rng.uniform(
                max(float(clearance_offset_m), 1.0e-6),
                max(float(clearance_margin_m), float(clearance_offset_m) + 1.0e-6),
                size=int(np.count_nonzero(project_ok)),
            ).astype(np.float32)
            step = np.clip(
                (d0[project_ok].astype(np.float32) - target_clearance)
                / max(float(clearance_margin_m), 1.0e-6),
                0.0,
                0.45,
            )
            q0n_projected = np.clip(q0n[project_ok].astype(np.float32) - n0_qn[project_ok] * step[:, None], -0.5, 0.5)
            q1_radius_projected = rng.uniform(0.02, 0.28, size=len(q0n_projected))
            q1n_projected = np.clip(
                q0n_projected.astype(np.float64)
                + _unit_directions_batch(rng, len(q0n_projected)) * q1_radius_projected[:, None],
                -0.5,
                0.5,
            ).astype(np.float32)
            q0_projected = kinematics.denormalize(q0n_projected).astype(np.float32)
            q1_projected = kinematics.denormalize(q1n_projected).astype(np.float32)
            d0_projected, n0_projected = checker.clearance_and_normal_batch(q0_projected)
            d1_projected, n1_projected = checker.clearance_and_normal_batch(q1_projected)
            q0 = np.concatenate((q0, q0_projected), axis=0)
            q1 = np.concatenate((q1, q1_projected), axis=0)
            d0 = np.concatenate((d0, d0_projected), axis=0)
            d1 = np.concatenate((d1, d1_projected), axis=0)
            n0 = np.concatenate((n0, n0_projected), axis=0)
            n1 = np.concatenate((n1, n1_projected), axis=0)
            projected_q0 += int(len(q0_projected))

        normal_ok = (np.linalg.norm(n0, axis=1) > 1.0e-8) & (np.linalg.norm(n1, axis=1) > 1.0e-8)
        valid = np.isfinite(d0) & np.isfinite(d1) & normal_ok & (d0 >= float(clearance_offset_m))
        idx_valid = np.where(valid)[0]
        if len(idx_valid):
            need = target - sum(len(x) for x in q0_pool)
            idx_local = _stratified_indices_by_clearance(
                rng,
                d0[idx_valid],
                min(max(need * 2, need), len(idx_valid)),
                clearance_margin_m,
                near_boundary_only=False,
            )
            idx_keep = idx_valid[idx_local]
            q0_pool.append(q0[idx_keep])
            q1_pool.append(q1[idx_keep])
            d0_pool.append(d0[idx_keep])
            d1_pool.append(d1[idx_keep])
            n0_pool.append(n0[idx_keep])
            n1_pool.append(n1[idx_keep])

    if q0_pool:
        q0_all = np.concatenate(q0_pool, axis=0).astype(np.float32)
        q1_all = np.concatenate(q1_pool, axis=0).astype(np.float32)
        d0_all = np.concatenate(d0_pool, axis=0).astype(np.float32)
        d1_all = np.concatenate(d1_pool, axis=0).astype(np.float32)
        n0_all = np.concatenate(n0_pool, axis=0).astype(np.float32)
        n1_all = np.concatenate(n1_pool, axis=0).astype(np.float32)
        final_idx = _stratified_indices_by_clearance(
            rng, d0_all, min(target, len(q0_all)), clearance_margin_m, near_boundary_only=False
        )
        q0_all = q0_all[final_idx]
        q1_all = q1_all[final_idx]
        d0_all = d0_all[final_idx]
        d1_all = d1_all[final_idx]
        n0_all = n0_all[final_idx]
        n1_all = n1_all[final_idx]
    else:
        q0_all = np.zeros((0, 6), dtype=np.float32)
        q1_all = np.zeros((0, 6), dtype=np.float32)
        d0_all = np.zeros((0,), dtype=np.float32)
        d1_all = np.zeros((0,), dtype=np.float32)
        n0_all = np.zeros((0, 6), dtype=np.float32)
        n1_all = np.zeros((0, 6), dtype=np.float32)

    raw, rows, stats = _make_rows_from_labeled_q_pairs(
        checker,
        kinematics,
        q0_all,
        q1_all,
        d0_all,
        d1_all,
        n0_all,
        n1_all,
        clearance_margin_m,
        clearance_offset_m,
        clearance_label_floor=clearance_label_floor,
        clearance_label_power=clearance_label_power,
        sampling_mode="path_centered:stratified",
    )
    stats.update(
        {
            "attempts": float(attempts),
            "ik_seed_tries": float(seed_batches),
            "accepted_pairs": float(len(rows)),
            "acceptance_rate": 0.0 if attempts <= 0 else float(len(rows)) / float(attempts),
            "accepted_per_seed": 0.0 if seed_batches <= 0 else float(len(rows)) / float(seed_batches),
            "samples_per_seed": float(proposal_batch),
            "refined_q0": float(projected_q0),
            "refined_q1": 0.0,
            "anchor_seed_tries": float(attempts),
            "anchor_seed_success": float(len(rows)),
            "roi_seed_tries": 0.0,
            "workspace_seed_tries": 0.0,
            "roi_seed_success": 0.0,
            "workspace_seed_success": 0.0,
        }
    )
    return raw, rows, _finalize_sampler_stats(stats, rows, d0_all, d1_all, clearance_margin_m)


def _sample_workspace_seed_q(
    kinematics: UR5Kinematics,
    rng: np.random.Generator,
    seed_hint_q: np.ndarray | None = None,
    attempts: int = 24,
) -> np.ndarray | None:
    base_target = np.array([0.35, 0.0, 0.35], dtype=np.float64)
    workspace_min = np.array([0.20, -0.60, -0.05], dtype=np.float64)
    workspace_max = np.array([1.10, 0.60, 0.95], dtype=np.float64)
    local_spin = np.eye(3, dtype=np.float64)
    seed = (
        kinematics.clamp(np.asarray(seed_hint_q, dtype=np.float64))
        if seed_hint_q is not None
        else 0.5 * (kinematics.joint_min + kinematics.joint_max)
    )

    for _ in range(attempts):
        pos = rng.uniform(workspace_min, workspace_max)
        target = base_target + rng.uniform(
            low=np.array([-0.15, -0.20, -0.15], dtype=np.float64),
            high=np.array([0.25, 0.20, 0.20], dtype=np.float64),
        )
        rot = look_at_rotation(pos, target)
        spin = rng.uniform(-0.3 * np.pi, 0.3 * np.pi)
        c = float(np.cos(spin))
        s = float(np.sin(spin))
        local_spin[:, :] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot @ local_spin
        pose[:3, 3] = pos
        q = kinematics.solve_ik(pose, seed)
        if q is not None:
            return q
    return None


def _sample_anchor_seed_q(
    kinematics: UR5Kinematics,
    rng: np.random.Generator,
    anchor_qs: np.ndarray,
    attempts: int = 8,
) -> np.ndarray | None:
    anchors = np.asarray(anchor_qs, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 6 or len(anchors) == 0:
        return None
    for _ in range(attempts):
        q_anchor = anchors[int(rng.integers(0, len(anchors)))]
        q_anchor_n = kinematics.normalize(q_anchor)
        jitter = _sample_unit_ball_direction(rng, 6) * rng.uniform(0.005, 0.045)
        q = kinematics.denormalize(np.clip(q_anchor_n + jitter, -0.5, 0.5))
        if np.all(np.isfinite(q)):
            return kinematics.clamp(q)
    return None


def _sample_roi_focus_seed_q(
    kinematics: UR5Kinematics,
    rng: np.random.Generator,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    seed_hint_q: np.ndarray | None = None,
    attempts: int = 48,
) -> np.ndarray | None:
    roi_min = np.asarray(roi_min, dtype=np.float64).reshape(3)
    roi_max = np.asarray(roi_max, dtype=np.float64).reshape(3)
    roi_size = np.maximum(roi_max - roi_min, 1.0e-3)
    roi_center = 0.5 * (roi_min + roi_max)
    shrink = np.minimum(0.18 * roi_size, np.array([0.06, 0.08, 0.08], dtype=np.float64))
    inner_min = np.minimum(roi_min + shrink, roi_center - 1.0e-3)
    inner_max = np.maximum(roi_max - shrink, roi_center + 1.0e-3)
    base_seed = (
        kinematics.clamp(np.asarray(seed_hint_q, dtype=np.float64))
        if seed_hint_q is not None
        else 0.5 * (kinematics.joint_min + kinematics.joint_max)
    )
    alt_seed = 0.5 * (kinematics.joint_min + kinematics.joint_max)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_dirs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ]
    local_spin = np.eye(3, dtype=np.float64)

    for _ in range(attempts):
        target = rng.uniform(inner_min, inner_max)
        family = float(rng.uniform())
        if family < 0.65:
            outward = axis_dirs[int(rng.integers(0, len(axis_dirs)))]
            tangential = rng.normal(size=3)
            tangential -= np.dot(tangential, outward) * outward
            t_norm = np.linalg.norm(tangential)
            if t_norm > 1e-8:
                tangential /= t_norm
            else:
                tangential = np.zeros((3,), dtype=np.float64)
            lateral_mag = float(rng.uniform(0.0, 0.18 * float(np.max(roi_size))))
            standoff = float(rng.uniform(0.08, 0.24 + 0.12 * float(np.max(roi_size))))
            pos = target + outward * standoff + tangential * lateral_mag
            look_target = target
        elif family < 0.90:
            opening_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            pos = target - opening_axis * float(rng.uniform(0.04, 0.14))
            pos[1] += float(rng.uniform(-0.18, 0.18)) * roi_size[1]
            pos[2] += float(rng.uniform(-0.18, 0.18)) * roi_size[2]
            look_target = np.minimum(np.maximum(target + opening_axis * float(rng.uniform(0.06, 0.16)), inner_min), inner_max)
        else:
            pos = target.copy()
            pos[0] -= float(rng.uniform(0.04, 0.10))
            look_target = np.minimum(np.maximum(target + np.array([0.10, 0.0, 0.0], dtype=np.float64), inner_min), inner_max)

        rot = look_at_rotation(pos, look_target, world_up)
        spin = float(rng.uniform(-np.pi, np.pi))
        c = float(np.cos(spin))
        s = float(np.sin(spin))
        local_spin[:, :] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot @ local_spin
        pose[:3, 3] = pos

        for ik_seed in (base_seed, alt_seed):
            q = kinematics.solve_ik(pose, ik_seed)
            if q is not None:
                return q
    return None


def _sample_training_seed_q(
    kinematics: UR5Kinematics,
    rng: np.random.Generator,
    roi_min: np.ndarray | None = None,
    roi_max: np.ndarray | None = None,
    seed_hint_q: np.ndarray | None = None,
    anchor_qs: np.ndarray | None = None,
    anchor_seed_probability: float = 0.0,
) -> tuple[np.ndarray | None, str, bool]:
    if anchor_qs is not None and len(anchor_qs) > 0 and float(rng.uniform()) < float(np.clip(anchor_seed_probability, 0.0, 1.0)):
        q = _sample_anchor_seed_q(kinematics, rng, anchor_qs)
        if q is not None:
            return q, "anchor", False
    if roi_min is not None and roi_max is not None:
        q = _sample_roi_focus_seed_q(
            kinematics,
            rng,
            roi_min=roi_min,
            roi_max=roi_max,
            seed_hint_q=seed_hint_q,
        )
        if q is not None:
            return q, "roi", True
        return _sample_workspace_seed_q(kinematics, rng, seed_hint_q=seed_hint_q), "workspace", True
    return _sample_workspace_seed_q(kinematics, rng, seed_hint_q=seed_hint_q), "workspace", False


def _sample_joint_seed_q(
    kinematics: UR5Kinematics,
    rng: np.random.Generator,
    seed_hint_q: np.ndarray | None = None,
    anchor_qs: np.ndarray | None = None,
    anchor_seed_probability: float = 0.0,
) -> tuple[np.ndarray | None, str, bool]:
    anchors = np.asarray(anchor_qs, dtype=np.float64) if anchor_qs is not None else np.zeros((0, 6), dtype=np.float64)
    if anchors.ndim == 2 and anchors.shape[1] == 6 and len(anchors) > 0 and float(rng.uniform()) < float(np.clip(anchor_seed_probability, 0.0, 1.0)):
        q = _sample_anchor_seed_q(kinematics, rng, anchors)
        if q is not None:
            return q, "anchor", False
    if seed_hint_q is not None:
        seed = kinematics.clamp(np.asarray(seed_hint_q, dtype=np.float64))
        qn = kinematics.normalize(seed).astype(np.float64)
        qn = np.clip(qn + _sample_unit_ball_direction(rng, 6) * float(rng.uniform(0.0, 0.20)), -0.5, 0.5)
        return kinematics.denormalize(qn), "joint_local", False
    q = rng.uniform(kinematics.joint_min, kinematics.joint_max).astype(np.float64)
    return kinematics.clamp(q), "joint_random", False


def sample_cspace_training_batch(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    num_pairs: int,
    clearance_margin_m: float,
    clearance_offset_m: float,
    rng: np.random.Generator,
    samples_per_seed: int = 24,
    roi_min: np.ndarray | None = None,
    roi_max: np.ndarray | None = None,
    seed_hint_q: np.ndarray | None = None,
    anchor_qs: np.ndarray | None = None,
    anchor_seed_probability: float = 0.0,
    clearance_label_floor: float = 0.0,
    clearance_label_power: float = 1.0,
    sampling_mode: str = "joint_local_6d",
    proposal_batch_size: int = 256,
    near_boundary_only: bool = False,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if sampling_mode == "legacy_pair_6d":
        raw_rows = []
        pair_rows = []
        attempts = 0
        ik_seed_tries = 0
        roi_pose_tries = 0
        roi_pose_success = 0
        workspace_pose_tries = 0
        workspace_pose_success = 0
        d0_values: list[float] = []
        d1_values: list[float] = []
        report_every_attempts = max(200, num_pairs // 4)
        report_every_accepts = max(32, num_pairs // 8)
        proposal_batch_size = max(32, int(proposal_batch_size))
        max_attempts = max(num_pairs * 80, num_pairs + 1)
        while len(pair_rows) < num_pairs and attempts < max_attempts:
            remaining = max_attempts - attempts
            pose_batch_size = min(proposal_batch_size, remaining)
            pose_batch, use_roi_batch = _sample_legacy_pair_pose_batch(
                rng,
                pose_batch_size,
                roi_min=roi_min,
                roi_max=roi_max,
            )
            attempts += pose_batch_size
            roi_pose_tries += int(np.count_nonzero(use_roi_batch))
            workspace_pose_tries += int(pose_batch_size - np.count_nonzero(use_roi_batch))
            ik_seed_tries += pose_batch_size
            q0n_batch = _solve_ik_pose_batch_gpu(kinematics, pose_batch)
            if len(q0n_batch) == 0:
                if progress_cb is not None and attempts % report_every_attempts == 0:
                    progress_cb(attempts, len(pair_rows), ik_seed_tries)
                continue
            roi_pose_success += int(min(len(q0n_batch), np.count_nonzero(use_roi_batch)))
            workspace_pose_success += int(max(0, len(q0n_batch) - min(len(q0n_batch), np.count_nonzero(use_roi_batch))))
            q0_batch = kinematics.denormalize(q0n_batch).astype(np.float32)
            d0_batch, n0_q_batch = checker.clearance_and_normal_batch(q0_batch)
            valid_q0 = (d0_batch >= clearance_offset_m) & (np.linalg.norm(n0_q_batch, axis=1) > 1e-8)
            if near_boundary_only:
                valid_q0 &= d0_batch <= clearance_margin_m
                keep_prob = _q0_low_speed_keep_probability(d0_batch, clearance_margin_m, clearance_offset_m)
                valid_q0 &= rng.random(len(d0_batch)) <= keep_prob
            if not np.any(valid_q0):
                if progress_cb is not None and attempts % report_every_attempts == 0:
                    progress_cb(attempts, len(pair_rows), ik_seed_tries)
                continue
            q0n_valid = q0n_batch[valid_q0]
            q0_valid = q0_batch[valid_q0]
            d0_valid = d0_batch[valid_q0]
            n0_q_valid = n0_q_batch[valid_q0]
            q1n_candidates = []
            q0_keep_idx = []
            for idx, q0n in enumerate(q0n_valid):
                direction = _sample_unit_ball_direction(rng, 6)
                radius = float(rng.uniform(0.0, 0.5))
                q1n = q0n.astype(np.float64) + direction * radius
                if np.any(q1n < -0.5) or np.any(q1n > 0.5):
                    continue
                q1n_candidates.append(q1n.astype(np.float32))
                q0_keep_idx.append(idx)
            if not q1n_candidates:
                if progress_cb is not None and attempts % report_every_attempts == 0:
                    progress_cb(attempts, len(pair_rows), ik_seed_tries)
                continue
            q1n_valid = np.asarray(q1n_candidates, dtype=np.float32)
            q0n_valid = q0n_valid[q0_keep_idx]
            q0_valid = q0_valid[q0_keep_idx]
            d0_valid = d0_valid[q0_keep_idx]
            n0_q_valid = n0_q_valid[q0_keep_idx]
            q1_valid = kinematics.denormalize(q1n_valid).astype(np.float32)
            d1_valid, n1_q_batch = checker.clearance_and_normal_batch(q1_valid)
            for q0n, q1n, q0, q1, d0, d1, n0_q, n1_q in zip(
                q0n_valid, q1n_valid, q0_valid, q1_valid, d0_valid, d1_valid, n0_q_valid, n1_q_batch
            ):
                if np.linalg.norm(n1_q) < 1e-8:
                    continue
                y0 = _clearance_to_speed_label(
                    float(d0),
                    clearance_margin_m,
                    clearance_offset_m,
                    label_floor=clearance_label_floor,
                    label_power=clearance_label_power,
                )
                y1 = _clearance_to_speed_label(
                    float(d1),
                    clearance_margin_m,
                    clearance_offset_m,
                    label_floor=clearance_label_floor,
                    label_power=clearance_label_power,
                )
                n0 = _physical_gradient_to_normalized_direction(kinematics, n0_q)
                n1 = _physical_gradient_to_normalized_direction(kinematics, n1_q)
                if np.linalg.norm(n0) < 1e-8 or np.linalg.norm(n1) < 1e-8:
                    continue
                cp0 = _normalized_contact_point(kinematics, q0, n0, float(d0), clearance_margin_m)
                cp1 = _normalized_contact_point(kinematics, q1, n1, float(d1), clearance_margin_m)
                raw_rows.append(np.concatenate((q0n.astype(np.float32), cp0.astype(np.float32)), axis=0))
                raw_rows.append(np.concatenate((q1n.astype(np.float32), cp1.astype(np.float32)), axis=0))
                pair_rows.append(
                    np.concatenate((q0n.astype(np.float32), q1n.astype(np.float32), [y0, y1], n0.astype(np.float32), n1.astype(np.float32)), axis=0)
                )
                d0_values.append(float(d0))
                d1_values.append(float(d1))
                if len(pair_rows) >= num_pairs:
                    break
            if progress_cb is not None and (
                len(pair_rows) % report_every_accepts == 0 or attempts % report_every_attempts == 0
            ):
                progress_cb(attempts, len(pair_rows), ik_seed_tries)
        stats = {
            "sampling_mode": f"{sampling_mode}:{'near' if near_boundary_only else 'broad'}",
            "attempts": float(attempts),
            "ik_seed_tries": float(ik_seed_tries),
            "accepted_pairs": float(len(pair_rows)),
            "acceptance_rate": 0.0 if attempts <= 0 else float(len(pair_rows)) / float(attempts),
            "accepted_per_seed": 0.0 if ik_seed_tries <= 0 else float(len(pair_rows)) / float(ik_seed_tries),
            "samples_per_seed": float(proposal_batch_size),
            "refined_q0": 0.0,
            "anchor_seed_tries": 0.0,
            "anchor_seed_success": 0.0,
            "roi_seed_tries": float(roi_pose_tries),
            "workspace_seed_tries": float(workspace_pose_tries),
            "roi_seed_success": float(roi_pose_success),
            "workspace_seed_success": float(workspace_pose_success),
        }
        rows = np.asarray(pair_rows, dtype=np.float32) if len(pair_rows) else np.zeros((0, 26), dtype=np.float32)
        stats = _finalize_sampler_stats(stats, rows, d0_values, d1_values, clearance_margin_m)
        if not pair_rows:
            return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), stats
        return np.asarray(raw_rows, dtype=np.float32), rows, stats

    if sampling_mode in ("joint_local_6d", "joint_noik_6d"):
        raw_fast, rows_fast, stats_fast = _sample_joint_local_stratified_batch(
            checker,
            kinematics,
            num_pairs,
            clearance_margin_m,
            clearance_offset_m,
            rng,
            samples_per_seed=samples_per_seed,
            seed_hint_q=seed_hint_q,
            anchor_qs=anchor_qs,
            anchor_seed_probability=anchor_seed_probability,
            clearance_label_floor=clearance_label_floor,
            clearance_label_power=clearance_label_power,
            sampling_mode=sampling_mode,
            proposal_batch_size=proposal_batch_size,
            near_boundary_only=near_boundary_only,
            progress_cb=progress_cb,
        )
        if len(rows_fast) >= max(1, int(0.75 * max(1, num_pairs))):
            return raw_fast, rows_fast, stats_fast
        # Fall through to the older rejection sampler only if the stratified
        # pass cannot find enough valid rows in this scene.

    raw_rows = []
    pair_rows = []

    attempts = 0
    ik_seed_tries = 0
    anchor_seed_tries = 0
    anchor_seed_success = 0
    roi_seed_tries = 0
    workspace_seed_tries = 0
    roi_seed_success = 0
    workspace_seed_success = 0
    d0_values: list[float] = []
    d1_values: list[float] = []
    report_every_attempts = max(200, num_pairs // 4)
    report_every_accepts = max(32, num_pairs // 8)
    samples_per_seed = max(1, int(samples_per_seed))
    while len(pair_rows) < num_pairs and attempts < num_pairs * 60:
        if sampling_mode in ("joint_local_6d", "joint_noik_6d"):
            q_seed, seed_kind, attempted_roi = _sample_joint_seed_q(
                kinematics,
                rng,
                seed_hint_q=seed_hint_q if (near_boundary_only or float(rng.uniform()) < 0.40) else None,
                anchor_qs=anchor_qs,
                anchor_seed_probability=anchor_seed_probability,
            )
        else:
            q_seed, seed_kind, attempted_roi = _sample_training_seed_q(
                kinematics,
                rng,
                roi_min=roi_min,
                roi_max=roi_max,
                seed_hint_q=seed_hint_q,
                anchor_qs=anchor_qs,
                anchor_seed_probability=anchor_seed_probability,
            )
        ik_seed_tries += 1
        if seed_kind == "anchor":
            anchor_seed_tries += 1
        if attempted_roi:
            roi_seed_tries += 1
        if seed_kind in ("workspace", "joint_local", "joint_random"):
            workspace_seed_tries += 1
        if q_seed is None:
            if progress_cb is not None and ik_seed_tries % 8 == 0:
                progress_cb(attempts, len(pair_rows), ik_seed_tries)
            continue
        if seed_kind == "anchor":
            anchor_seed_success += 1
        elif seed_kind == "roi":
            roi_seed_success += 1
        else:
            workspace_seed_success += 1
        q_seed_n = kinematics.normalize(q_seed).astype(np.float64)
        q0n_candidates = []
        for _ in range(samples_per_seed):
            if len(pair_rows) >= num_pairs or attempts >= num_pairs * 60:
                break
            attempts += 1
            q0_dir = _sample_unit_ball_direction(rng, 6)
            if not near_boundary_only:
                q0_radius = rng.uniform(0.0, 0.50)
            elif seed_kind == "anchor":
                q0_radius = rng.uniform(0.006, 0.035)
            elif seed_kind == "roi":
                q0_radius = rng.uniform(0.008, 0.06)
            else:
                q0_radius = rng.uniform(0.008, 0.09)
            q0n = np.clip(q_seed_n + q0_dir * q0_radius, -0.5, 0.5)
            q0n_candidates.append(q0n)
        if not q0n_candidates:
            continue
        q0n_batch = np.asarray(q0n_candidates, dtype=np.float32)
        q0_batch = kinematics.denormalize(q0n_batch).astype(np.float32)
        d0_batch, n0_q_batch = checker.clearance_and_normal_batch(q0_batch)

        valid_q0 = (d0_batch >= clearance_offset_m) & (np.linalg.norm(n0_q_batch, axis=1) > 1e-8)
        if near_boundary_only:
            valid_q0 &= d0_batch <= clearance_margin_m
            keep_prob = _q0_low_speed_keep_probability(d0_batch, clearance_margin_m, clearance_offset_m)
            valid_q0 &= rng.random(len(d0_batch)) <= keep_prob
        if not np.any(valid_q0):
            if progress_cb is not None and attempts % report_every_attempts == 0:
                progress_cb(attempts, len(pair_rows), ik_seed_tries)
            continue
        q0_valid = q0_batch[valid_q0]
        q0n_valid = kinematics.normalize(q0_valid).astype(np.float32)
        d0_valid = d0_batch[valid_q0]
        n0_q_valid = n0_q_batch[valid_q0]

        q1n_valid = []
        for q0n in q0n_valid:
            direction = _sample_unit_ball_direction(rng, 6)
            if not near_boundary_only:
                radius = rng.uniform(0.0, 0.50)
            elif seed_kind == "anchor":
                radius = rng.uniform(0.01, 0.08)
            elif seed_kind == "roi":
                radius = rng.uniform(0.02, 0.12)
            else:
                radius = rng.uniform(0.02, 0.18)
            q1n_valid.append(np.clip(q0n + direction * radius, -0.5, 0.5))
        q1n_valid = np.asarray(q1n_valid, dtype=np.float32)
        q1_valid = kinematics.denormalize(q1n_valid).astype(np.float32)
        d1_valid, n1_q_batch = checker.clearance_and_normal_batch(q1_valid)

        q1n_valid = kinematics.normalize(q1_valid).astype(np.float32)

        for q0n, q1n, q0, q1, d0, d1, n0_q, n1_q in zip(
            q0n_valid, q1n_valid, q0_valid, q1_valid, d0_valid, d1_valid, n0_q_valid, n1_q_batch
        ):
            if np.linalg.norm(n0_q) < 1e-8 or np.linalg.norm(n1_q) < 1e-8:
                continue
            y0 = _clearance_to_speed_label(
                d0,
                clearance_margin_m,
                clearance_offset_m,
                label_floor=clearance_label_floor,
                label_power=clearance_label_power,
            )
            y1 = _clearance_to_speed_label(
                d1,
                clearance_margin_m,
                clearance_offset_m,
                label_floor=clearance_label_floor,
                label_power=clearance_label_power,
            )
            n0 = _physical_gradient_to_normalized_direction(kinematics, n0_q)
            n1 = _physical_gradient_to_normalized_direction(kinematics, n1_q)
            if np.linalg.norm(n0) < 1e-8 or np.linalg.norm(n1) < 1e-8:
                continue
            cp0 = _normalized_contact_point(kinematics, q0, n0, d0, clearance_margin_m)
            cp1 = _normalized_contact_point(kinematics, q1, n1, d1, clearance_margin_m)
            raw_rows.append(np.concatenate((q0n, cp0), axis=0).astype(np.float32))
            raw_rows.append(np.concatenate((q1n, cp1), axis=0).astype(np.float32))
            pair_rows.append(np.concatenate((q0n, q1n, [y0, y1], n0, n1), axis=0).astype(np.float32))
            d0_values.append(float(d0))
            d1_values.append(float(d1))
            if len(pair_rows) >= num_pairs:
                break
        if progress_cb is not None and (
            len(pair_rows) % report_every_accepts == 0 or attempts % report_every_attempts == 0
        ):
            progress_cb(attempts, len(pair_rows), ik_seed_tries)

    stats = {
        "sampling_mode": f"{sampling_mode}:{'near' if near_boundary_only else 'broad'}",
        "attempts": float(attempts),
        "ik_seed_tries": float(ik_seed_tries),
        "accepted_pairs": float(len(pair_rows)),
        "acceptance_rate": 0.0 if attempts <= 0 else float(len(pair_rows)) / float(attempts),
        "accepted_per_seed": 0.0 if ik_seed_tries <= 0 else float(len(pair_rows)) / float(ik_seed_tries),
        "samples_per_seed": float(samples_per_seed),
        "refined_q0": 0.0,
        "refined_q1": 0.0,
        "anchor_seed_tries": float(anchor_seed_tries),
        "anchor_seed_success": float(anchor_seed_success),
        "roi_seed_tries": float(roi_seed_tries),
        "workspace_seed_tries": float(workspace_seed_tries),
        "roi_seed_success": float(roi_seed_success),
        "workspace_seed_success": float(workspace_seed_success),
    }
    rows = np.asarray(pair_rows, dtype=np.float32) if len(pair_rows) else np.zeros((0, 26), dtype=np.float32)
    stats = _finalize_sampler_stats(stats, rows, d0_values, d1_values, clearance_margin_m)
    if not pair_rows:
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), stats
    return np.asarray(raw_rows, dtype=np.float32), rows, stats
