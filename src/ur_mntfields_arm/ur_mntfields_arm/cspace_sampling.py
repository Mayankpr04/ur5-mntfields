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


def _speed_label_to_clearance(
    speed_label: float,
    clearance_margin_m: float,
    clearance_offset_m: float,
    label_floor: float = 0.0,
    label_power: float = 1.0,
) -> float:
    """Invert the clearance label map for a bounded free-space clearance."""
    margin = max(float(clearance_margin_m), 1.0e-6)
    offset = float(np.clip(clearance_offset_m, 0.0, margin - 1.0e-6))
    floor = float(np.clip(label_floor, 0.0, 1.0))
    if floor >= 1.0 - 1.0e-6:
        return margin
    alpha = float(np.clip((float(speed_label) - floor) / (1.0 - floor), 0.0, 1.0))
    clearance = margin * alpha ** (1.0 / max(1.0e-6, float(label_power)))
    return float(np.clip(clearance, offset, margin))


def _sample_near_target_clearances(
    rng: np.random.Generator,
    count: int,
    clearance_margin_m: float,
    clearance_offset_m: float,
    *,
    label_floor: float = 0.0,
    label_power: float = 1.0,
    critical_only: bool = False,
) -> np.ndarray:
    """Sample a near band with enough labels in the field's low-speed region.

    Uniform clearance targets leave the first 20% of speed labels rare after
    broad q1 pairing. Biasing only the projected q0 endpoint preserves broad
    pair coverage while making the safety-critical low-speed band observable.
    """
    total = max(0, int(count))
    if total == 0:
        return np.zeros((0,), dtype=np.float32)
    margin = max(float(clearance_margin_m), 1.0e-6)
    offset = float(np.clip(clearance_offset_m, 0.0, margin - 1.0e-6))
    low_hi = _speed_label_to_clearance(
        0.20,
        margin,
        offset,
        label_floor=label_floor,
        label_power=label_power,
    )
    if critical_only:
        return rng.uniform(offset, max(offset, low_hi), size=total).astype(np.float32)
    mid_hi = _speed_label_to_clearance(
        0.60,
        margin,
        offset,
        label_floor=label_floor,
        label_power=label_power,
    )
    low_count = min(total, int(round(0.65 * total)))
    mid_count = min(total - low_count, int(round(0.25 * total)))
    high_count = total - low_count - mid_count
    targets = np.empty((total,), dtype=np.float32)
    start = 0
    if low_count:
        targets[start : start + low_count] = rng.uniform(offset, max(offset, low_hi), size=low_count)
        start += low_count
    if mid_count:
        targets[start : start + mid_count] = rng.uniform(max(offset, low_hi), max(low_hi, mid_hi), size=mid_count)
        start += mid_count
    if high_count:
        targets[start:] = rng.uniform(max(offset, mid_hi), margin, size=high_count)
    rng.shuffle(targets)
    return targets


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
            clearance_offset_m,
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
    stats = _finalize_sampler_stats(
        stats, rows, d0_values, d1_values, clearance_margin_m, clearance_offset_m
    )
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
    return raw, rows, _finalize_sampler_stats(
        stats, rows, d0_values, d1_values, clearance_margin_m, clearance_offset_m
    )


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
            np.where(d0 <= 0.20 * margin)[0],
            np.where((d0 > 0.20 * margin) & (d0 <= 0.60 * margin))[0],
            np.where((d0 > 0.60 * margin) & (d0 <= margin))[0],
        )
        quotas = (0.55, 0.30, 0.15)
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


def _project_toward_clearance_band(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    q_seed: np.ndarray,
    clearance_seed: np.ndarray,
    normal_seed_q: np.ndarray,
    target_clearance: np.ndarray,
    minimum_clearance_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move safe joint states toward a requested clearance using re-queries.

    A short physical-joint probe measures metres per radian along the returned
    normal before proposing the final step. Missed low-speed targets receive a
    bounded local re-probe; ordinary and already-correct proposals still use
    only the two-query slope estimate.
    """
    q_seed = np.asarray(q_seed, dtype=np.float32)
    clearance_seed = np.asarray(clearance_seed, dtype=np.float32).reshape(-1)
    normal_seed_q = np.asarray(normal_seed_q, dtype=np.float32)
    target_clearance = np.asarray(target_clearance, dtype=np.float32).reshape(-1)
    count = min(len(q_seed), len(clearance_seed), len(normal_seed_q), len(target_clearance))
    if count <= 0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
        )

    q_seed = q_seed[:count]
    clearance_seed = clearance_seed[:count]
    normal_seed_q = normal_seed_q[:count]
    target_clearance = target_clearance[:count]
    best_q = q_seed.copy()
    best_d = clearance_seed.copy()
    best_n = normal_seed_q.copy()
    best_error = np.abs(best_d - target_clearance)

    # The checker normal supplies a direction, not a metres-to-radians scale.
    # Estimate that scale from one bounded probe, then re-query the final state.
    probe_step_rad = 0.12
    probe_q = np.clip(
        q_seed - probe_step_rad * normal_seed_q,
        kinematics.joint_min,
        kinematics.joint_max,
    ).astype(np.float32)
    probe_d, _probe_n = checker.clearance_and_normal_batch(probe_q)
    probe_d = np.asarray(probe_d, dtype=np.float32).reshape(-1)
    slope = (clearance_seed - probe_d) / probe_step_rad
    valid_slope = np.isfinite(probe_d) & np.isfinite(slope) & (slope > 1.0e-5)
    estimated_step = np.full((count,), 0.60, dtype=np.float32)
    estimated_step[valid_slope] = np.clip(
        (clearance_seed[valid_slope] - target_clearance[valid_slope]) / slope[valid_slope],
        0.02,
        0.95,
    )
    candidate_q = np.clip(
        q_seed - estimated_step[:, None] * normal_seed_q,
        kinematics.joint_min,
        kinematics.joint_max,
    ).astype(np.float32)
    candidate_d, candidate_n = checker.clearance_and_normal_batch(candidate_q)
    candidate_d = np.asarray(candidate_d, dtype=np.float32).reshape(-1)
    candidate_n = np.asarray(candidate_n, dtype=np.float32)
    normal_ok = np.linalg.norm(candidate_n, axis=1) > 1.0e-8
    candidate_ok = np.isfinite(candidate_d) & normal_ok & (candidate_d >= float(minimum_clearance_m))
    candidate_error = np.abs(candidate_d - target_clearance)
    improve = candidate_ok & (candidate_error < best_error)
    if np.any(improve):
        best_q[improve] = candidate_q[improve]
        best_d[improve] = candidate_d[improve]
        best_n[improve] = candidate_n[improve]
        best_error[improve] = candidate_error[improve]

    # The seed-normal slope can be inaccurate after the closest robot sphere
    # changes. Refine only the safety-critical targets that it missed. A
    # local probe at the candidate supplies a better metres-per-radian slope
    # without returning to the old many-step clearance ladder.
    low_target_limit = max(
        float(minimum_clearance_m) + 0.015,
        0.25 * float(np.max(target_clearance)),
    )
    # ``target_clearance`` is sampled from the label band, so use the actual
    # requested target rather than a global fixed threshold when deciding
    # whether the additional work is warranted.
    low_target_mask = target_clearance <= max(float(minimum_clearance_m) + 0.015, low_target_limit)
    overshot = (
        np.isfinite(candidate_d)
        & (candidate_d < float(minimum_clearance_m))
        & (clearance_seed > target_clearance + 1.0e-5)
        & low_target_mask
    )

    for _ in range(3):
        missed_high = (
            np.isfinite(best_d)
            & (best_d > target_clearance + 0.004)
            & (best_d < clearance_seed - 1.0e-5)
            & (np.linalg.norm(best_n, axis=1) > 1.0e-8)
            & low_target_mask
        )
        if not np.any(missed_high):
            break
        refine_idx = np.where(missed_high)[0]
        local_probe_step = 0.12
        local_probe_q = np.clip(
            best_q[refine_idx] - local_probe_step * best_n[refine_idx],
            kinematics.joint_min,
            kinematics.joint_max,
        ).astype(np.float32)
        local_probe_d, _local_probe_n = checker.clearance_and_normal_batch(local_probe_q)
        local_probe_d = np.asarray(local_probe_d, dtype=np.float32).reshape(-1)
        actual_step = np.linalg.norm(local_probe_q - best_q[refine_idx], axis=1)
        local_slope = (best_d[refine_idx] - local_probe_d) / np.maximum(actual_step, 1.0e-5)
        valid_local_slope = np.isfinite(local_probe_d) & np.isfinite(local_slope) & (local_slope > 1.0e-5)
        correction = np.full((len(refine_idx),), 0.12, dtype=np.float32)
        correction[valid_local_slope] = np.clip(
            1.05 * (best_d[refine_idx][valid_local_slope] - target_clearance[refine_idx][valid_local_slope])
            / local_slope[valid_local_slope],
            0.01,
            0.45,
        )
        refined_q = np.clip(
            best_q[refine_idx] - correction[:, None] * best_n[refine_idx],
            kinematics.joint_min,
            kinematics.joint_max,
        ).astype(np.float32)
        refined_d, refined_n = checker.clearance_and_normal_batch(refined_q)
        refined_d = np.asarray(refined_d, dtype=np.float32).reshape(-1)
        refined_n = np.asarray(refined_n, dtype=np.float32)
        refined_ok = np.isfinite(refined_d) & (refined_d >= float(minimum_clearance_m)) & (
            np.linalg.norm(refined_n, axis=1) > 1.0e-8
        )
        refined_error = np.abs(refined_d - target_clearance[refine_idx])
        refined_improve = refined_ok & (refined_error < best_error[refine_idx])
        if not np.any(refined_improve):
            break
        dst_idx = refine_idx[refined_improve]
        best_q[dst_idx] = refined_q[refined_improve]
        best_d[dst_idx] = refined_d[refined_improve]
        best_n[dst_idx] = refined_n[refined_improve]
        best_error[dst_idx] = refined_error[refined_improve]

    # If the seed-normal proposal crossed the positive clearance floor, use
    # its safe-to-unsafe bracket to re-query one interpolated configuration.
    # This keeps the label in Q_free instead of discarding the useful near
    # contact direction and falling back to the distant seed.
    if np.any(overshot):
        refine_idx = np.where(overshot)[0]
        denom = clearance_seed[refine_idx] - candidate_d[refine_idx]
        fraction = (clearance_seed[refine_idx] - target_clearance[refine_idx]) / np.maximum(denom, 1.0e-5)
        fraction = np.clip(fraction, 0.05, 0.95).astype(np.float32)
        refined_q = np.clip(
            q_seed[refine_idx] + fraction[:, None] * (candidate_q[refine_idx] - q_seed[refine_idx]),
            kinematics.joint_min,
            kinematics.joint_max,
        ).astype(np.float32)
        refined_d, refined_n = checker.clearance_and_normal_batch(refined_q)
        refined_d = np.asarray(refined_d, dtype=np.float32).reshape(-1)
        refined_n = np.asarray(refined_n, dtype=np.float32)
        refined_ok = np.isfinite(refined_d) & (refined_d >= float(minimum_clearance_m)) & (
            np.linalg.norm(refined_n, axis=1) > 1.0e-8
        )
        refined_error = np.abs(refined_d - target_clearance[refine_idx])
        refined_improve = refined_ok & (refined_error < best_error[refine_idx])
        if np.any(refined_improve):
            dst_idx = refine_idx[refined_improve]
            best_q[dst_idx] = refined_q[refined_improve]
            best_d[dst_idx] = refined_d[refined_improve]
            best_n[dst_idx] = refined_n[refined_improve]
            best_error[dst_idx] = refined_error[refined_improve]
    return best_q.astype(np.float32), best_d.astype(np.float32), best_n.astype(np.float32)


def _project_out_of_collision_to_clearance_band(
    checker: UR5PointCloudCollisionChecker,
    kinematics: UR5Kinematics,
    q_seed: np.ndarray,
    normal_seed_q: np.ndarray,
    target_clearance: np.ndarray,
    minimum_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project colliding proposals onto a verified free-side C-space shell.

    Clearance is clamped at zero inside collision, so a local derivative alone
    cannot estimate the penetration depth.  Expand along the checker's outward
    joint-space normal until a positive-clearance bracket is found, then query
    an interpolation toward the requested shell.  Unprojectable rows remain at
    zero clearance and are rejected by the caller.
    """
    q_seed = np.asarray(q_seed, dtype=np.float32)
    normal_seed_q = np.asarray(normal_seed_q, dtype=np.float32)
    target_clearance = np.asarray(target_clearance, dtype=np.float32).reshape(-1)
    count = min(len(q_seed), len(normal_seed_q), len(target_clearance))
    if count <= 0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
        )

    q_seed = q_seed[:count]
    normal_seed_q = normal_seed_q[:count]
    target_clearance = target_clearance[:count]
    best_q = q_seed.copy()
    best_d = np.zeros((count,), dtype=np.float32)
    best_n = normal_seed_q.copy()
    best_error = np.full((count,), np.inf, dtype=np.float32)
    bracket_q = np.zeros_like(q_seed)
    bracket_d = np.zeros((count,), dtype=np.float32)
    bracket_n = np.zeros_like(normal_seed_q)
    bracket_found = np.zeros((count,), dtype=bool)

    for step_rad in (0.04, 0.08, 0.16, 0.32, 0.64, 0.95):
        unresolved = ~bracket_found
        if not np.any(unresolved):
            break
        idx = np.where(unresolved)[0]
        candidate_q = np.clip(
            q_seed[idx] + float(step_rad) * normal_seed_q[idx],
            kinematics.joint_min,
            kinematics.joint_max,
        ).astype(np.float32)
        candidate_d, candidate_n = checker.clearance_and_normal_batch(candidate_q)
        candidate_d = np.asarray(candidate_d, dtype=np.float32).reshape(-1)
        candidate_n = np.asarray(candidate_n, dtype=np.float32)
        found = (
            np.isfinite(candidate_d)
            & (candidate_d >= target_clearance[idx])
            & (np.linalg.norm(candidate_n, axis=1) > 1.0e-8)
        )
        if np.any(found):
            dst = idx[found]
            bracket_q[dst] = candidate_q[found]
            bracket_d[dst] = candidate_d[found]
            bracket_n[dst] = candidate_n[found]
            bracket_found[dst] = True

    if np.any(bracket_found):
        idx = np.where(bracket_found)[0]
        low_q = q_seed[idx].copy()
        high_q = bracket_q[idx].copy()
        high_d = bracket_d[idx].copy()
        high_n = bracket_n[idx].copy()
        # Clearance is flat at zero through the penetrated interval, so linear
        # distance scaling is invalid. Bisect the actual collision/free bracket
        # and keep the smallest queried state that reaches the target shell.
        for _ in range(6):
            mid_q = (0.5 * (low_q + high_q)).astype(np.float32)
            mid_d, mid_n = checker.clearance_and_normal_batch(mid_q)
            mid_d = np.asarray(mid_d, dtype=np.float32).reshape(-1)
            mid_n = np.asarray(mid_n, dtype=np.float32)
            mid_free = np.isfinite(mid_d) & (mid_d >= target_clearance[idx])
            if np.any(mid_free):
                high_q[mid_free] = mid_q[mid_free]
                high_d[mid_free] = mid_d[mid_free]
                high_n[mid_free] = mid_n[mid_free]
            if np.any(~mid_free):
                low_q[~mid_free] = mid_q[~mid_free]
        shell_q = high_q
        shell_d = high_d
        shell_n = high_n
        shell_ok = (
            np.isfinite(shell_d)
            & (shell_d >= float(minimum_clearance_m))
            & (np.linalg.norm(shell_n, axis=1) > 1.0e-8)
        )
        shell_error = np.abs(shell_d - target_clearance[idx])
        if np.any(shell_ok):
            dst = idx[shell_ok]
            best_q[dst] = shell_q[shell_ok]
            best_d[dst] = shell_d[shell_ok]
            best_n[dst] = shell_n[shell_ok]
            best_error[dst] = shell_error[shell_ok]

        # Retain the verified outer bracket if a degenerate normal invalidates
        # the final bisection point.
        missing = ~shell_ok
        if np.any(missing):
            dst = idx[missing]
            outer_error = np.abs(bracket_d[dst] - target_clearance[dst])
            use_outer = bracket_d[dst] >= float(minimum_clearance_m)
            if np.any(use_outer):
                out_dst = dst[use_outer]
                best_q[out_dst] = bracket_q[out_dst]
                best_d[out_dst] = bracket_d[out_dst]
                best_n[out_dst] = bracket_n[out_dst]
                best_error[out_dst] = outer_error[use_outer]

    return best_q.astype(np.float32), best_d.astype(np.float32), best_n.astype(np.float32)


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
    roi_min: np.ndarray | None,
    roi_max: np.ndarray | None,
    roi_seed_fraction: float,
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
    roi_min_arr = None
    roi_max_arr = None
    if roi_min is not None and roi_max is not None:
        candidate_min = np.asarray(roi_min, dtype=np.float64).reshape(3)
        candidate_max = np.asarray(roi_max, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(candidate_min)) and np.all(np.isfinite(candidate_max)) and np.all(candidate_max > candidate_min):
            roi_min_arr = candidate_min
            roi_max_arr = candidate_max
    roi_seed_fraction = float(np.clip(roi_seed_fraction, 0.0, 1.0))

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
    roi_seed_tries = 0
    roi_seed_success = 0
    workspace_seed_tries = 0
    workspace_seed_success = 0
    projected_q0 = 0

    # Build a small seed bank once per sampling call. Reusing it across
    # clearance microbatches keeps ROI conditioning inexpensive even when the
    # near-clearance sampler needs many proposal rounds.
    roi_seed_qn_bank = np.zeros((0, 6), dtype=np.float32)
    seed_width = max(1, int(samples_per_seed))
    if roi_min_arr is not None and roi_max_arr is not None and roi_seed_fraction > 0.0:
        desired_roi_rows = max(1, int(np.ceil(target * roi_seed_fraction)))
        # Match the bounded v6 online budget. The subsequent 24-seed expansion
        # increased sampling cost without improving the v8 field calibration.
        seed_budget = min(8, max(1, int(np.ceil(desired_roi_rows / seed_width))))
        roi_seed_qn: list[np.ndarray] = []
        for _ in range(seed_budget):
            roi_seed_tries += 1
            roi_seed = _sample_roi_focus_seed_q(
                kinematics,
                rng,
                roi_min_arr,
                roi_max_arr,
                seed_hint_q=seed_hint_q,
                attempts=max(4, min(12, seed_width)),
            )
            if roi_seed is None:
                continue
            roi_seed_success += 1
            roi_seed_qn.append(kinematics.normalize(roi_seed).astype(np.float32))
        if roi_seed_qn:
            roi_seed_qn_bank = np.asarray(roi_seed_qn, dtype=np.float32)

    # Treat proposal_batch_size as a hard CUDA clearance batch cap. Near-band
    # sampling may need many rounds, but one huge cdist allocation can OOM.
    proposal_batch_size = max(1, int(proposal_batch_size))
    # Broad pairs need both independently sampled endpoints to be free. Keep
    # a bounded but sufficiently large proposal budget so SDF-backed online
    # batches retain their high-clearance coverage instead of training mostly
    # from near-boundary and replay rows.
    max_attempts = max(target * (24 if near_boundary_only else 16), proposal_batch_size)
    report_every = max(proposal_batch_size, target)
    critical_clearance_target = _speed_label_to_clearance(
        0.20,
        clearance_margin_m,
        clearance_offset_m,
        label_floor=clearance_label_floor,
        label_power=clearance_label_power,
    )
    pooled_rows = 0
    pooled_critical_rows = 0
    # Critical-clearance coverage is a stratification preference, not a
    # second acceptance requirement. In a real cabinet the requested row
    # count can be complete while a 50% <=0.2-speed quota is unattainable.
    # Requiring both exhausted the full attempt budget after already logging
    # ``accepted=target/target``. Final stratification below still retains all
    # critical rows that were found.
    while pooled_rows < target and attempts < max_attempts:
        batch = min(proposal_batch_size, max_attempts - attempts)
        if batch <= 0:
            break
        attempts += batch
        seed_batches += max(1, int(np.ceil(batch / max(1, int(samples_per_seed)))))

        q0n = rng.uniform(-0.5, 0.5, size=(batch, 6)).astype(np.float32)
        workspace_seed_tries += batch
        workspace_seed_success += batch

        if seed_hint_qn is not None:
            # Keep some current-pose locality, but do not let one shell around
            # the camera pose consume most of a 6-D boundary batch.
            hint_frac = 0.15 if near_boundary_only else 0.20
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

        # Joint-space proposals alone do not know about the workspace ROI. Mix
        # a bounded number of IK-derived view states into each fast batch, then
        # retain local C-space jitter around those states. ``samples_per_seed``
        # limits IK work while ``proposal_batch_size`` remains the hard
        # collision-query cap.
        if len(roi_seed_qn_bank) > 0:
            roi_count = min(batch, max(1, int(round(batch * roi_seed_fraction))))
            roi_indices = rng.choice(batch, size=roi_count, replace=False)
            for start_idx in range(0, roi_count, seed_width):
                group = roi_indices[start_idx : start_idx + seed_width]
                roi_seed_qn = roi_seed_qn_bank[int(rng.integers(0, len(roi_seed_qn_bank)))]
                radius_hi = 0.08 if near_boundary_only else 0.24
                radii = rng.uniform(0.003, radius_hi, size=len(group))
                dirs = _unit_directions_batch(rng, len(group))
                q0n[group] = np.clip(roi_seed_qn[None, :] + dirs * radii[:, None], -0.5, 0.5)

        # A near-boundary q0 needs a genuinely local second endpoint. Large
        # 6-D jumps turn an otherwise valid boundary state into collision and
        # force the sampler to waste proposals in tight workcells.
        # Pair a shell state with a direction-neutral neighbourhood.
        # The former 0.06-radius, always-inward critical override made nearly
        # every q1 beside the boundary a floor-speed label.  A modestly wider
        # unbiased neighbourhood retains obstacle-side examples while also
        # supervising the surrounding clear/high-speed side of the shell. The
        # V6 used a 0.12 normalized radius. Arbitrary global state/goal pairs
        # are produced later by endpoint reshuffling; expanding this local
        # label proposal in v7/v8 reduced free-side calibration.
        q1_radius = rng.uniform(0.003, 0.12, size=batch)
        q1n_unbounded = q0n.astype(np.float64) + _unit_directions_batch(rng, batch) * q1_radius[:, None]
        q1_in_bounds = np.all((q1n_unbounded >= -0.5) & (q1n_unbounded <= 0.5), axis=1)
        q1n = np.clip(q1n_unbounded, -0.5, 0.5).astype(np.float32)
        q0 = kinematics.denormalize(q0n).astype(np.float32)
        q1 = kinematics.denormalize(q1n).astype(np.float32)
        d0, n0 = checker.clearance_and_normal_batch(q0)
        if near_boundary_only:
            critical_clearance = critical_clearance_target
            normal_mag = np.linalg.norm(n0, axis=1)
            direct_ok = (
                np.isfinite(d0)
                & (normal_mag > 1.0e-8)
                & (d0 >= float(clearance_offset_m))
                & (d0 <= float(clearance_margin_m))
            )
            # Project both distant and mid-band seeds toward the requested
            # critical quota. Tight workcells rarely expose d > margin, but
            # a safe d in the upper near band still has a usable normal for a
            # short local refinement.
            project_ok = np.isfinite(d0) & (normal_mag > 1.0e-8) & (d0 > critical_clearance)
            collision_project_ok = (
                np.isfinite(d0)
                & (normal_mag > 1.0e-8)
                & (d0 < float(clearance_offset_m))
            )
            q0_parts: list[np.ndarray] = []
            q1_parts: list[np.ndarray] = []
            d0_parts: list[np.ndarray] = []
            d1_parts: list[np.ndarray] = []
            n0_parts: list[np.ndarray] = []
            n1_parts: list[np.ndarray] = []
            q1_bounds_parts: list[np.ndarray] = []
            critical_parts: list[np.ndarray] = []

            # In constrained cabinet scenes many valid states already lie in
            # the desired clearance band. Keep them instead of requiring an
            # impossible projection from d > clearance_margin_m.
            if np.any(direct_ok):
                q0_direct = q0[direct_ok]
                q1_direct = q1[direct_ok].copy()
                q1_direct_in_bounds = q1_in_bounds[direct_ok].copy()
                d1_direct, n1_direct = checker.clearance_and_normal_batch(q1_direct)
                q0_parts.append(q0_direct)
                q1_parts.append(q1_direct)
                d0_parts.append(d0[direct_ok])
                d1_parts.append(d1_direct)
                n0_parts.append(n0[direct_ok])
                n1_parts.append(n1_direct)
                q1_bounds_parts.append(q1_direct_in_bounds)
                critical_parts.append(np.zeros((len(q0_direct),), dtype=bool))
            if np.any(project_ok):
                target_clearance = _sample_near_target_clearances(
                    rng,
                    int(np.count_nonzero(project_ok)),
                    clearance_margin_m,
                    clearance_offset_m,
                    label_floor=clearance_label_floor,
                    label_power=clearance_label_power,
                )
                requested_critical = target_clearance <= critical_clearance
                q0_projected, d0_projected, n0_projected = _project_toward_clearance_band(
                    checker,
                    kinematics,
                    q0[project_ok],
                    d0[project_ok],
                    n0[project_ok],
                    target_clearance,
                    minimum_clearance_m=clearance_offset_m,
                )
                q0n_projected = np.asarray(
                    [kinematics.normalize(q) for q in q0_projected], dtype=np.float32
                )
                q1_radius_projected = rng.uniform(0.003, 0.12, size=len(q0n_projected))
                q1n_projected_unbounded = (
                    q0n_projected.astype(np.float64)
                    + _unit_directions_batch(rng, len(q0n_projected)) * q1_radius_projected[:, None]
                )
                q1_projected_in_bounds = np.all(
                    (q1n_projected_unbounded >= -0.5) & (q1n_projected_unbounded <= 0.5), axis=1
                )
                q1n_projected = np.clip(q1n_projected_unbounded, -0.5, 0.5).astype(np.float32)
                q1_projected = kinematics.denormalize(q1n_projected).astype(np.float32)
                d1_projected, n1_projected = checker.clearance_and_normal_batch(q1_projected)
                q0_parts.append(q0_projected)
                q1_parts.append(q1_projected)
                d0_parts.append(d0_projected)
                d1_parts.append(d1_projected)
                n0_parts.append(n0_projected)
                n1_parts.append(n1_projected)
                q1_bounds_parts.append(q1_projected_in_bounds)
                critical_parts.append(requested_critical.astype(bool))
                projected_q0 += int(len(q0_projected))

            # Random 6-D proposals frequently land inside obstacles.  They are
            # valuable boundary discovery seeds, not wasted attempts: expand
            # along the C-space clearance normal and retain the verified
            # free-side shell point. Pair it with an unbiased local endpoint;
            # forcing every endpoint inward caused the learned field to
            # collapse toward its floor in sparsely covered safe regions.
            if np.any(collision_project_ok):
                collision_count = int(np.count_nonzero(collision_project_ok))
                target_clearance = _sample_near_target_clearances(
                    rng,
                    collision_count,
                    clearance_margin_m,
                    clearance_offset_m,
                    label_floor=clearance_label_floor,
                    label_power=clearance_label_power,
                    critical_only=True,
                )
                q0_collision, d0_collision, n0_collision = _project_out_of_collision_to_clearance_band(
                    checker,
                    kinematics,
                    q0[collision_project_ok],
                    n0[collision_project_ok],
                    target_clearance,
                    minimum_clearance_m=clearance_offset_m,
                )
                q0n_collision = np.asarray(
                    [kinematics.normalize(q) for q in q0_collision], dtype=np.float32
                )
                q1n_collision_unbounded = (
                    q0n_collision.astype(np.float64)
                    + _unit_directions_batch(rng, collision_count)
                    * rng.uniform(0.003, 0.12, size=collision_count)[:, None]
                )
                q1_collision_in_bounds = np.all(
                    (q1n_collision_unbounded >= -0.5)
                    & (q1n_collision_unbounded <= 0.5),
                    axis=1,
                )
                q1_collision = kinematics.denormalize(
                    np.clip(q1n_collision_unbounded, -0.5, 0.5).astype(np.float32)
                ).astype(np.float32)
                d1_collision, n1_collision = checker.clearance_and_normal_batch(q1_collision)
                q0_parts.append(q0_collision)
                q1_parts.append(q1_collision)
                d0_parts.append(d0_collision)
                d1_parts.append(d1_collision)
                n0_parts.append(n0_collision)
                n1_parts.append(n1_collision)
                q1_bounds_parts.append(q1_collision_in_bounds)
                critical_parts.append(np.ones((collision_count,), dtype=bool))
                projected_q0 += int(np.count_nonzero(d0_collision >= float(clearance_offset_m)))

            if q0_parts:
                q0 = np.concatenate(q0_parts, axis=0).astype(np.float32, copy=False)
                q1 = np.concatenate(q1_parts, axis=0).astype(np.float32, copy=False)
                d0 = np.concatenate(d0_parts, axis=0).astype(np.float32, copy=False)
                d1 = np.concatenate(d1_parts, axis=0).astype(np.float32, copy=False)
                n0 = np.concatenate(n0_parts, axis=0).astype(np.float32, copy=False)
                n1 = np.concatenate(n1_parts, axis=0).astype(np.float32, copy=False)
                q1_in_bounds = np.concatenate(q1_bounds_parts, axis=0)
                requested_critical = np.concatenate(critical_parts, axis=0)
            else:
                q0 = np.zeros((0, 6), dtype=np.float32)
                q1 = np.zeros((0, 6), dtype=np.float32)
                d0 = np.zeros((0,), dtype=np.float32)
                d1 = np.zeros((0,), dtype=np.float32)
                n0 = np.zeros((0, 6), dtype=np.float32)
                n1 = np.zeros((0, 6), dtype=np.float32)
                q1_in_bounds = np.zeros((0,), dtype=bool)
                requested_critical = np.zeros((0,), dtype=bool)
        else:
            d1, n1 = checker.clearance_and_normal_batch(q1)
        normal_ok = (np.linalg.norm(n0, axis=1) > 1.0e-8) & (np.linalg.norm(n1, axis=1) > 1.0e-8)
        # q0 defines the verified free-space boundary shell.  q1 follows the
        # reference data generator and may lie on the obstacle side, where its
        # clearance is clamped to the configured floor-speed label.
        endpoint_clearance_ok = (d0 >= float(clearance_offset_m)) & (d1 >= 0.0)
        finite_ok = np.isfinite(d0) & np.isfinite(d1) & normal_ok & endpoint_clearance_ok & q1_in_bounds
        if near_boundary_only:
            finite_ok &= d0 <= float(clearance_margin_m)
            finite_ok &= (~requested_critical) | (d0 <= critical_clearance)
        idx_valid = np.where(finite_ok)[0]
        if len(idx_valid):
            need = max(1, target - pooled_rows)
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
            pooled_rows += int(len(idx_keep))
            if near_boundary_only:
                pooled_critical_rows += int(np.count_nonzero(d0[idx_keep] <= critical_clearance_target))
        accepted_so_far = min(target, pooled_rows)
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
        roi_seed_tries=roi_seed_tries,
        roi_seed_success=roi_seed_success,
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
    clearance_offset_m: float = 0.0,
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
        out["speed0_critical_frac"] = float(np.mean(speeds[:, 0] <= 0.20))
        out["speed1_critical_frac"] = float(np.mean(speeds[:, 1] <= 0.20))
    else:
        out["speed0_sat_frac"] = 0.0
        out["speed1_sat_frac"] = 0.0
        out["speed0_low_frac"] = 0.0
        out["speed1_low_frac"] = 0.0
        out["speed0_critical_frac"] = 0.0
        out["speed1_critical_frac"] = 0.0
    if len(d0):
        out["q0_clearance_mean"] = float(np.mean(d0))
        out["q0_clearance_min"] = float(np.min(d0))
        out["q0_clearance_max"] = float(np.max(d0))
        out["q0_near_margin_frac"] = float(np.mean(d0 <= float(clearance_margin_m)))
        shell_hi = max(float(clearance_offset_m), min(float(clearance_margin_m), 3.0 * float(clearance_offset_m)))
        out["q0_boundary_shell_frac"] = float(
            np.mean((d0 >= float(clearance_offset_m)) & (d0 <= shell_hi))
        )
    else:
        out["q0_clearance_mean"] = 0.0
        out["q0_clearance_min"] = 0.0
        out["q0_clearance_max"] = 0.0
        out["q0_near_margin_frac"] = 0.0
        out["q0_boundary_shell_frac"] = 0.0
    if len(d1):
        out["q1_clearance_mean"] = float(np.mean(d1))
        out["q1_clearance_min"] = float(np.min(d1))
        out["q1_clearance_max"] = float(np.max(d1))
        out["q1_near_margin_frac"] = float(np.mean(d1 <= float(clearance_margin_m)))
        out["q1_obstacle_side_frac"] = float(np.mean(d1 < float(clearance_offset_m)))
    else:
        out["q1_clearance_mean"] = 0.0
        out["q1_clearance_min"] = 0.0
        out["q1_clearance_max"] = 0.0
        out["q1_near_margin_frac"] = 0.0
        out["q1_obstacle_side_frac"] = 0.0
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
    proposal_batch_size: int = 256,
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
    # Keep every clearance query under the same hard cap used by the global
    # sampler. A path-centered proposal can have many rounds, but allocating
    # one cdist tensor for the full target is what caused the CUDA OOM.
    proposal_batch = max(1, int(proposal_batch_size))
    max_attempts = max(target * 24, proposal_batch)
    while sum(len(x) for x in q0_pool) < target and attempts < max_attempts:
        batch = min(proposal_batch, max_attempts - attempts)
        attempts += batch
        seed_batches += 1

        anchor_idx = rng.integers(0, len(anchor_qn), size=batch)
        # These rows exist to close local coverage holes along important
        # C-space corridors. Keep q0 close to the supplied anchors; broad and
        # near-boundary stages already provide global obstacle coverage.
        q0_radius = rng.uniform(0.0, 0.025, size=batch)
        q0n = np.clip(
            anchor_qn[anchor_idx].astype(np.float64) + _unit_directions_batch(rng, batch) * q0_radius[:, None],
            -0.5,
            0.5,
        ).astype(np.float32)
        q1_radius = rng.uniform(0.02, 0.12, size=batch)
        q1n_unbounded = q0n.astype(np.float64) + _unit_directions_batch(rng, batch) * q1_radius[:, None]
        q1_in_bounds = np.all((q1n_unbounded >= -0.5) & (q1n_unbounded <= 0.5), axis=1)
        q1n = np.clip(q1n_unbounded, -0.5, 0.5).astype(np.float32)
        q0 = kinematics.denormalize(q0n).astype(np.float32)
        q1 = kinematics.denormalize(q1n).astype(np.float32)
        d0, n0 = checker.clearance_and_normal_batch(q0)
        d1, n1 = checker.clearance_and_normal_batch(q1)

        normal_ok = (np.linalg.norm(n0, axis=1) > 1.0e-8) & (np.linalg.norm(n1, axis=1) > 1.0e-8)
        endpoint_clearance_ok = (d0 >= float(clearance_offset_m)) & (d1 >= float(clearance_offset_m))
        valid = np.isfinite(d0) & np.isfinite(d1) & normal_ok & endpoint_clearance_ok & q1_in_bounds
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
        sampling_mode="path_centered:local",
    )
    stats.update(
        {
            "attempts": float(attempts),
            "ik_seed_tries": float(seed_batches),
            "accepted_pairs": float(len(rows)),
            "acceptance_rate": 0.0 if attempts <= 0 else float(len(rows)) / float(attempts),
            "accepted_per_seed": 0.0 if seed_batches <= 0 else float(len(rows)) / float(seed_batches),
            "samples_per_seed": float(proposal_batch),
            "refined_q0": 0.0,
            "refined_q1": 0.0,
            "anchor_seed_tries": float(attempts),
            "anchor_seed_success": float(len(rows)),
            "roi_seed_tries": 0.0,
            "workspace_seed_tries": 0.0,
            "roi_seed_success": 0.0,
            "workspace_seed_success": 0.0,
        }
    )
    return raw, rows, _finalize_sampler_stats(
        stats, rows, d0_all, d1_all, clearance_margin_m, clearance_offset_m
    )


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
    roi_seed_fraction: float = 0.0,
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
        proposal_batch_size = max(1, int(proposal_batch_size))
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
            roi_min=roi_min,
            roi_max=roi_max,
            roi_seed_fraction=roi_seed_fraction,
            anchor_qs=anchor_qs,
            anchor_seed_probability=anchor_seed_probability,
            clearance_label_floor=clearance_label_floor,
            clearance_label_power=clearance_label_power,
            sampling_mode=sampling_mode,
            proposal_batch_size=proposal_batch_size,
            near_boundary_only=near_boundary_only,
            progress_cb=progress_cb,
        )
        # This sampler has an explicit clearance-query budget. Falling through
        # to the legacy rejection loop when a map becomes restrictive defeats
        # that budget and can turn one online batch into tens of thousands of
        # exact point-cloud queries. Replay supplies the remaining optimizer
        # rows, so return the valid, collision-labelled subset immediately.
        return raw_fast, rows_fast, stats_fast

    raw_rows = []
    pair_rows = []
    proposal_batch_size = max(1, int(proposal_batch_size))

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
        d0_parts = []
        n0_parts = []
        for start in range(0, len(q0_batch), proposal_batch_size):
            d_chunk, n_chunk = checker.clearance_and_normal_batch(q0_batch[start : start + proposal_batch_size])
            d0_parts.append(np.asarray(d_chunk, dtype=np.float32))
            n0_parts.append(np.asarray(n_chunk, dtype=np.float32))
        d0_batch = np.concatenate(d0_parts, axis=0)
        n0_q_batch = np.concatenate(n0_parts, axis=0)

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
        d1_parts = []
        n1_parts = []
        for start in range(0, len(q1_valid), proposal_batch_size):
            d_chunk, n_chunk = checker.clearance_and_normal_batch(q1_valid[start : start + proposal_batch_size])
            d1_parts.append(np.asarray(d_chunk, dtype=np.float32))
            n1_parts.append(np.asarray(n_chunk, dtype=np.float32))
        d1_valid = np.concatenate(d1_parts, axis=0)
        n1_q_batch = np.concatenate(n1_parts, axis=0)

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
