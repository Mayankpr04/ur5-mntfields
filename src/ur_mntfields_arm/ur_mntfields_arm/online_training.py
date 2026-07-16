"""State replay, calibration, and certification for fresh online fields.

The classes in this module deliberately have no ROS dependency.  Mapping and
planning nodes can therefore share exactly the same replay/versioning and
certification rules, and the safety invariants are unit-testable on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc


STATE_WIDTH = 17
Q_SLICE = slice(0, 6)
CLEARANCE_COLUMN = 6
SPEED_COLUMN = 7
NORMAL_SLICE = slice(8, 14)
UNSAFE_COLUMN = 14
SOURCE_COLUMN = 15
MAP_VERSION_COLUMN = 16


class SampleSource(IntEnum):
    BROAD = 0
    BOUNDARY_SHELL = 1
    FREE_BAND = 2
    FALSE_FREE = 3
    TRAJECTORY = 4
    COVERAGE = 5


SOURCE_SHARES = {
    SampleSource.BROAD: 0.25,
    SampleSource.BOUNDARY_SHELL: 0.25,
    SampleSource.FREE_BAND: 0.20,
    SampleSource.FALSE_FREE: 0.15,
    SampleSource.TRAJECTORY: 0.10,
    SampleSource.COVERAGE: 0.05,
}


def make_state_rows(
    normalized_q: np.ndarray,
    clearance_m: np.ndarray,
    speed_label: np.ndarray,
    clearance_normal: np.ndarray,
    unsafe_label: np.ndarray,
    sample_source: SampleSource | int | np.ndarray,
    map_version: int | np.ndarray,
) -> np.ndarray:
    q = np.asarray(normalized_q, dtype=np.float32).reshape(-1, 6)
    count = len(q)
    normal = np.asarray(clearance_normal, dtype=np.float32).reshape(count, 6)
    def col(value) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=np.float32), (count,)).reshape(-1, 1)
    rows = np.concatenate(
        (q, col(clearance_m), col(speed_label), normal, col(unsafe_label),
         col(sample_source), col(map_version)), axis=1
    )
    if not np.all(np.isfinite(rows)):
        raise ValueError("State replay rows must be finite")
    rows[:, Q_SLICE] = np.clip(rows[:, Q_SLICE], -0.5, 0.5)
    rows[:, SPEED_COLUMN] = np.clip(rows[:, SPEED_COLUMN], 0.0, 1.0)
    rows[:, UNSAFE_COLUMN] = np.clip(rows[:, UNSAFE_COLUMN], 0.0, 1.0)
    return rows.astype(np.float32)


def assign_clearance_sources(
    rows: np.ndarray,
    *,
    broad_source: SampleSource = SampleSource.BROAD,
) -> np.ndarray:
    """Classify ordinary exact labels without collapsing free space into shell.

    Explicit hard, trajectory, and coverage sources should bypass this helper;
    it is for broad/shell generation where clearance defines the reservoir.
    """
    result = np.asarray(rows, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != STATE_WIDTH or not len(result):
        return result.reshape(-1, STATE_WIDTH)
    clearance = result[:, CLEARANCE_COLUMN]
    result[:, SOURCE_COLUMN] = float(broad_source)
    result[clearance <= 0.03, SOURCE_COLUMN] = float(SampleSource.BOUNDARY_SHELL)
    free_band = (clearance > 0.03) & (clearance < 0.10)
    result[free_band, SOURCE_COLUMN] = float(SampleSource.FREE_BAND)
    return result


def legacy_pairs_to_state_rows(
    pair_rows: np.ndarray,
    *,
    source: SampleSource,
    map_version: int,
    clearance_margin_m: float = 0.10,
    label_floor: float = 0.0,
) -> np.ndarray:
    """Convert the online sampler's independently labelled pair endpoints."""
    rows = np.asarray(pair_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
        return np.zeros((0, STATE_WIDTH), dtype=np.float32)
    q = np.concatenate((rows[:, :6], rows[:, 6:12]), axis=0)
    speed = np.concatenate((rows[:, 12], rows[:, 13]), axis=0)
    normal = np.concatenate((rows[:, 14:20], rows[:, 20:26]), axis=0)
    floor = float(label_floor)
    unsafe = speed <= floor + 1.0e-6
    clearance = np.where(unsafe, 0.0, speed * float(clearance_margin_m))
    return make_state_rows(q, clearance, speed, normal, unsafe, source, map_version)


def scrambled_sobol_states(count: int, *, seed: int = 0, skip: int = 0) -> np.ndarray:
    """Deterministic scrambled Sobol coverage in normalized joint space."""
    engine = qmc.Sobol(d=6, scramble=True, seed=int(seed))
    if skip:
        engine.fast_forward(int(skip))
    return (engine.random(max(0, int(count))).astype(np.float32) - 0.5)


class StateReplay:
    """Deduplicated, map-versioned 300k-state replay with source quotas."""

    def __init__(self, capacity: int = 300_000, grid_size: float = 0.0025):
        self.capacity = max(1, int(capacity))
        self.grid_size = max(1.0e-6, float(grid_size))
        self._rows = np.zeros((0, STATE_WIDTH), dtype=np.float32)
        self.current_map_version = 0

    @property
    def rows(self) -> np.ndarray:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def _grid_keys(self, rows: np.ndarray) -> np.ndarray:
        return np.floor((rows[:, Q_SLICE] + 0.5) / self.grid_size).astype(np.int32)

    @staticmethod
    def _prefer(candidate: np.ndarray, incumbent: np.ndarray) -> bool:
        candidate_version = int(round(float(candidate[MAP_VERSION_COLUMN])))
        incumbent_version = int(round(float(incumbent[MAP_VERSION_COLUMN])))
        if candidate_version != incumbent_version:
            return candidate_version > incumbent_version
        # Never replace a false-free or lower-clearance state with an ordinary
        # sample in the same normalized C-space cell.
        candidate_hard = int(round(float(candidate[SOURCE_COLUMN]))) == int(SampleSource.FALSE_FREE)
        incumbent_hard = int(round(float(incumbent[SOURCE_COLUMN]))) == int(SampleSource.FALSE_FREE)
        if candidate_hard != incumbent_hard:
            return candidate_hard
        if float(candidate[CLEARANCE_COLUMN]) != float(incumbent[CLEARANCE_COLUMN]):
            return float(candidate[CLEARANCE_COLUMN]) < float(incumbent[CLEARANCE_COLUMN])
        return float(candidate[MAP_VERSION_COLUMN]) > float(incumbent[MAP_VERSION_COLUMN])

    def add(self, rows: np.ndarray) -> None:
        incoming = np.asarray(rows, dtype=np.float32)
        if incoming.ndim != 2 or incoming.shape[1] != STATE_WIDTH:
            raise ValueError(f"Expected replay rows [N,{STATE_WIDTH}]")
        if len(incoming) == 0:
            return
        merged = np.concatenate((self._rows, incoming), axis=0)
        keys = self._grid_keys(merged)
        selected: dict[tuple[int, ...], int] = {}
        for index, key_array in enumerate(keys):
            key = tuple(int(v) for v in key_array)
            old = selected.get(key)
            if old is None or self._prefer(merged[index], merged[old]):
                selected[key] = index
        deduped = merged[np.asarray(list(selected.values()), dtype=np.int64)]
        self._rows = self._apply_quotas(deduped)

    def _apply_quotas(self, rows: np.ndarray) -> np.ndarray:
        """Keep independent per-source reservoirs from the first insertion.

        Reserving capacity is intentional: unavailable hard/trajectory data
        must not be silently replaced by shell samples, because doing so made
        the state head fail closed long before the replay reached capacity.
        """
        chosen: list[np.ndarray] = []
        source_values = np.rint(rows[:, SOURCE_COLUMN]).astype(np.int32)
        for source, share in SOURCE_SHARES.items():
            group = rows[source_values == int(source)]
            quota = int(round(self.capacity * share))
            if not len(group):
                chosen.append(group)
                continue
            version = np.rint(group[:, MAP_VERSION_COLUMN]).astype(np.int64)
            if source in (SampleSource.BOUNDARY_SHELL, SampleSource.FALSE_FREE):
                # Current-map, lowest-clearance hard states are most valuable.
                order = np.lexsort((group[:, CLEARANCE_COLUMN], -version))
            else:
                # Current-map rows come first; deduplication already provides
                # spatial diversity within each normalized C-space cell.
                order = np.argsort(-version, kind="stable")
            chosen.append(group[order[:quota]] if len(group) else group)
        result = np.concatenate(chosen, axis=0)
        return result[: self.capacity].astype(np.float32, copy=False)

    @staticmethod
    def _balanced_counts(
        rows: np.ndarray,
        count: int,
        *,
        max_repeats_per_state: int,
    ) -> dict[SampleSource, int]:
        values = np.rint(rows[:, SOURCE_COLUMN]).astype(np.int32)
        desired = {
            source: int(np.floor(max(0, count) * share))
            for source, share in SOURCE_SHARES.items()
        }
        for source in list(SOURCE_SHARES)[: max(0, count - sum(desired.values()))]:
            desired[source] += 1
        capacities = {
            source: int(np.count_nonzero(values == int(source))) * max(1, int(max_repeats_per_state))
            for source in SOURCE_SHARES
        }
        for source in desired:
            desired[source] = min(desired[source], capacities[source])
        missing = max(0, int(count) - sum(desired.values()))
        preferred = [SampleSource.BROAD, SampleSource.FREE_BAND, SampleSource.COVERAGE]
        available = [source for source in preferred if capacities[source] > desired[source]]
        if not available:
            available = [source for source in SOURCE_SHARES if capacities[source] > desired[source]]
        weights = {
            SampleSource.BROAD: 0.50,
            SampleSource.FREE_BAND: 0.35,
            SampleSource.COVERAGE: 0.15,
        }
        while available and missing > 0:
            total_weight = sum(weights.get(source, 1.0) for source in available)
            progressed = 0
            for source in list(available):
                room = capacities[source] - desired[source]
                if room <= 0:
                    continue
                extra = min(
                    room,
                    max(1, int(np.floor(missing * weights.get(source, 1.0) / total_weight))),
                )
                desired[source] += extra
                missing -= extra
                progressed += extra
                if missing <= 0:
                    break
            available = [source for source in available if capacities[source] > desired[source]]
            if progressed <= 0:
                break
        return desired

    def sample_balanced(
        self,
        count: int,
        *,
        rows: np.ndarray | None = None,
        replace: bool = True,
        max_repeats_per_state: int = 4,
    ) -> np.ndarray:
        """Sample a quota-balanced current-map batch.

        Sparse hard reservoirs may be sampled with replacement, while absent
        reservoirs redistribute only toward broad/free/coverage evidence.
        """
        data = self.valid_rows() if rows is None else np.asarray(rows, dtype=np.float32)
        count = max(0, int(count))
        if data.ndim != 2 or len(data) == 0 or count <= 0:
            return np.zeros((0, STATE_WIDTH), dtype=np.float32)
        values = np.rint(data[:, SOURCE_COLUMN]).astype(np.int32)
        parts: list[np.ndarray] = []
        repeat_cap = max(1, int(max_repeats_per_state)) if replace else 1
        for source, take in self._balanced_counts(
            data, count, max_repeats_per_state=repeat_cap
        ).items():
            group = data[values == int(source)]
            if take <= 0 or not len(group):
                continue
            actual_take = take if replace else min(take, len(group))
            if replace and actual_take > len(group):
                full, remainder = divmod(actual_take, len(group))
                idx_parts = [np.random.permutation(len(group)) for _ in range(full)]
                if remainder:
                    idx_parts.append(np.random.permutation(len(group))[:remainder])
                idx = np.concatenate(idx_parts)
            else:
                idx = np.random.choice(len(group), size=actual_take, replace=False)
            parts.append(group[idx])
        if not parts:
            return np.zeros((0, STATE_WIDTH), dtype=np.float32)
        batch = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        return batch[np.random.permutation(len(batch))]

    @staticmethod
    def _capped_sample_indices(
        pool: np.ndarray, count: int, *, max_repeats_per_state: int
    ) -> np.ndarray:
        """Sample indices without allowing a tiny reservoir to dominate."""
        available = np.asarray(pool, dtype=np.int64).reshape(-1)
        take = min(
            max(0, int(count)),
            len(available) * max(1, int(max_repeats_per_state)),
        )
        if take <= 0:
            return np.zeros((0,), dtype=np.int64)
        parts: list[np.ndarray] = []
        remaining = take
        while remaining > 0:
            order = np.random.permutation(available)
            chunk = order[: min(remaining, len(order))]
            parts.append(chunk)
            remaining -= len(chunk)
        return np.concatenate(parts).astype(np.int64, copy=False)

    def sample_learning_balanced_indices(
        self,
        count: int,
        *,
        rows: np.ndarray | None = None,
        max_repeats_per_state: int = 4,
    ) -> np.ndarray:
        """Return label- and source-aware optimizer indices.

        Replay admission already enforces source quotas. Optimizer batches
        additionally need verified free-space evidence; source-only sampling
        can otherwise produce a mostly-colliding batch and teach the unsafe
        head the trivial fail-closed solution. The disjoint target mixture is
        45% high-clearance free, 20% other free, 25% unsafe/shell, and 10%
        hard-mined trajectory or false-free states. False-blocked trajectory
        rows remain in the hard pool, adding focused free evidence.
        """
        data = self.valid_rows() if rows is None else np.asarray(rows, dtype=np.float32)
        count = max(0, int(count))
        if data.ndim != 2 or data.shape[1] != STATE_WIDTH or not len(data) or count <= 0:
            return np.zeros((0,), dtype=np.int64)

        all_idx = np.arange(len(data), dtype=np.int64)
        source = np.rint(data[:, SOURCE_COLUMN]).astype(np.int32)
        safe = data[:, UNSAFE_COLUMN] < 0.5
        high_free = safe & (data[:, SPEED_COLUMN] >= 0.5)
        hard = np.isin(
            source,
            (int(SampleSource.FALSE_FREE), int(SampleSource.TRAJECTORY)),
        )
        pools = {
            "high_free": all_idx[high_free & ~hard],
            "other_free": all_idx[safe & ~high_free & ~hard],
            "unsafe": all_idx[~safe & ~hard],
            "hard": all_idx[hard],
        }
        shares = {
            "high_free": 0.45,
            "other_free": 0.20,
            "unsafe": 0.25,
            "hard": 0.10,
        }
        desired = {name: int(np.floor(count * share)) for name, share in shares.items()}
        for name in tuple(shares)[: count - sum(desired.values())]:
            desired[name] += 1
        cap = max(1, int(max_repeats_per_state))
        for name, pool in pools.items():
            desired[name] = min(desired[name], len(pool) * cap)

        # Missing categories preferentially become useful free evidence, then
        # unsafe evidence, rather than more copies of a tiny hard reservoir.
        missing = count - sum(desired.values())
        for name in ("high_free", "other_free", "unsafe", "hard"):
            room = len(pools[name]) * cap - desired[name]
            extra = min(max(0, room), max(0, missing))
            desired[name] += extra
            missing -= extra
            if missing <= 0:
                break

        parts = [
            self._capped_sample_indices(
                pools[name], desired[name], max_repeats_per_state=cap
            )
            for name in shares
            if desired[name] > 0
        ]
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        indices = np.concatenate(parts).astype(np.int64, copy=False)
        return indices[np.random.permutation(len(indices))]

    def sample_learning_balanced(
        self, count: int, *, max_repeats_per_state: int = 4
    ) -> np.ndarray:
        data = self.valid_rows()
        indices = self.sample_learning_balanced_indices(
            count, rows=data, max_repeats_per_state=max_repeats_per_state
        )
        return data[indices] if len(indices) else np.zeros((0, STATE_WIDTH), dtype=np.float32)

    def stale_rows_balanced(self, count: int) -> np.ndarray:
        stale = self.stale_rows()
        if not len(stale):
            return stale
        if len(stale) <= max(0, int(count)):
            return stale.copy()
        return self.sample_balanced(min(int(count), len(stale)), rows=stale, replace=False)

    def discard_stale(self) -> int:
        stale_count = len(self.stale_rows())
        self._rows = self.valid_rows().copy()
        return stale_count

    def limit_stale(self, count: int) -> tuple[int, int]:
        """Retain only a balanced, bounded relabelling queue."""
        stale = self.stale_rows()
        keep = self.stale_rows_balanced(max(0, int(count)))
        valid = self.valid_rows()
        self._rows = (
            np.concatenate((valid, keep), axis=0).astype(np.float32, copy=False)
            if len(keep) else valid.copy()
        )
        return len(keep), max(0, len(stale) - len(keep))

    def set_map_version(self, version: int) -> None:
        self.current_map_version = int(version)

    def valid_rows(self) -> np.ndarray:
        if not len(self._rows):
            return self._rows
        return self._rows[
            np.rint(self._rows[:, MAP_VERSION_COLUMN]).astype(np.int64)
            == self.current_map_version
        ]

    def stale_rows(self) -> np.ndarray:
        if not len(self._rows):
            return self._rows
        return self._rows[
            np.rint(self._rows[:, MAP_VERSION_COLUMN]).astype(np.int64)
            != self.current_map_version
        ]

    def replace_relabelled(self, rows: np.ndarray) -> None:
        incoming = np.asarray(rows, dtype=np.float32)
        if len(incoming) and not np.all(
            np.rint(incoming[:, MAP_VERSION_COLUMN]).astype(np.int64) == self.current_map_version
        ):
            raise ValueError("Relabelled rows must carry the current map version")
        # ``add`` replaces matching stale grid cells with the newer version
        # while retaining other stale rows for later bounded relabel batches.
        self.add(incoming)

    def source_counts(self, *, valid_only: bool = True) -> dict[str, int]:
        rows = self.valid_rows() if valid_only else self._rows
        values = np.rint(rows[:, SOURCE_COLUMN]).astype(np.int32) if len(rows) else np.zeros(0)
        return {source.name.lower(): int(np.count_nonzero(values == int(source))) for source in SampleSource}

    def coverage_tree(self) -> cKDTree | None:
        rows = self.valid_rows()
        return cKDTree(rows[:, Q_SLICE]) if len(rows) else None


def paired_shell_states(q_boundary: np.ndarray, normal: np.ndarray, epsilon: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_boundary, dtype=np.float32).reshape(-1, 6)
    n = np.asarray(normal, dtype=np.float32).reshape(-1, 6)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1.0e-8)
    eps = float(epsilon)
    return np.clip(q - eps * n, -0.5, 0.5), np.clip(q + eps * n, -0.5, 0.5)


def binary_search_boundary(
    colliding_q: np.ndarray,
    free_q: np.ndarray,
    clearance_fn,
    *,
    iterations: int = 12,
) -> np.ndarray:
    """Bisect verified collision/free brackets in normalized C-space."""
    low = np.asarray(colliding_q, dtype=np.float32).reshape(-1, 6).copy()
    high = np.asarray(free_q, dtype=np.float32).reshape(-1, 6).copy()
    if len(low) != len(high):
        raise ValueError("Collision/free bracket arrays must have equal length")
    low_clearance = np.asarray(clearance_fn(low), dtype=np.float32).reshape(-1)
    high_clearance = np.asarray(clearance_fn(high), dtype=np.float32).reshape(-1)
    if np.any(low_clearance > 0.0) or np.any(high_clearance <= 0.0):
        raise ValueError("Boundary search requires verified collision/free brackets")
    for _ in range(max(1, int(iterations))):
        mid = 0.5 * (low + high)
        collision = np.asarray(clearance_fn(mid), dtype=np.float32).reshape(-1) <= 0.0
        low[collision] = mid[collision]
        high[~collision] = mid[~collision]
    return 0.5 * (low + high)


def exact_label_states(
    checker,
    kinematics,
    voxel_map,
    q_batch: np.ndarray,
    *,
    clearance_margin_m: float = 0.10,
    clearance_offset_m: float = 0.01,
    source: SampleSource = SampleSource.BROAD,
    state_batch_size: int = 128,
    return_observed: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Label physical joint states and optionally return map support.

    Occupancy comes only from the accumulated cloud inside ``checker``;
    bounding boxes are expected to have been supplied to ``voxel_map`` as ROI
    bounds, not passed as obstacle geometry. Observation support is deliberately
    kept separate from the physical collision label: unknown configurations are
    rejected by replay/coverage support, not taught to the geometry head as if
    they were obstacles.
    """
    q = np.asarray(q_batch, dtype=np.float32).reshape(-1, 6)
    batch_size = max(1, int(state_batch_size))
    clearance_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    for start in range(0, len(q), batch_size):
        q_chunk = q[start : start + batch_size]
        clearance_chunk, normal_chunk = checker.clearance_and_normal_batch(q_chunk)
        clearance_parts.append(np.asarray(clearance_chunk, dtype=np.float32))
        normal_parts.append(np.asarray(normal_chunk, dtype=np.float32))
        observed_chunk = np.ones(len(q_chunk), dtype=bool)
        if hasattr(checker, "robot_spheres_batch"):
            centers_batch, _radii = checker.robot_spheres_batch(q_chunk)
            for index, centers in enumerate(centers_batch):
                observed_chunk[index] = all(voxel_map.is_observed_free(center) for center in centers)
        else:
            for index, state in enumerate(q_chunk):
                centers, _radii = checker.robot_spheres(state)
                observed_chunk[index] = all(voxel_map.is_observed_free(center) for center in centers)
        observed_parts.append(observed_chunk)
    clearance = np.concatenate(clearance_parts) if clearance_parts else np.zeros((0,), dtype=np.float32)
    physical_normal = np.concatenate(normal_parts, axis=0) if normal_parts else np.zeros((0, 6), dtype=np.float32)
    observed = np.concatenate(observed_parts) if observed_parts else np.zeros((0,), dtype=bool)
    normalized = np.asarray([kinematics.normalize(row) for row in q], dtype=np.float32)
    span = np.asarray(kinematics.joint_max - kinematics.joint_min, dtype=np.float32)
    normal = physical_normal * span[None, :]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(1.0e-8)
    unsafe = clearance <= 0.0
    speed = np.clip(clearance, clearance_offset_m, clearance_margin_m) / max(clearance_margin_m, 1.0e-6)
    speed[unsafe] = 0.0
    rows = make_state_rows(
        normalized, clearance, speed, normal, unsafe.astype(np.float32),
        source, voxel_map.map_version,
    )
    if return_observed:
        return rows, observed
    return rows


def calibrate_conservative_speed(
    prediction: np.ndarray,
    target: np.ndarray,
    clearance_m: np.ndarray,
    *,
    coverage_distance: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit subtractive 99th-percentile margins in inference-available bins.

    ``clearance_m`` remains part of the interface and validates held-out label
    alignment, but bin selection deliberately uses predicted speed because
    exact clearance is unavailable during learned-only execution.
    """
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    labelled_clearance = np.asarray(clearance_m, dtype=np.float64).reshape(-1)
    coverage = np.zeros_like(pred) if coverage_distance is None else np.asarray(coverage_distance, dtype=np.float64).reshape(-1)
    if not (len(pred) == len(truth) == len(labelled_clearance) == len(coverage)):
        raise ValueError("Calibration arrays must have equal length")
    error = np.maximum(pred - truth, 0.0)
    result = {"global": float(np.quantile(error, 0.99)) if len(error) else 1.0}
    # Calibration bins must use a quantity available at inference. The old
    # code binned held-out rows by exact clearance but selected margins using
    # predicted clearance, assigning free states margins learned for a
    # different population. Use identical predicted-speed bins on both sides.
    predicted_clearance = 0.10 * np.clip(pred, 0.0, 1.0)
    clearance_bins = ((-np.inf, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, np.inf))
    coverage_bins = ((-np.inf, 0.02), (0.02, 0.05), (0.05, np.inf))
    for ci, (clo, chi) in enumerate(clearance_bins):
        for di, (dlo, dhi) in enumerate(coverage_bins):
            mask = (
                (predicted_clearance >= clo)
                & (predicted_clearance < chi)
                & (coverage >= dlo)
                & (coverage < dhi)
            )
            result[f"clearance_{ci}_coverage_{di}"] = (
                float(np.quantile(error[mask], 0.99)) if np.count_nonzero(mask) >= 20 else result["global"]
            )
    return result


def derive_coverage_radius(
    distance: np.ndarray,
    false_free: np.ndarray,
    *,
    max_false_free_rate: float = 0.02,
    cap: float = 0.08,
    min_states: int = 50,
) -> float:
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    false_free = np.asarray(false_free, dtype=bool).reshape(-1)
    candidates = np.unique(np.minimum(distance[distance <= cap], cap))
    selected = 0.0
    for radius in candidates:
        mask = distance <= radius
        if np.count_nonzero(mask) >= int(min_states) and float(np.mean(false_free[mask])) <= max_false_free_rate:
            selected = float(radius)
    return min(selected, float(cap))


def select_active_candidates(
    model,
    replay: StateReplay,
    *,
    candidate_count: int = 65_536,
    label_count: int = 2_048,
    seed: int = 0,
) -> np.ndarray:
    """Rank global and known-free-local candidates for informative labels."""
    candidates = scrambled_sobol_states(candidate_count, seed=seed)
    # Reserve part of the active pool for perturbations around verified-free
    # replay states. Pure global Sobol mining repeatedly found unsafe states in
    # cluttered scenes and supplied almost no signal for free-space recall.
    valid = replay.valid_rows()
    free = valid[
        (valid[:, UNSAFE_COLUMN] < 0.5) & (valid[:, SPEED_COLUMN] >= 0.5)
    ]
    local_count = min(len(candidates) // 3, max(0, len(candidates) - 1))
    if len(free) and local_count > 0:
        rng = np.random.default_rng(int(seed) + 1_048_583)
        base = free[rng.integers(0, len(free), size=local_count), Q_SLICE]
        scale = rng.choice(np.asarray([0.015, 0.04, 0.08], dtype=np.float32), size=(local_count, 1))
        local = np.clip(base + rng.normal(size=(local_count, 6)) * scale, -0.5, 0.5)
        candidates[-local_count:] = local.astype(np.float32)
    speed, unsafe, conservative = model.predict_normalized_state_geometry(candidates)
    tree = replay.coverage_tree()
    if tree is None:
        distance = np.ones(len(candidates), dtype=np.float32)
    else:
        distance, _ = tree.query(candidates, k=1)
    count = min(max(0, int(label_count)), len(candidates))
    if count <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    uncertainty = 1.0 - 2.0 * np.abs(np.asarray(unsafe) - 0.5)
    coverage = np.minimum(np.asarray(distance) / 0.08, 2.0)
    scores = (
        np.asarray(conservative) * (1.0 - np.asarray(unsafe)),
        uncertainty,
        coverage,
        (1.0 - np.asarray(conservative)) + np.asarray(unsafe) - 0.25 * coverage,
    )
    shares = (0.30, 0.25, 0.25, 0.20)
    selected: list[int] = []
    used = np.zeros(len(candidates), dtype=bool)
    for score, share in zip(scores, shares):
        take = min(count - len(selected), int(round(count * share)))
        if take <= 0:
            continue
        available = np.flatnonzero(~used)
        local = available[np.argpartition(score[available], -take)[-take:]]
        selected.extend(int(index) for index in local)
        used[local] = True
    if len(selected) < count:
        remaining = np.flatnonzero(~used)
        selected.extend(int(index) for index in remaining[: count - len(selected)])
    return candidates[np.asarray(selected[:count], dtype=np.int64)].astype(np.float32)


def false_free_mask(
    predicted_speed: np.ndarray,
    unsafe_probability: np.ndarray,
    target_speed: np.ndarray,
    *,
    low_speed_threshold: float = 0.20,
) -> np.ndarray:
    return (
        (np.asarray(target_speed) <= low_speed_threshold)
        & (np.asarray(predicted_speed) >= low_speed_threshold)
        & (np.asarray(unsafe_probability) < 0.10)
    )


@dataclass(frozen=True)
class CertificationMetrics:
    low_clearance_states: int
    low_clearance_false_free_rate: float
    route_attempts: int
    route_acceptance_rate: float
    accepted_goal_reach_rate: float
    accepted_collision_free_rate: float
    scene_version_match: bool
    direct_edge_median_ms: float
    planning_p95_ms: float
    route_requested: int = 0
    route_free_states: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.low_clearance_states >= 1000
            and self.low_clearance_false_free_rate <= 0.02
            and self.route_attempts >= 200
            and self.route_acceptance_rate >= 0.60
            and self.accepted_goal_reach_rate >= 0.95
            and self.accepted_collision_free_rate >= 0.95
            and self.scene_version_match
            and self.direct_edge_median_ms <= 25.0
            and self.planning_p95_ms <= 100.0
        )


class OnlineTrainingBudget:
    """Monotonic 12-minute schedule starting at the first valid observation."""

    def __init__(self):
        self.started_at: float | None = None

    def start(self, now: float | None = None) -> None:
        if self.started_at is None:
            self.started_at = time.monotonic() if now is None else float(now)

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (time.monotonic() if now is None else float(now)) - self.started_at)

    def phase(self, now: float | None = None) -> str:
        elapsed = self.elapsed(now)
        if elapsed < 90: return "bootstrap"
        if elapsed < 540: return "train"
        if elapsed < 600: return "certify"
        if elapsed < 660: return "focused_recovery"
        if elapsed < 720: return "final_certify"
        return "expired"

    def may_stop_early(self, certification_passed: bool, now: float | None = None) -> bool:
        return bool(certification_passed) and self.elapsed(now) >= 300.0

    @staticmethod
    def cycle_allocation(cycle_seconds: float = 30.0) -> dict[str, float]:
        duration = max(0.0, float(cycle_seconds))
        return {
            "label_and_relabel": 0.45 * duration,
            "optimizer": 0.40 * duration,
            "audit_and_trajectory_mining": 0.15 * duration,
        }


def checkpoint_scene_metadata(
    voxel_map,
    replay: StateReplay,
    artifact_dir: str | Path,
    *,
    training_wall_time: float,
) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    map_path = voxel_map.save(artifact_dir / "voxel_map_final.npz")
    return {
        "scene_signature": voxel_map.scene_signature(),
        "voxel_map_path": map_path.name,
        "map_version": int(voxel_map.map_version),
        "training_wall_time": float(training_wall_time),
        "sample_source_counts": replay.source_counts(valid_only=True),
    }


def save_certified_checkpoint(model, output_dir: str | Path, metrics: CertificationMetrics, metadata: dict[str, object]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined = dict(metadata)
    coverage_ready = (
        float(getattr(model, "shell_coverage_radius", 0.0)) > 0.0
        and float(getattr(model, "free_coverage_radius", 0.0)) > 0.0
        and len(getattr(model, "coverage_states", ())) > 0
    )
    scene_ready = all(key in combined for key in ("scene_signature", "voxel_map_path", "map_version"))
    passed = metrics.passed and coverage_ready and scene_ready
    combined["certification_passed"] = passed
    combined["certification_metrics"] = metrics.__dict__
    model.certification_passed = passed
    target = output / ("weights_final.pt" if passed else "weights_partial.pt")
    model.save_checkpoint(target, metadata=combined)
    diagnostics = output / "certification.json"
    diagnostics.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    return target
