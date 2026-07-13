from __future__ import annotations

import numpy as np


def spatially_stratified_targets(
    points: np.ndarray,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    max_targets: int,
    bins_per_axis: int = 2,
) -> np.ndarray:
    """Return spatially separated representative points inside an ROI."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    limit = max(0, int(max_targets))
    if len(pts) == 0 or limit == 0:
        return np.zeros((0, 3), dtype=np.float64)

    lo = np.asarray(roi_min, dtype=np.float64).reshape(3)
    hi = np.asarray(roi_max, dtype=np.float64).reshape(3)
    span = np.maximum(hi - lo, 1.0e-6)
    finite = np.all(np.isfinite(pts), axis=1)
    inside = np.all((pts >= lo[None, :]) & (pts <= hi[None, :]), axis=1)
    pts = pts[finite & inside]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    bins = max(1, int(bins_per_axis))
    normalized = np.clip((pts - lo[None, :]) / span[None, :], 0.0, 1.0 - 1.0e-12)
    indices = np.floor(normalized * bins).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for point_idx, bin_idx in enumerate(indices):
        key = tuple(int(v) for v in bin_idx)
        buckets.setdefault(key, []).append(point_idx)

    keys = sorted(buckets)
    representatives = np.asarray(
        [pts[buckets[key]].mean(axis=0) for key in keys], dtype=np.float64
    )
    counts = np.asarray([len(buckets[key]) for key in keys], dtype=np.int64)
    if len(representatives) <= limit:
        return representatives

    # Select bins with farthest-point sampling. The peripheral seed prevents a
    # dense central bin from becoming the sole ROI target.
    roi_center = 0.5 * (lo + hi)
    center_dist = np.linalg.norm(representatives - roi_center[None, :], axis=1)
    first = min(
        range(len(representatives)),
        key=lambda idx: (-float(center_dist[idx]), -int(counts[idx]), keys[idx]),
    )
    selected = [first]
    remaining = set(range(len(representatives)))
    remaining.remove(first)
    while remaining and len(selected) < limit:
        chosen = min(
            remaining,
            key=lambda idx: (
                -float(
                    np.min(
                        np.linalg.norm(
                            representatives[idx][None, :] - representatives[selected], axis=1
                        )
                    )
                ),
                -int(counts[idx]),
                keys[idx],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return representatives[np.asarray(selected, dtype=np.int64)]


def distribute_candidate_budget(total_candidates: int, target_count: int) -> tuple[int, ...]:
    """Split a fixed candidate budget across target points without overflow."""
    total = max(0, int(total_candidates))
    count = max(0, int(target_count))
    if total == 0 or count == 0:
        return ()
    count = min(count, total)
    base, remainder = divmod(total, count)
    return tuple(base + (1 if idx < remainder else 0) for idx in range(count))
