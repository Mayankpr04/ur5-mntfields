import numpy as np

from ur_mntfields_arm.roi_targeting import (
    distribute_candidate_budget,
    spatially_stratified_targets,
)


def test_spatial_targets_cover_each_populated_octant():
    corners = np.asarray(
        [
            [x, y, z]
            for x in (0.15, 0.85)
            for y in (0.15, 0.85)
            for z in (0.15, 0.85)
        ],
        dtype=np.float64,
    )
    points = np.vstack((corners, np.tile(np.asarray([[0.51, 0.51, 0.51]]), (200, 1))))

    targets = spatially_stratified_targets(
        points, np.zeros(3), np.ones(3), max_targets=8
    )

    octants = {tuple((target >= 0.5).astype(int)) for target in targets}
    assert len(targets) == 8
    assert len(octants) == 8
    assert np.max(np.linalg.norm(targets - 0.5, axis=1)) > 0.5


def test_limited_targets_remain_spatially_separated():
    points = np.asarray(
        [
            [x, y, z]
            for x in (0.1, 0.9)
            for y in (0.1, 0.9)
            for z in (0.1, 0.9)
        ],
        dtype=np.float64,
    )

    targets = spatially_stratified_targets(
        points, np.zeros(3), np.ones(3), max_targets=3
    )
    pairwise = np.linalg.norm(targets[:, None, :] - targets[None, :, :], axis=2)

    assert len(targets) == 3
    assert np.min(pairwise[np.triu_indices(3, k=1)]) >= 0.8


def test_candidate_budget_stays_bounded_and_balanced():
    budget = distribute_candidate_budget(18, 8)

    assert sum(budget) == 18
    assert len(budget) == 8
    assert max(budget) - min(budget) <= 1
