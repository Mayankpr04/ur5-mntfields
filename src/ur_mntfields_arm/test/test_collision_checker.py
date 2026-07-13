import torch

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker


def test_support_box_masks_only_fixed_base_attachment_spheres():
    checker = object.__new__(UR5PointCloudCollisionChecker)
    checker.support_box_count = 1
    checker.support_box_mask_t = torch.tensor([False, True], dtype=torch.bool)
    checker.support_contact_sphere_mask_t = torch.tensor([True, True, True, True, True, False], dtype=torch.bool)

    signed_dist = torch.tensor(
        [
            [-0.20, -0.80],
            [-0.30, -0.70],
            [-0.25, -0.90],
            [-0.35, -0.60],
            [-0.28, -0.75],
            [-0.22, -0.65],
            [-0.24, -0.55],
            [-0.31, -0.68],
            [-0.29, -0.58],
            [-0.27, -0.72],
            [-0.19, -0.62],
            [-0.23, -0.66],
        ],
        dtype=torch.float32,
    )

    masked = checker._box_distance_candidates(signed_dist, batch_size=2)

    ignored = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    checked = [5, 11]
    assert torch.equal(masked[ignored, 0], signed_dist[ignored, 0])
    assert torch.isinf(masked[ignored, 1]).all()
    assert torch.equal(masked[checked], signed_dist[checked])
