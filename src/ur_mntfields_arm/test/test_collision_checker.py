import torch
import numpy as np

from ur_mntfields_arm.collision_checker import (
    UR5PointCloudCollisionChecker,
    UR5SDFCollisionChecker,
    wrist_camera_collision_spheres,
)


def test_wrist_camera_collision_sphere_matches_mount_and_box_circumsphere():
    camera_in_tool = np.eye(4, dtype=np.float64)
    camera_in_tool[:3, 3] = [0.0, -0.08, 0.02]
    spheres = wrist_camera_collision_spheres(camera_in_tool)
    assert spheres.shape == (3, 4)
    assert np.allclose(spheres[-1, :3], [0.0, -0.08, 0.02])
    assert spheres[-1, 3] == np.linalg.norm([0.02, 0.02, 0.01])


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


def test_torch_sdf_lookup_matches_cpu_trilinear_reference():
    checker = object.__new__(UR5SDFCollisionChecker)
    rng = np.random.default_rng(8)
    checker.sdf_origin = np.asarray([-0.2, -0.3, -0.4], dtype=np.float64)
    checker.sdf_effective_voxel_size_m = 0.05
    checker.sdf_grid = rng.uniform(0.0, 1.0, size=(8, 9, 10)).astype(np.float32)
    checker.sdf_grad_grid = rng.normal(size=(8, 9, 10, 3)).astype(np.float32)
    checker.sdf_grid_t = torch.from_numpy(checker.sdf_grid)
    checker.sdf_grad_grid_t = torch.from_numpy(checker.sdf_grad_grid)
    checker.sdf_upper = checker.sdf_origin + 0.05 * (np.asarray(checker.sdf_grid.shape) - 1)
    points = rng.uniform(checker.sdf_origin - 0.03, checker.sdf_upper + 0.03, size=(128, 3))

    cpu_distance, cpu_normal = checker._point_sdf_lookup(points)
    torch_distance, torch_normal = checker._point_sdf_lookup_torch(
        torch.as_tensor(points, dtype=torch.float32)
    )
    np.testing.assert_allclose(torch_distance.numpy(), cpu_distance, atol=2.0e-6)
    np.testing.assert_allclose(torch_normal.numpy(), cpu_normal, atol=2.0e-5)


def test_robot_self_filter_removes_points_inside_collision_spheres():
    checker = object.__new__(UR5PointCloudCollisionChecker)
    checker._sphere_samples = lambda _q: (
        np.asarray([[0.2, -0.1, 0.4]], dtype=np.float64),
        np.asarray([0.05], dtype=np.float64),
    )
    points = np.asarray(
        [[0.2, -0.1, 0.4], [0.24, -0.1, 0.4], [0.40, -0.1, 0.4]],
        dtype=np.float32,
    )

    filtered, removed = checker.filter_robot_self_points(
        points, np.zeros(6), padding_m=0.01
    )

    assert removed == 2
    np.testing.assert_allclose(filtered, [[0.40, -0.1, 0.4]])
