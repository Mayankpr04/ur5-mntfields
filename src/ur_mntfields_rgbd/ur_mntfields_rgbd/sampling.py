import math
from typing import Dict

import numpy as np


def quaternion_to_matrix_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def transform_to_matrix(translation, rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_to_matrix_xyzw(rotation.x, rotation.y, rotation.z, rotation.w)
    matrix[0, 3] = translation.x
    matrix[1, 3] = translation.y
    matrix[2, 3] = translation.z
    return matrix


def ray_dirs_camera(height: int, width: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (cols - cx) / fx
    y = (rows - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    return np.stack((x, y, z), axis=-1)


def depth_to_meters(depth_image: np.ndarray, depth_scale: float, max_depth: float) -> np.ndarray:
    if depth_image.dtype == np.uint16:
        depth_m = depth_image.astype(np.float32) * depth_scale
    else:
        depth_m = depth_image.astype(np.float32)

    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m > max_depth] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m


def compute_normalization_bound(
    raw_frame_data: np.ndarray,
    xy_padding: float,
    z_padding: float,
) -> np.ndarray:
    all_pts = raw_frame_data.reshape(-1, 3)
    bound_min = all_pts.min(axis=0).astype(np.float32)
    bound_max = all_pts.max(axis=0).astype(np.float32)
    bound_min[:2] -= xy_padding
    bound_max[:2] += xy_padding
    bound_min[2] -= z_padding
    bound_max[2] += z_padding
    span = np.maximum(bound_max - bound_min, 1e-3)
    bound_max = bound_min + span
    return np.stack((bound_min, bound_max), axis=0).astype(np.float32)


def compute_normalization_bound_from_points(
    points_xyz: np.ndarray,
    xy_padding: float,
    z_padding: float,
) -> np.ndarray:
    bound_min = points_xyz.min(axis=0).astype(np.float32)
    bound_max = points_xyz.max(axis=0).astype(np.float32)
    bound_min[:2] -= xy_padding
    bound_max[:2] += xy_padding
    bound_min[2] -= z_padding
    bound_max[2] += z_padding
    span = np.maximum(bound_max - bound_min, 1e-3)
    bound_max = bound_min + span
    return np.stack((bound_min, bound_max), axis=0).astype(np.float32)


def normalize_tensor_np(data: np.ndarray, bound: np.ndarray) -> np.ndarray:
    repeats = data.shape[1] // 3
    bound_len = np.tile(bound[1] - bound[0], repeats)
    bound_start = np.tile(bound[0], repeats)
    return ((data - bound_start) / bound_len - 0.5).astype(np.float32)


def calculate_speed_and_normal(raw_frame_data: np.ndarray, bound: np.ndarray, sample_min: float, sample_max: float) -> np.ndarray:
    x0 = raw_frame_data[:, :3]
    x1 = raw_frame_data[:, 3:6]
    p0 = raw_frame_data[:, 6:9]
    p1 = raw_frame_data[:, 9:12]

    n0 = x0 - p0
    n1 = x1 - p1
    y0 = np.linalg.norm(n0, axis=1, keepdims=True)
    y1 = np.linalg.norm(n1, axis=1, keepdims=True)

    valid = (y0[:, 0] > sample_min) & (y0[:, 0] <= sample_max)
    if not np.any(valid):
        return np.zeros((0, 14), dtype=np.float32)

    x0 = x0[valid]
    x1 = x1[valid]
    y0 = y0[valid]
    y1 = y1[valid]
    n0 = n0[valid]
    n1 = n1[valid]

    n0 = n0 / (y0 + 1e-8)
    n1 = n1 / (y1 + 1e-8)

    x0_n = normalize_tensor_np(x0.astype(np.float32), bound)
    x1_n = normalize_tensor_np(x1.astype(np.float32), bound)
    y0 = (np.clip(y0, sample_min, sample_max) / sample_max).astype(np.float32)
    y1 = (np.clip(y1, sample_min, sample_max) / sample_max).astype(np.float32)

    return np.concatenate((x0_n, x1_n, y0, y1, n0.astype(np.float32), n1.astype(np.float32)), axis=1)


def sample_training_frame(
    depth_m: np.ndarray,
    t_world_camera: np.ndarray,
    dirs_c: np.ndarray,
    n_rays: int,
    n_strat_samples: int,
    dist_behind_surf: float,
    min_depth: float,
    max_depth: float,
    sample_min: float,
    sample_max: float,
    num_pairs: int,
    scale_factor: float,
    bound_padding_xy: float,
    bound_padding_z: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    valid_mask = depth_m > min_depth
    valid_pixels = np.argwhere(valid_mask)
    if valid_pixels.size == 0:
        raise ValueError("No valid depth pixels found in frame.")

    replace = len(valid_pixels) < n_rays
    ray_pixels = valid_pixels[rng.choice(len(valid_pixels), size=n_rays, replace=replace)]

    rows = ray_pixels[:, 0]
    cols = ray_pixels[:, 1]
    depth_sample = depth_m[rows, cols]
    dirs_c_sample = dirs_c[rows, cols]

    rot_wc = t_world_camera[:3, :3]
    origin_w = t_world_camera[:3, 3]
    dirs_w = dirs_c_sample @ rot_wc.T

    max_sample_depth = np.minimum(depth_sample + dist_behind_surf, max_depth)
    bin_edges = np.linspace(0.0, 1.0, n_strat_samples + 1, dtype=np.float32)
    lower = min_depth + (max_sample_depth[:, None] - min_depth) * bin_edges[:-1][None, :]
    upper = min_depth + (max_sample_depth[:, None] - min_depth) * bin_edges[1:][None, :]
    z_strat = rng.uniform(lower, upper).astype(np.float32)

    z_vals = np.concatenate((depth_sample[:, None], z_strat), axis=1)
    pc = origin_w[None, None, :] + dirs_w[:, None, :] * z_vals[:, :, None]
    surf_pc = pc[:, 0, :]

    diff = pc[:, :, None, :] - surf_pc[None, None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    bounds = dists.min(axis=-1).astype(np.float32)
    bounds[z_vals > depth_sample[:, None]] *= -1.0

    pc_flat = pc.reshape(-1, 3).astype(np.float32)
    surf_pc_repeated = np.repeat(surf_pc[:, None, :], z_vals.shape[1], axis=1).reshape(-1, 3).astype(np.float32)
    bounds_flat = bounds.reshape(-1, 1).astype(np.float32)
    speeds_flat = np.clip(bounds_flat, sample_min, sample_max) / sample_max

    valid_indices = np.flatnonzero((speeds_flat[:, 0] > 0.0) & (speeds_flat[:, 0] <= 1.0))
    if valid_indices.size == 0:
        raise ValueError("No valid positive-bound samples generated from frame.")

    start_indices = rng.choice(valid_indices, size=num_pairs, replace=valid_indices.size < num_pairs)
    end_indices = rng.integers(0, pc_flat.shape[0], size=num_pairs)

    start_points = pc_flat[start_indices]
    end_points = pc_flat[end_indices]
    start_closest = surf_pc_repeated[start_indices]
    end_closest = surf_pc_repeated[end_indices]

    raw_frame_data = np.concatenate((start_points, end_points, start_closest, end_closest), axis=1).astype(np.float32)
    normalization_bound = compute_normalization_bound(raw_frame_data, bound_padding_xy, bound_padding_z)
    frame_data = calculate_speed_and_normal(raw_frame_data, normalization_bound, sample_min, sample_max)

    return {
        "frame_data": frame_data,
        "raw_frame_data": raw_frame_data.astype(np.float32),
        "normalization_bound": normalization_bound,
        "pc_world": pc_flat.astype(np.float32),
        "surf_pc_world": surf_pc.astype(np.float32),
        "ray_pixels": ray_pixels.astype(np.int32),
        "z_vals": z_vals.astype(np.float32),
        "depth_sample": depth_sample.astype(np.float32),
        "camera_pose_world": t_world_camera.astype(np.float32),
    }


def translation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))
