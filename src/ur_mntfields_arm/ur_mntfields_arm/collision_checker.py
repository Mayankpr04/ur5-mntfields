from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch

from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


class UR5PointCloudCollisionChecker:
    def __init__(
        self,
        kinematics: UR5Kinematics,
        occupied_points: np.ndarray,
        box_obstacles: np.ndarray | None = None,
        sphere_assets_root: str = "/home/mayank/ur_ws/ntrl-demo/datasets/arm/UR5/meshes/sphere/sphere",
        support_box_count: int = 0,
        support_point_ignore_padding_m: float = 0.15,
        attached_spheres_local: np.ndarray | None = None,
    ):
        self.kinematics = kinematics
        raw_occupied_points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 3)
        if box_obstacles is None:
            self.box_obstacles = np.zeros((0, 6), dtype=np.float64)
            valid = np.zeros((0,), dtype=bool)
        else:
            boxes = np.asarray(box_obstacles, dtype=np.float64).reshape(-1, 6)
            valid = np.all(np.isfinite(boxes), axis=1) & np.all(boxes[:, 3:] > 0.0, axis=1)
            self.box_obstacles = boxes[valid]
        # Support boxes are appended after ordinary scene boxes. The base
        # shoulder and the four proximal upper-arm spheres model the bolted
        # shoulder pivot, which intentionally touches the support. All distal
        # upper-arm and wrist geometry remains checked against it.
        requested_support_count = min(max(0, int(support_box_count)), len(valid))
        self.support_box_count = int(np.count_nonzero(valid[-requested_support_count:])) if requested_support_count else 0
        self.support_point_ignore_padding_m = float(max(0.0, support_point_ignore_padding_m))
        self.ignored_support_point_count = 0
        if self.support_box_count and len(raw_occupied_points):
            support_boxes = self.box_obstacles[-self.support_box_count :]
            support_delta = np.abs(raw_occupied_points[:, None, :] - support_boxes[None, :, :3])
            support_half_extents = 0.5 * support_boxes[None, :, 3:] + self.support_point_ignore_padding_m
            inside_support = np.any(np.all(support_delta <= support_half_extents, axis=2), axis=1)
            self.ignored_support_point_count = int(np.count_nonzero(inside_support))
            self.occupied_points = raw_occupied_points[~inside_support]
        else:
            self.occupied_points = raw_occupied_points
        self.kdtree = cKDTree(self.occupied_points) if len(self.occupied_points) else None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #print("Using device :" , self.device)
        self.sphere_assets_root = Path(sphere_assets_root)
        self.link_spheres_local = self._load_link_spheres()
        attached = np.asarray(
            np.zeros((0, 4), dtype=np.float64)
            if attached_spheres_local is None
            else attached_spheres_local,
            dtype=np.float64,
        ).reshape(-1, 4)
        if len(attached):
            valid_attached = np.all(np.isfinite(attached), axis=1) & (attached[:, 3] > 0.0)
            if not np.all(valid_attached):
                raise ValueError("attached_spheres_local must contain finite xyz and positive radii")
            # The final link frame is tool0, so attached geometry shares the
            # wrist_3 transform and contributes to the correct joint gradient.
            self.link_spheres_local[-1] = np.vstack((self.link_spheres_local[-1], attached))
        self._init_torch_buffers()
        voxel_size = 0.05
        if len(self.occupied_points):
            self.occupied_keys = {
                tuple(np.floor(p / voxel_size).astype(int).tolist())
                for p in self.occupied_points
            }
        else:
            self.occupied_keys = set()
        self.free_keys: set[tuple[int, int, int]] = set()
        self.voxel_size = voxel_size

    def _load_link_spheres(self) -> list[np.ndarray]:
        link_names = [
            "shoulder_link.npy",
            "upper_arm_link.npy",
            "forearm_link.npy",
            "wrist_1_link.npy",
            "wrist_2_link.npy",
            "wrist_3_link.npy",
        ]
        out: list[np.ndarray] = []
        for name in link_names:
            path = self.sphere_assets_root / name
            arr = np.load(path).astype(np.float64)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"Unexpected sphere asset format: {path} shape={arr.shape}")
            out.append(arr)
        return out

    def _init_torch_buffers(self):
        self.base_link_from_dh_t = torch.as_tensor(
            self.kinematics.base_link_from_dh, dtype=torch.float32, device=self.device
        )
        self.a_t = torch.as_tensor(self.kinematics.a, dtype=torch.float32, device=self.device)
        self.d_t = torch.as_tensor(self.kinematics.d, dtype=torch.float32, device=self.device)
        self.alpha_t = torch.as_tensor(self.kinematics.alpha, dtype=torch.float32, device=self.device)
        self.joint_min_t = torch.as_tensor(self.kinematics.joint_min, dtype=torch.float32, device=self.device)
        self.joint_max_t = torch.as_tensor(self.kinematics.joint_max, dtype=torch.float32, device=self.device)
        self.link_spheres_local_t = [
            torch.as_tensor(arr, dtype=torch.float32, device=self.device) for arr in self.link_spheres_local
        ]
        sphere_count = sum(int(len(spheres)) for spheres in self.link_spheres_local)
        self.support_contact_sphere_mask_t = torch.zeros((sphere_count,), dtype=torch.bool, device=self.device)
        if sphere_count:
            self.support_contact_sphere_mask_t[0] = True
        if len(self.link_spheres_local) > 1 and len(self.link_spheres_local[1]):
            upper_arm_offset = len(self.link_spheres_local[0])
            proximal_count = min(4, len(self.link_spheres_local[1]))
            self.support_contact_sphere_mask_t[upper_arm_offset : upper_arm_offset + proximal_count] = True
        self.urdf_joint_origin_t = [
            self._make_transform_torch((0.0, 0.0, float(self.kinematics.d[0])), (0.0, 0.0, 0.0)),
            self._make_transform_torch((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            self._make_transform_torch((float(self.kinematics.a[1]), 0.0, 0.0), (0.0, 0.0, 0.0)),
            self._make_transform_torch((float(self.kinematics.a[2]), 0.0, float(self.kinematics.d[3])), (0.0, 0.0, 0.0)),
            self._make_transform_torch((0.0, -float(self.kinematics.d[4]), 0.0), (math.pi / 2.0, 0.0, 0.0)),
            self._make_transform_torch((0.0, float(self.kinematics.d[5]), 0.0), (math.pi / 2.0, math.pi, math.pi)),
        ]
        if len(self.occupied_points):
            self.occupied_points_t = torch.as_tensor(self.occupied_points, dtype=torch.float32, device=self.device)
        else:
            self.occupied_points_t = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        if len(self.box_obstacles):
            boxes = torch.as_tensor(self.box_obstacles, dtype=torch.float32, device=self.device)
            self.box_centers_t = boxes[:, :3]
            self.box_half_extents_t = 0.5 * boxes[:, 3:]
            self.support_box_mask_t = torch.zeros((len(self.box_obstacles),), dtype=torch.bool, device=self.device)
            if self.support_box_count:
                self.support_box_mask_t[-self.support_box_count :] = True
        else:
            self.box_centers_t = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            self.box_half_extents_t = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            self.support_box_mask_t = torch.zeros((0,), dtype=torch.bool, device=self.device)

    def _box_distance_candidates(
        self,
        signed_dist: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Mask only fixed support contact before selecting a box."""
        if self.support_box_count <= 0:
            return signed_dist
        ignore_support = self.support_contact_sphere_mask_t.repeat(int(batch_size))
        return signed_dist.masked_fill(
            ignore_support[:, None] & self.support_box_mask_t[None, :],
            float("inf"),
        )

    def _make_transform_torch(self, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> torch.Tensor:
        roll, pitch, yaw = rpy
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        rot = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = rot
        out[:3, 3] = np.asarray(xyz, dtype=np.float32)
        return torch.as_tensor(out, dtype=torch.float32, device=self.device)

    def _rot_z_batch(self, theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta)
        s = torch.sin(theta)
        out = torch.zeros((theta.shape[0], 4, 4), dtype=torch.float32, device=theta.device)
        out[:, 0, 0] = c
        out[:, 0, 1] = -s
        out[:, 1, 0] = s
        out[:, 1, 1] = c
        out[:, 2, 2] = 1.0
        out[:, 3, 3] = 1.0
        return out

    def _dh_batch(self, a: torch.Tensor, d: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        out = torch.zeros((theta.shape[0], 4, 4), dtype=torch.float32, device=theta.device)
        out[:, 0, 0] = ct
        out[:, 0, 1] = -st * ca
        out[:, 0, 2] = st * sa
        out[:, 0, 3] = a * ct
        out[:, 1, 0] = st
        out[:, 1, 1] = ct * ca
        out[:, 1, 2] = -ct * sa
        out[:, 1, 3] = a * st
        out[:, 2, 1] = sa
        out[:, 2, 2] = ca
        out[:, 2, 3] = d
        out[:, 3, 3] = 1.0
        return out

    def _batch_fk_info_torch(self, q_batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        q_batch = torch.maximum(torch.minimum(q_batch, self.joint_max_t), self.joint_min_t)
        batch_size = int(q_batch.shape[0])
        cur = self.base_link_from_dh_t.unsqueeze(0).expand(batch_size, -1, -1).clone()
        joint_frames_before: list[torch.Tensor] = []
        link_frames: list[torch.Tensor] = []
        for idx in range(6):
            origin = self.urdf_joint_origin_t[idx].unsqueeze(0).expand(batch_size, -1, -1)
            joint_frame = torch.bmm(cur, origin)
            joint_frames_before.append(joint_frame.clone())
            cur = torch.bmm(joint_frame, self._rot_z_batch(q_batch[:, idx]))
            link_frames.append(cur.clone())
        return joint_frames_before, link_frames

    def _sphere_samples_batch_torch(
        self, q_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        joint_frames_before, link_frames = self._batch_fk_info_torch(q_batch)
        batch_size = int(q_batch.shape[0])
        centers_world = []
        radii = []
        link_ids = []
        for idx, spheres in enumerate(self.link_spheres_local_t):
            cur = link_frames[idx]
            local_xyz = spheres[:, :3]
            local_r = spheres[:, 3]
            rot = cur[:, :3, :3]
            trans = cur[:, :3, 3]
            world_xyz = torch.matmul(rot, local_xyz.t()).transpose(1, 2) + trans[:, None, :]
            centers_world.append(world_xyz)
            radii.append(local_r.unsqueeze(0).expand(batch_size, -1))
            link_ids.append(torch.full((world_xyz.shape[1],), idx + 1, dtype=torch.long, device=self.device))
        return (
            torch.cat(centers_world, dim=1),
            torch.cat(radii, dim=1),
            torch.cat(link_ids, dim=0),
            joint_frames_before,
        )

    def clearance_and_normal_batch(
        self, q_batch: np.ndarray, obstacle_chunk_size: int = 2048
    ) -> tuple[np.ndarray, np.ndarray]:
        q_batch = np.asarray(q_batch, dtype=np.float32)
        if q_batch.ndim == 1:
            q_batch = q_batch[None, :]
        if len(q_batch) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
        if (self.kdtree is None or len(self.occupied_points) == 0) and len(self.box_obstacles) == 0:
            return np.ones((len(q_batch),), dtype=np.float32), np.zeros((len(q_batch), 6), dtype=np.float32)
        with torch.no_grad():
            q_t = torch.as_tensor(q_batch, dtype=torch.float32, device=self.device)
            centers, radii, sphere_link_ids, joint_frames_before = self._sphere_samples_batch_torch(q_t)
            flat_centers = centers.reshape(-1, 3)
            flat_count = int(flat_centers.shape[0])

            point_dists = torch.full((flat_count,), float("inf"), dtype=torch.float32, device=self.device)
            point_inds = torch.zeros((flat_count,), dtype=torch.long, device=self.device)
            if int(self.occupied_points_t.shape[0]) > 0:
                for start in range(0, int(self.occupied_points_t.shape[0]), obstacle_chunk_size):
                    obs_chunk = self.occupied_points_t[start:start + obstacle_chunk_size]
                    d_chunk = torch.cdist(flat_centers, obs_chunk, p=2.0)
                    chunk_min_vals, chunk_min_inds = torch.min(d_chunk, dim=1)
                    chunk_min_inds = chunk_min_inds + start
                    better = chunk_min_vals < point_dists
                    point_dists = torch.where(better, chunk_min_vals, point_dists)
                    point_inds = torch.where(better, chunk_min_inds, point_inds)

            box_dists = torch.full((flat_count,), float("inf"), dtype=torch.float32, device=self.device)
            box_normals = torch.zeros((flat_count, 3), dtype=torch.float32, device=self.device)
            if int(self.box_centers_t.shape[0]) > 0:
                delta = flat_centers[:, None, :] - self.box_centers_t[None, :, :]
                abs_delta = torch.abs(delta)
                q_box = abs_delta - self.box_half_extents_t[None, :, :]
                outside = torch.clamp(q_box, min=0.0)
                outside_norm = torch.linalg.norm(outside, dim=2)
                max_q, _ = torch.max(q_box, dim=2)
                signed_dist = outside_norm + torch.minimum(max_q, torch.zeros_like(max_q))
                box_distance_candidates = self._box_distance_candidates(
                    signed_dist, centers.shape[0]
                )
                box_dists, box_inds = torch.min(box_distance_candidates, dim=1)

                chosen_delta = delta[torch.arange(flat_count, device=self.device), box_inds]
                chosen_q = q_box[torch.arange(flat_count, device=self.device), box_inds]
                chosen_outside = outside[torch.arange(flat_count, device=self.device), box_inds]
                chosen_outside_norm = outside_norm[torch.arange(flat_count, device=self.device), box_inds]
                sign = torch.where(chosen_delta >= 0.0, torch.ones_like(chosen_delta), -torch.ones_like(chosen_delta))
                outside_normal = sign * chosen_outside / torch.clamp(chosen_outside_norm[:, None], min=1e-8)
                face_idx = torch.argmax(chosen_q, dim=1)
                inside_normal = torch.zeros_like(outside_normal)
                inside_normal[torch.arange(flat_count, device=self.device), face_idx] = sign[
                    torch.arange(flat_count, device=self.device), face_idx
                ]
                is_outside = chosen_outside_norm > 1e-8
                box_normals = torch.where(is_outside[:, None], outside_normal, inside_normal)

            use_box = box_dists < point_dists
            obstacle_dists = torch.where(use_box, box_dists, point_dists)
            sphere_clearances = obstacle_dists.reshape(centers.shape[0], centers.shape[1]) - radii
            sphere_dists = obstacle_dists.reshape(centers.shape[0], centers.shape[1])
            argmin_sphere = torch.argmin(sphere_clearances, dim=1)
            clearances = torch.clamp(torch.gather(sphere_clearances, 1, argmin_sphere[:, None]).squeeze(1), min=0.0)

            batch_idx = torch.arange(centers.shape[0], device=self.device)
            closest_centers = centers[batch_idx, argmin_sphere]
            closest_dist = sphere_dists[batch_idx, argmin_sphere]
            flat_sphere_idx = batch_idx * centers.shape[1] + argmin_sphere
            selected_use_box = use_box[flat_sphere_idx]
            closest_obs_inds = point_inds.reshape(centers.shape[0], centers.shape[1])[batch_idx, argmin_sphere]
            closest_obs = self.occupied_points_t[closest_obs_inds] if int(self.occupied_points_t.shape[0]) > 0 else closest_centers
            point_normal = closest_centers - closest_obs
            workspace_normal = torch.where(selected_use_box[:, None], box_normals[flat_sphere_idx], point_normal)
            ws_norm = torch.linalg.norm(workspace_normal, dim=1, keepdim=True)
            workspace_normal = torch.where(
                ws_norm > 1e-8, workspace_normal / torch.clamp(ws_norm, min=1e-8), torch.zeros_like(workspace_normal)
            )
            closest_link_ids = sphere_link_ids[argmin_sphere]

            joint_grad = torch.zeros((centers.shape[0], 6), dtype=torch.float32, device=self.device)
            for j in range(6):
                frame = joint_frames_before[j]
                joint_origin = frame[:, :3, 3]
                joint_axis = frame[:, :3, 2]
                deriv = torch.cross(joint_axis, closest_centers - joint_origin, dim=1)
                contrib = torch.sum(deriv * workspace_normal, dim=1)
                active = closest_link_ids > j
                joint_grad[:, j] = torch.where(active, contrib, torch.zeros_like(contrib))
            grad_norm = torch.linalg.norm(joint_grad, dim=1, keepdim=True)
            joint_grad = torch.where(
                grad_norm > 1e-8, joint_grad / torch.clamp(grad_norm, min=1e-8), torch.zeros_like(joint_grad)
            )
            # Point-cloud contacts can have an undefined normal exactly at the
            # nearest point. Box SDF contacts still have a valid face normal
            # even inside the box, so do not suppress those penetration labels.
            zero_mask = ((~selected_use_box) & (closest_dist <= 1e-8))[:, None]
            joint_grad = torch.where(zero_mask, torch.zeros_like(joint_grad), joint_grad)
            return (
                clearances.detach().cpu().numpy().astype(np.float32),
                joint_grad.detach().cpu().numpy().astype(np.float32),
            )

    def clearance_batch(self, q_batch: np.ndarray, obstacle_chunk_size: int = 2048) -> np.ndarray:
        clearances, _ = self.clearance_and_normal_batch(q_batch, obstacle_chunk_size=obstacle_chunk_size)
        return clearances

    def clearance_gradient_batch(self, q_batch: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        _clearances, normals = self.clearance_and_normal_batch(q_batch)
        return normals

    def _sphere_samples(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transforms = self._urdf_link_frames_np(q)
        centers_world = []
        radii = []
        for idx, spheres in enumerate(self.link_spheres_local):
            tf = transforms[idx]
            local_xyz = spheres[:, :3]
            local_r = spheres[:, 3]
            world_xyz = (tf[:3, :3] @ local_xyz.T).T + tf[:3, 3][None, :]
            centers_world.append(world_xyz)
            radii.append(local_r)
        return np.concatenate(centers_world, axis=0), np.concatenate(radii, axis=0)

    def robot_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._sphere_samples(q)

    def robot_spheres_batch(self, q_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q_batch, dtype=np.float32).reshape(-1, 6)
        if len(q) == 0:
            return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        with torch.no_grad():
            centers, radii, _link_ids, _frames = self._sphere_samples_batch_torch(
                torch.as_tensor(q, dtype=torch.float32, device=self.device)
            )
        return (
            centers.detach().cpu().numpy().astype(np.float32),
            radii.detach().cpu().numpy().astype(np.float32),
        )

    def filter_robot_self_points(
        self,
        points: np.ndarray,
        q: np.ndarray,
        padding_m: float = 0.04,
        chunk_size: int = 4096,
        extra_spheres: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32), 0
        centers, radii = self._sphere_samples(q)
        if extra_spheres is not None:
            extra = np.asarray(extra_spheres, dtype=np.float64).reshape(-1, 4)
            valid = np.all(np.isfinite(extra), axis=1) & (extra[:, 3] > 0.0)
            if np.any(valid):
                centers = np.vstack((centers, extra[valid, :3]))
                radii = np.concatenate((radii, extra[valid, 3]))
        if len(centers) == 0:
            return pts.astype(np.float32, copy=False), 0
        inflated = np.asarray(radii, dtype=np.float64) + max(0.0, float(padding_m))
        keep = np.ones((len(pts),), dtype=bool)
        step = max(1, int(chunk_size))
        for start in range(0, len(pts), step):
            chunk = pts[start:start + step]
            dist = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2) - inflated[None, :]
            keep[start:start + step] = np.min(dist, axis=1) > 0.0
        filtered = pts[keep].astype(np.float32, copy=False)
        return filtered, int(len(pts) - len(filtered))

    def robot_self_occlusion_fraction(
        self,
        q: np.ndarray,
        origin: np.ndarray,
        targets: np.ndarray,
        padding_m: float = 0.03,
        ignore_near_origin_m: float = 0.16,
        max_targets: int = 64,
        extra_spheres: np.ndarray | None = None,
    ) -> float:
        target_pts = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        if len(target_pts) == 0:
            return 0.0
        if len(target_pts) > max_targets:
            stride = max(1, int(math.ceil(len(target_pts) / max(1, int(max_targets)))))
            target_pts = target_pts[::stride][:max_targets]
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        centers, radii = self._sphere_samples(q)
        if extra_spheres is not None:
            extra = np.asarray(extra_spheres, dtype=np.float64).reshape(-1, 4)
            valid = np.all(np.isfinite(extra), axis=1) & (extra[:, 3] > 0.0)
            if np.any(valid):
                centers = np.vstack((centers, extra[valid, :3]))
                radii = np.concatenate((radii, extra[valid, 3]))
        if len(centers) == 0:
            return 0.0
        inflated = np.asarray(radii, dtype=np.float64) + max(0.0, float(padding_m))
        center_dist = np.linalg.norm(centers - origin[None, :], axis=1)
        use = center_dist > max(0.0, float(ignore_near_origin_m))
        centers = centers[use]
        inflated = inflated[use]
        if len(centers) == 0:
            return 0.0
        occluded = 0
        for target in target_pts:
            ray = target - origin
            ray_len2 = float(np.dot(ray, ray))
            if ray_len2 < 1.0e-10:
                continue
            alpha = np.clip(((centers - origin[None, :]) @ ray) / ray_len2, 0.03, 0.97)
            closest = origin[None, :] + alpha[:, None] * ray[None, :]
            clearance = np.linalg.norm(centers - closest, axis=1) - inflated
            if float(np.min(clearance)) <= 0.0:
                occluded += 1
        return float(occluded) / float(max(1, len(target_pts)))

    def _rpy_matrix_np(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _transform_np(self, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self._rpy_matrix_np(*rpy)
        out[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return out

    def _rotz_np(self, theta: float) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        c = math.cos(theta)
        s = math.sin(theta)
        out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return out

    def _urdf_link_frames_np(self, q: np.ndarray) -> list[np.ndarray]:
        q = self.kinematics.clamp(np.asarray(q, dtype=np.float64))
        origins = [
            self._transform_np((0.0, 0.0, float(self.kinematics.d[0])), (0.0, 0.0, 0.0)),
            self._transform_np((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            self._transform_np((float(self.kinematics.a[1]), 0.0, 0.0), (0.0, 0.0, 0.0)),
            self._transform_np((float(self.kinematics.a[2]), 0.0, float(self.kinematics.d[3])), (0.0, 0.0, 0.0)),
            self._transform_np((0.0, -float(self.kinematics.d[4]), 0.0), (math.pi / 2.0, 0.0, 0.0)),
            self._transform_np((0.0, float(self.kinematics.d[5]), 0.0), (math.pi / 2.0, math.pi, math.pi)),
        ]
        cur = self.kinematics.base_link_from_dh.copy()
        frames = []
        for idx in range(6):
            cur = cur @ origins[idx] @ self._rotz_np(float(q[idx]))
            frames.append(cur.copy())
        return frames

    def key_to_center(self, key: tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(key, dtype=np.float64) + 0.5) * self.voxel_size

    def clearance(self, q: np.ndarray) -> float:
        return float(self.clearance_batch(np.asarray(q, dtype=np.float32))[0])

    def clearance_gradient(self, q: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        return self.clearance_gradient_batch(np.asarray(q, dtype=np.float32), eps=eps)[0].astype(np.float64)


class UR5SDFCollisionChecker(UR5PointCloudCollisionChecker):
    """UR5 sphere collision checker backed by a voxel EDT for point-cloud obstacles.

    The UR5 sphere FK and box-obstacle handling remain identical to
    UR5PointCloudCollisionChecker. Only nearest-point queries against the
    observed point cloud are replaced by one SDF lookup per sphere.
    """

    def __init__(
        self,
        kinematics: UR5Kinematics,
        occupied_points: np.ndarray,
        box_obstacles: np.ndarray | None = None,
        sphere_assets_root: str = "/home/mayank/ur_ws/ntrl-demo/datasets/arm/UR5/meshes/sphere/sphere",
        sdf_voxel_size_m: float = 0.04,
        sdf_padding_m: float = 0.75,
        sdf_max_cells: int = 4_000_000,
        support_box_count: int = 0,
        support_point_ignore_padding_m: float = 0.15,
        attached_spheres_local: np.ndarray | None = None,
    ):
        super().__init__(
            kinematics=kinematics,
            occupied_points=occupied_points,
            box_obstacles=box_obstacles,
            support_box_count=support_box_count,
            support_point_ignore_padding_m=support_point_ignore_padding_m,
            sphere_assets_root=sphere_assets_root,
            attached_spheres_local=attached_spheres_local,
        )
        self.sdf_voxel_size_m = float(max(1.0e-3, sdf_voxel_size_m))
        self.sdf_padding_m = float(max(0.0, sdf_padding_m))
        self.sdf_max_cells = int(max(10_000, sdf_max_cells))
        self.sdf_origin: np.ndarray | None = None
        self.sdf_upper: np.ndarray | None = None
        self.sdf_grid: np.ndarray | None = None
        self.sdf_grad_grid: np.ndarray | None = None
        self.sdf_grid_t: torch.Tensor | None = None
        self.sdf_grad_grid_t: torch.Tensor | None = None
        self.sdf_effective_voxel_size_m = self.sdf_voxel_size_m
        self._build_point_sdf()

    def _build_point_sdf(self) -> None:
        pts = np.asarray(self.occupied_points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            return

        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        if len(self.box_obstacles):
            boxes = np.asarray(self.box_obstacles, dtype=np.float64).reshape(-1, 6)
            box_lo = boxes[:, :3] - 0.5 * boxes[:, 3:]
            box_hi = boxes[:, :3] + 0.5 * boxes[:, 3:]
            lo = np.minimum(lo, np.min(box_lo, axis=0))
            hi = np.maximum(hi, np.max(box_hi, axis=0))
        lo = lo - self.sdf_padding_m
        hi = hi + self.sdf_padding_m

        voxel = self.sdf_voxel_size_m
        dims = np.ceil((hi - lo) / voxel).astype(int) + 3
        cell_count = int(np.prod(np.maximum(dims, 1)))
        if cell_count > self.sdf_max_cells:
            scale = (float(cell_count) / float(self.sdf_max_cells)) ** (1.0 / 3.0)
            voxel = float(voxel * scale)
            dims = np.ceil((hi - lo) / voxel).astype(int) + 3

        dims = np.maximum(dims, 3).astype(int)
        occ = np.zeros(tuple(int(v) for v in dims), dtype=bool)
        idx = np.floor((pts - lo[None, :]) / voxel).astype(int)
        idx = np.clip(idx, 0, dims[None, :] - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

        dist = distance_transform_edt(~occ, sampling=(voxel, voxel, voxel)).astype(np.float32)
        grad = np.stack(np.gradient(dist, voxel, edge_order=1), axis=-1).astype(np.float32)

        self.sdf_origin = lo.astype(np.float64)
        self.sdf_upper = (lo + (dims.astype(np.float64) - 1.0) * voxel).astype(np.float64)
        self.sdf_grid = dist
        self.sdf_grad_grid = grad
        # High-volume online labels already compute robot FK on this device.
        # Keep the lookup tables beside it so every batch does not bounce all
        # sphere centers GPU -> NumPy -> GPU.
        self.sdf_grid_t = torch.as_tensor(dist, dtype=torch.float32, device=self.device)
        self.sdf_grad_grid_t = torch.as_tensor(grad, dtype=torch.float32, device=self.device)
        self.sdf_effective_voxel_size_m = float(voxel)

    def _lookup_trilinear_torch(
        self, grid: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        if self.sdf_origin is None:
            raise RuntimeError("SDF origin is not initialized")
        origin = torch.as_tensor(
            self.sdf_origin, dtype=points.dtype, device=points.device
        )
        dims = torch.as_tensor(
            grid.shape[:3], dtype=torch.long, device=points.device
        )
        coords = (points - origin[None, :]) / self.sdf_effective_voxel_size_m
        base_unclamped = torch.floor(coords).to(torch.long)
        frac = torch.clamp(coords - base_unclamped.to(coords.dtype), 0.0, 1.0)
        base = torch.minimum(
            torch.maximum(base_unclamped, torch.zeros_like(base_unclamped)),
            dims[None, :] - 2,
        )
        i0, j0, k0 = base.unbind(dim=1)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        fx, fy, fz = frac.unbind(dim=1)
        vector_grid = grid.ndim == 4
        if vector_grid:
            fx, fy, fz = fx[:, None], fy[:, None], fz[:, None]
        c000, c100 = grid[i0, j0, k0], grid[i1, j0, k0]
        c010, c110 = grid[i0, j1, k0], grid[i1, j1, k0]
        c001, c101 = grid[i0, j0, k1], grid[i1, j0, k1]
        c011, c111 = grid[i0, j1, k1], grid[i1, j1, k1]
        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz) + c1 * fz

    def _point_sdf_lookup_torch(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.sdf_grid_t is None
            or self.sdf_grad_grid_t is None
            or self.sdf_origin is None
            or self.sdf_upper is None
        ):
            return (
                torch.full((len(points),), float("inf"), dtype=points.dtype, device=points.device),
                torch.zeros((len(points), 3), dtype=points.dtype, device=points.device),
            )
        distance = self._lookup_trilinear_torch(self.sdf_grid_t, points)
        gradient = self._lookup_trilinear_torch(self.sdf_grad_grid_t, points)
        lower = torch.as_tensor(self.sdf_origin, dtype=points.dtype, device=points.device)
        upper = torch.as_tensor(self.sdf_upper, dtype=points.dtype, device=points.device)
        below = torch.clamp(lower[None, :] - points, min=0.0)
        above = torch.clamp(points - upper[None, :], min=0.0)
        outside = above - below
        outside_norm = torch.linalg.vector_norm(outside, dim=1)
        outside_mask = outside_norm > 1.0e-8
        distance = distance + outside_norm
        outside_normal = outside / outside_norm[:, None].clamp_min(1.0e-8)
        gradient = torch.where(outside_mask[:, None], outside_normal, gradient)
        norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
        gradient = torch.where(
            norm > 1.0e-8, gradient / norm.clamp_min(1.0e-8), torch.zeros_like(gradient)
        )
        return distance, gradient

    def _lookup_trilinear(self, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        origin = self.sdf_origin
        if origin is None:
            raise RuntimeError("SDF origin is not initialized")
        voxel = self.sdf_effective_voxel_size_m
        dims = np.asarray(grid.shape[:3], dtype=np.int64)
        coords = (points - origin[None, :]) / voxel
        base = np.floor(coords).astype(np.int64)
        frac = coords - base.astype(np.float64)
        base = np.clip(base, 0, dims[None, :] - 2)
        frac = np.clip(frac, 0.0, 1.0)

        i0, j0, k0 = base[:, 0], base[:, 1], base[:, 2]
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        c000 = grid[i0, j0, k0]
        c100 = grid[i1, j0, k0]
        c010 = grid[i0, j1, k0]
        c110 = grid[i1, j1, k0]
        c001 = grid[i0, j0, k1]
        c101 = grid[i1, j0, k1]
        c011 = grid[i0, j1, k1]
        c111 = grid[i1, j1, k1]

        c00 = c000 * (1.0 - fx)[:, None] + c100 * fx[:, None] if grid.ndim == 4 else c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx)[:, None] + c110 * fx[:, None] if grid.ndim == 4 else c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx)[:, None] + c101 * fx[:, None] if grid.ndim == 4 else c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx)[:, None] + c111 * fx[:, None] if grid.ndim == 4 else c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy)[:, None] + c10 * fy[:, None] if grid.ndim == 4 else c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy)[:, None] + c11 * fy[:, None] if grid.ndim == 4 else c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz)[:, None] + c1 * fz[:, None] if grid.ndim == 4 else c0 * (1.0 - fz) + c1 * fz

    def _point_sdf_lookup(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.sdf_grid is None or self.sdf_grad_grid is None or self.sdf_origin is None or self.sdf_upper is None:
            return (
                np.full((len(points),), np.inf, dtype=np.float32),
                np.zeros((len(points), 3), dtype=np.float32),
            )
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        d = self._lookup_trilinear(self.sdf_grid, pts).astype(np.float32)
        g = self._lookup_trilinear(self.sdf_grad_grid, pts).astype(np.float32)

        below = np.maximum(self.sdf_origin[None, :] - pts, 0.0)
        above = np.maximum(pts - self.sdf_upper[None, :], 0.0)
        outside = above - below
        outside_norm = np.linalg.norm(outside, axis=1)
        outside_mask = outside_norm > 1.0e-8
        if np.any(outside_mask):
            d[outside_mask] = d[outside_mask] + outside_norm[outside_mask].astype(np.float32)
            g[outside_mask] = outside[outside_mask] / outside_norm[outside_mask, None]

        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        g = np.where(g_norm > 1.0e-8, g / np.maximum(g_norm, 1.0e-8), 0.0).astype(np.float32)
        return d, g

    def clearance_and_normal_batch(
        self, q_batch: np.ndarray, obstacle_chunk_size: int = 2048
    ) -> tuple[np.ndarray, np.ndarray]:
        q_batch = np.asarray(q_batch, dtype=np.float32)
        if q_batch.ndim == 1:
            q_batch = q_batch[None, :]
        if len(q_batch) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
        if self.sdf_grid is None and len(self.box_obstacles) == 0:
            return np.ones((len(q_batch),), dtype=np.float32), np.zeros((len(q_batch), 6), dtype=np.float32)

        with torch.no_grad():
            q_t = torch.as_tensor(q_batch, dtype=torch.float32, device=self.device)
            centers, radii, sphere_link_ids, joint_frames_before = self._sphere_samples_batch_torch(q_t)
            flat_centers = centers.reshape(-1, 3)
            flat_count = int(flat_centers.shape[0])

            point_dists, point_normals = self._point_sdf_lookup_torch(flat_centers)

            box_dists = torch.full((flat_count,), float("inf"), dtype=torch.float32, device=self.device)
            box_normals = torch.zeros((flat_count, 3), dtype=torch.float32, device=self.device)
            if int(self.box_centers_t.shape[0]) > 0:
                delta = flat_centers[:, None, :] - self.box_centers_t[None, :, :]
                abs_delta = torch.abs(delta)
                q_box = abs_delta - self.box_half_extents_t[None, :, :]
                outside = torch.clamp(q_box, min=0.0)
                outside_norm = torch.linalg.norm(outside, dim=2)
                max_q, _ = torch.max(q_box, dim=2)
                signed_dist = outside_norm + torch.minimum(max_q, torch.zeros_like(max_q))
                box_distance_candidates = self._box_distance_candidates(
                    signed_dist, centers.shape[0]
                )
                box_dists, box_inds = torch.min(box_distance_candidates, dim=1)

                chosen_delta = delta[torch.arange(flat_count, device=self.device), box_inds]
                chosen_q = q_box[torch.arange(flat_count, device=self.device), box_inds]
                chosen_outside = outside[torch.arange(flat_count, device=self.device), box_inds]
                chosen_outside_norm = outside_norm[torch.arange(flat_count, device=self.device), box_inds]
                sign = torch.where(chosen_delta >= 0.0, torch.ones_like(chosen_delta), -torch.ones_like(chosen_delta))
                outside_normal = sign * chosen_outside / torch.clamp(chosen_outside_norm[:, None], min=1e-8)
                face_idx = torch.argmax(chosen_q, dim=1)
                inside_normal = torch.zeros_like(outside_normal)
                inside_normal[torch.arange(flat_count, device=self.device), face_idx] = sign[
                    torch.arange(flat_count, device=self.device), face_idx
                ]
                is_outside = chosen_outside_norm > 1e-8
                box_normals = torch.where(is_outside[:, None], outside_normal, inside_normal)

            use_box = box_dists < point_dists
            obstacle_dists = torch.where(use_box, box_dists, point_dists)
            sphere_clearances = obstacle_dists.reshape(centers.shape[0], centers.shape[1]) - radii
            sphere_dists = obstacle_dists.reshape(centers.shape[0], centers.shape[1])
            argmin_sphere = torch.argmin(sphere_clearances, dim=1)
            clearances = torch.clamp(torch.gather(sphere_clearances, 1, argmin_sphere[:, None]).squeeze(1), min=0.0)

            batch_idx = torch.arange(centers.shape[0], device=self.device)
            closest_centers = centers[batch_idx, argmin_sphere]
            closest_dist = sphere_dists[batch_idx, argmin_sphere]
            flat_sphere_idx = batch_idx * centers.shape[1] + argmin_sphere
            selected_use_box = use_box[flat_sphere_idx]
            selected_point_normal = point_normals[flat_sphere_idx]
            workspace_normal = torch.where(selected_use_box[:, None], box_normals[flat_sphere_idx], selected_point_normal)
            ws_norm = torch.linalg.norm(workspace_normal, dim=1, keepdim=True)
            workspace_normal = torch.where(
                ws_norm > 1e-8, workspace_normal / torch.clamp(ws_norm, min=1e-8), torch.zeros_like(workspace_normal)
            )
            closest_link_ids = sphere_link_ids[argmin_sphere]

            joint_grad = torch.zeros((centers.shape[0], 6), dtype=torch.float32, device=self.device)
            for j in range(6):
                frame = joint_frames_before[j]
                joint_origin = frame[:, :3, 3]
                joint_axis = frame[:, :3, 2]
                deriv = torch.cross(joint_axis, closest_centers - joint_origin, dim=1)
                contrib = torch.sum(deriv * workspace_normal, dim=1)
                active = closest_link_ids > j
                joint_grad[:, j] = torch.where(active, contrib, torch.zeros_like(contrib))
            grad_norm = torch.linalg.norm(joint_grad, dim=1, keepdim=True)
            joint_grad = torch.where(
                grad_norm > 1e-8, joint_grad / torch.clamp(grad_norm, min=1e-8), torch.zeros_like(joint_grad)
            )
            zero_mask = ((~selected_use_box) & (closest_dist <= 1e-8))[:, None]
            joint_grad = torch.where(zero_mask, torch.zeros_like(joint_grad), joint_grad)
            return (
                clearances.detach().cpu().numpy().astype(np.float32),
                joint_grad.detach().cpu().numpy().astype(np.float32),
            )


def make_ur5_collision_checker(
    kinematics: UR5Kinematics,
    occupied_points: np.ndarray,
    box_obstacles: np.ndarray | None = None,
    clearance_backend: str = "original",
    sdf_voxel_size_m: float = 0.04,
    sdf_padding_m: float = 0.75,
    sdf_max_cells: int = 4_000_000,
    support_box_count: int = 0,
    support_point_ignore_padding_m: float = 0.15,
    attached_spheres_local: np.ndarray | None = None,
) -> UR5PointCloudCollisionChecker:
    backend = str(clearance_backend or "original").strip().lower()
    if backend in {"sdf", "edt", "sdf_edt"}:
        return UR5SDFCollisionChecker(
            kinematics,
            occupied_points,
            box_obstacles=box_obstacles,
            support_box_count=support_box_count,
            support_point_ignore_padding_m=support_point_ignore_padding_m,
            attached_spheres_local=attached_spheres_local,
            sdf_voxel_size_m=sdf_voxel_size_m,
            sdf_padding_m=sdf_padding_m,
            sdf_max_cells=sdf_max_cells,
        )
    if backend not in {"original", "pointcloud", "point_cloud", "cdist"}:
        raise ValueError(f"Unknown clearance_backend={clearance_backend!r}; use 'original' or 'sdf'.")
    return UR5PointCloudCollisionChecker(
        kinematics,
        occupied_points,
        box_obstacles=box_obstacles,
        support_box_count=support_box_count,
        support_point_ignore_padding_m=support_point_ignore_padding_m,
        attached_spheres_local=attached_spheres_local,
    )


def wrist_camera_collision_spheres(camera_in_tool: np.ndarray) -> np.ndarray:
    """Conservatively represent the wrist camera body and its small bracket."""

    transform = np.asarray(camera_in_tool, dtype=np.float64).reshape(4, 4)
    center = transform[:3, 3]
    # Exact circumsphere radius of the URDF camera collision box.
    radius = float(np.linalg.norm(np.asarray([0.02, 0.02, 0.01], dtype=np.float64)))
    bracket_centers = np.asarray(
        [0.35 * center, 0.70 * center],
        dtype=np.float64,
    )
    bracket = np.column_stack((bracket_centers, np.full((2,), 0.015, dtype=np.float64)))
    return np.vstack((bracket, np.asarray([[center[0], center[1], center[2], radius]])))
