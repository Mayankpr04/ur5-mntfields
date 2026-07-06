from __future__ import annotations

import math
import numpy as np
from sensor_msgs.msg import CameraInfo

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.frontier_bank import FrontierRecord
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics, ViewGoal, _transform, look_at_rotation


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class ViewGoalSelector:
    def __init__(
        self,
        kinematics: UR5Kinematics,
        camera_in_tool: np.ndarray,
        stand_offs_m: tuple[float, ...] = (0.30, 0.40, 0.50),
        lateral_offsets_m: tuple[float, ...] = (-0.12, 0.0, 0.12),
        vertical_offsets_m: tuple[float, ...] = (-0.12, 0.0, 0.12),
        min_goal_joint_delta_rad: float = 0.08,
        min_camera_goal_delta_m: float = 0.08,
        frontier_reselect_cooldown_steps: int = 8,
        min_frontier_visibility_score: float = 0.05,
        min_roi_coverage_ratio: float = 0.08,
        min_view_alignment: float = 0.72,
        min_actual_view_alignment: float = 0.88,
        target_context_radius_m: float = 0.35,
        yaw_offsets_deg: tuple[float, ...] = (-12.0, 0.0, 12.0),
        pitch_offsets_deg: tuple[float, ...] = (-10.0, 0.0, 10.0),
        roll_offsets_deg: tuple[float, ...] = (0.0,),
        fast_roi_nbv: bool = True,
        roi_nbv_max_pose_candidates: int = 18,
        enable_self_occlusion_filter: bool = True,
        min_self_occlusion_free_fraction: float = 0.80,
        self_occlusion_padding_m: float = 0.03,
        self_occlusion_tool_radius_m: float = 0.08,
        self_occlusion_mount_radius_m: float = 0.07,
    ):
        self.kinematics = kinematics
        self.camera_in_tool = camera_in_tool
        self.stand_offs_m = stand_offs_m
        self.lateral_offsets_m = lateral_offsets_m
        self.vertical_offsets_m = vertical_offsets_m
        self.min_goal_joint_delta_rad = float(max(0.0, min_goal_joint_delta_rad))
        self.min_camera_goal_delta_m = float(max(0.0, min_camera_goal_delta_m))
        self.frontier_reselect_cooldown_steps = int(max(0, frontier_reselect_cooldown_steps))
        self.min_frontier_visibility_score = float(max(0.0, min_frontier_visibility_score))
        self.min_roi_coverage_ratio = float(np.clip(min_roi_coverage_ratio, 0.0, 1.0))
        self.min_view_alignment = float(np.clip(min_view_alignment, -1.0, 1.0))
        self.min_actual_view_alignment = float(np.clip(min_actual_view_alignment, -1.0, 1.0))
        self.target_context_radius_m = float(max(0.05, target_context_radius_m))
        self.yaw_offsets_rad = tuple(math.radians(float(v)) for v in yaw_offsets_deg)
        self.pitch_offsets_rad = tuple(math.radians(float(v)) for v in pitch_offsets_deg)
        self.roll_offsets_rad = tuple(math.radians(float(v)) for v in roll_offsets_deg)
        self.fast_roi_nbv = bool(fast_roi_nbv)
        self.roi_nbv_max_pose_candidates = max(1, int(roi_nbv_max_pose_candidates))
        self.enable_self_occlusion_filter = bool(enable_self_occlusion_filter)
        self.min_self_occlusion_free_fraction = float(np.clip(min_self_occlusion_free_fraction, 0.0, 1.0))
        self.self_occlusion_padding_m = float(max(0.0, self_occlusion_padding_m))
        self.self_occlusion_tool_radius_m = float(max(0.0, self_occlusion_tool_radius_m))
        self.self_occlusion_mount_radius_m = float(max(0.0, self_occlusion_mount_radius_m))
        self.last_select_debug: dict[str, float | int] = {}
        self.frontier_pose_cache: dict[int, dict[str, object]] = {}

    def _tool_camera_extra_spheres(self, q_goal: np.ndarray) -> np.ndarray:
        tool_pose = self.kinematics.fk(q_goal)
        cam_pose = self.kinematics.tool_to_camera_pose(tool_pose, self.camera_in_tool)
        tool_xyz = np.asarray(tool_pose[:3, 3], dtype=np.float64)
        cam_xyz = np.asarray(cam_pose[:3, 3], dtype=np.float64)
        mount_xyz = 0.5 * (tool_xyz + cam_xyz)
        spheres = []
        if self.self_occlusion_tool_radius_m > 0.0:
            spheres.append(np.r_[tool_xyz, self.self_occlusion_tool_radius_m])
        if self.self_occlusion_mount_radius_m > 0.0 and np.linalg.norm(cam_xyz - tool_xyz) > 1.0e-4:
            spheres.append(np.r_[mount_xyz, self.self_occlusion_mount_radius_m])
        return np.asarray(spheres, dtype=np.float64).reshape(-1, 4)

    def _self_occlusion_free_fraction(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_goal: np.ndarray,
        cam_pos: np.ndarray,
        target_points: np.ndarray,
    ) -> float:
        if not self.enable_self_occlusion_filter:
            return 1.0
        occluded = checker.robot_self_occlusion_fraction(
            q_goal,
            cam_pos,
            target_points,
            padding_m=self.self_occlusion_padding_m,
            extra_spheres=self._tool_camera_extra_spheres(q_goal),
        )
        return float(np.clip(1.0 - occluded, 0.0, 1.0))

    def _roi_unknown_centers(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        max_points: int = 300,
    ) -> np.ndarray:
        voxel = float(checker.voxel_size)
        lo_key = np.floor(np.asarray(roi_min, dtype=np.float64) / voxel).astype(int)
        hi_key = np.floor(np.asarray(roi_max, dtype=np.float64) / voxel).astype(int)
        keys = []
        for ix in range(int(lo_key[0]), int(hi_key[0]) + 1):
            for iy in range(int(lo_key[1]), int(hi_key[1]) + 1):
                for iz in range(int(lo_key[2]), int(hi_key[2]) + 1):
                    key = (ix, iy, iz)
                    if key in checker.free_keys or key in checker.occupied_keys:
                        continue
                    keys.append(key)
        if not keys:
            return np.zeros((0, 3), dtype=np.float64)
        stride = max(1, int(math.ceil(len(keys) / max(1, int(max_points)))))
        centers = [checker.key_to_center(key) for key in keys[::stride]]
        return np.asarray(centers, dtype=np.float64)

    def _roi_candidate_camera_poses(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        current_camera_xyz: np.ndarray | None,
        target_xyz: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, str]]:
        roi_min = np.asarray(roi_min, dtype=np.float64)
        roi_max = np.asarray(roi_max, dtype=np.float64)
        center = 0.5 * (roi_min + roi_max)
        size = np.maximum(roi_max - roi_min, 0.05)
        target = center if target_xyz is None else np.asarray(target_xyz, dtype=np.float64).reshape(3)
        if current_camera_xyz is None:
            approach = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        else:
            approach = target - np.asarray(current_camera_xyz, dtype=np.float64)
            approach[2] *= 0.25
            if np.linalg.norm(approach) < 1e-6:
                approach = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            approach /= np.linalg.norm(approach)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = np.cross(up, approach)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(approach, lateral)
        vertical /= max(np.linalg.norm(vertical), 1e-6)

        if self.fast_roi_nbv:
            standoffs = (0.45, 0.65)
            lateral_offsets = (0.0, -0.35 * size[1], 0.35 * size[1])
            vertical_offsets = (0.0, 0.22 * size[2], -0.22 * size[2])
        else:
            standoffs = (0.35, 0.50, 0.70)
            lateral_offsets = (-0.45 * size[1], 0.0, 0.45 * size[1])
            vertical_offsets = (-0.35 * size[2], 0.0, 0.35 * size[2])
        poses: list[tuple[np.ndarray, str]] = []
        for dist in standoffs:
            base_pos = target - dist * approach
            for lateral_offset in lateral_offsets:
                for vertical_offset in vertical_offsets:
                    cam_pos = base_pos + lateral_offset * lateral + vertical_offset * vertical
                    if self.fast_roi_nbv:
                        poses.append((_transform(look_at_rotation(cam_pos, target), cam_pos), "roi_fast"))
                    else:
                        poses.extend(self._oriented_camera_poses(cam_pos, target, "roi"))
                    if self.fast_roi_nbv and len(poses) >= self.roi_nbv_max_pose_candidates:
                        return poses
        return poses

    def _roi_view_metrics(
        self,
        cam_pose: np.ndarray,
        cam_pos: np.ndarray,
        roi_center: np.ndarray,
        unknown_centers: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None,
    ) -> dict[str, float]:
        if camera_info is None or len(unknown_centers) == 0:
            return {"unknown_gain": 0.0, "visibility_score": -1e9, "target_visibility": 0.0}
        target_visibility = self._ray_clearance_score(checker, cam_pos, roi_center, collision_radius_m=0.04)
        if target_visibility <= 0.0 or self._project_to_image(cam_pose, roi_center, camera_info) is None:
            return {"unknown_gain": 0.0, "visibility_score": -1e9, "target_visibility": 0.0}
        visible = 0
        weighted_gain = 0.0
        for point in unknown_centers:
            proj = self._project_to_image(cam_pose, point, camera_info)
            if proj is None:
                continue
            _px, _py, depth = proj
            vis = self._ray_clearance_score(checker, cam_pos, point, collision_radius_m=0.035)
            if vis <= 0.0:
                continue
            depth_score = 1.0 / (1.0 + 0.5 * max(0.0, depth - 0.35))
            weighted_gain += vis * depth_score
            visible += 1
        unknown_gain = weighted_gain / float(max(1, len(unknown_centers)))
        visibility_score = 0.75 * unknown_gain + 0.25 * target_visibility
        return {
            "unknown_gain": float(unknown_gain),
            "visibility_score": float(visibility_score),
            "target_visibility": float(target_visibility),
            "visible_unknown": float(visible),
        }

    def ranked_roi_candidates(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        current_q: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None = None,
        current_camera_xyz: np.ndarray | None = None,
        max_candidates: int = 8,
    ) -> list[ViewGoal]:
        current_q = np.asarray(current_q, dtype=np.float64)
        roi_min = np.asarray(roi_min, dtype=np.float64)
        roi_max = np.asarray(roi_max, dtype=np.float64)
        roi_center = 0.5 * (roi_min + roi_max)
        unknown_centers = self._roi_unknown_centers(roi_min, roi_max, checker)
        roi_target = np.mean(unknown_centers, axis=0) if len(unknown_centers) else roi_center
        stats = {
            "candidates_total": 0,
            "unknown_points": int(len(unknown_centers)),
            "rejected_visibility": 0,
            "rejected_ik": 0,
            "rejected_clearance": 0,
            "rejected_same_pose": 0,
            "rejected_same_view": 0,
            "rejected_orientation": 0,
            "rejected_self_occlusion": 0,
            "accepted": 0,
        }
        ranked: list[ViewGoal] = []
        for cam_pose, pose_kind in self._roi_candidate_camera_poses(
            roi_min,
            roi_max,
            current_camera_xyz,
            target_xyz=roi_target,
        ):
            stats["candidates_total"] += 1
            cam_pos = cam_pose[:3, 3]
            if (
                current_camera_xyz is not None
                and float(np.linalg.norm(cam_pos - np.asarray(current_camera_xyz, dtype=np.float64))) < self.min_camera_goal_delta_m
            ):
                stats["rejected_same_view"] += 1
                continue
            view_metrics = self._roi_view_metrics(
                cam_pose, cam_pos, roi_target, unknown_centers, checker, camera_info
            )
            if view_metrics["visibility_score"] < self.min_frontier_visibility_score:
                stats["rejected_visibility"] += 1
                continue
            tool_pose = self.kinematics.camera_to_tool_pose(cam_pose, self.camera_in_tool)
            q_goal = self.kinematics.solve_ik(tool_pose, current_q)
            if q_goal is None:
                stats["rejected_ik"] += 1
                continue
            if float(np.max(np.abs(q_goal - current_q))) < self.min_goal_joint_delta_rad:
                stats["rejected_same_pose"] += 1
                continue
            actual_tool_pose = self.kinematics.fk(q_goal)
            actual_cam_pose = self.kinematics.tool_to_camera_pose(actual_tool_pose, self.camera_in_tool)
            actual_cam_pos = actual_cam_pose[:3, 3]
            actual_ray = roi_target - actual_cam_pos
            actual_ray_norm = float(np.linalg.norm(actual_ray))
            if actual_ray_norm < 1e-6:
                stats["rejected_orientation"] += 1
                continue
            actual_alignment = float(np.dot(actual_cam_pose[:3, 2], actual_ray / actual_ray_norm))
            if actual_alignment < self.min_actual_view_alignment:
                stats["rejected_orientation"] += 1
                continue
            actual_metrics = self._roi_view_metrics(
                actual_cam_pose, actual_cam_pos, roi_target, unknown_centers, checker, camera_info
            )
            if actual_metrics["visibility_score"] < self.min_frontier_visibility_score:
                stats["rejected_visibility"] += 1
                continue
            target_bundle = unknown_centers
            if len(target_bundle) == 0:
                target_bundle = roi_target.reshape(1, 3)
            else:
                target_bundle = np.vstack((roi_target.reshape(1, 3), target_bundle))
            self_free_fraction = self._self_occlusion_free_fraction(
                checker, q_goal, actual_cam_pos, target_bundle
            )
            if self_free_fraction < self.min_self_occlusion_free_fraction:
                stats["rejected_self_occlusion"] += 1
                continue
            clearance = checker.clearance(q_goal)
            if clearance <= 0.0:
                stats["rejected_clearance"] += 1
                continue
            move_cost = float(np.linalg.norm(q_goal - current_q))
            score = (
                actual_metrics["visibility_score"]
                - 0.30 * move_cost
                + 0.45 * min(float(clearance), 0.25)
            )
            ranked.append(
                ViewGoal(
                    frontier_id=-1,
                    centroid=roi_target.copy(),
                    camera_pose=actual_cam_pose.copy(),
                    tool_pose=actual_tool_pose.copy(),
                    q_goal=q_goal.copy(),
                    score=float(score),
                    pose_kind=pose_kind,
                    visibility_score=float(actual_metrics["visibility_score"]),
                    local_coverage=float(actual_metrics["unknown_gain"]),
                    gain_score=float(actual_metrics.get("visible_unknown", 0.0)),
                    move_cost=move_cost,
                    clearance=float(clearance),
                )
            )
            stats["accepted"] += 1
        self.last_select_debug = stats
        ranked.sort(key=lambda goal: goal.score, reverse=True)
        return ranked[: max(1, int(max_candidates))]

    def _oriented_camera_poses(self, cam_pos: np.ndarray, centroid: np.ndarray, pose_kind: str) -> list[tuple[np.ndarray, str]]:
        base_rot = look_at_rotation(cam_pos, centroid)
        poses: list[tuple[np.ndarray, str]] = []
        yaw_choices = self.yaw_offsets_rad if pose_kind in ("local", "roi") else (0.0,)
        pitch_choices = self.pitch_offsets_rad if pose_kind in ("local", "roi") else (0.0,)
        roll_choices = self.roll_offsets_rad
        for yaw in yaw_choices:
            for pitch in pitch_choices:
                for roll in roll_choices:
                    # Apply a small local orientation cone around the canonical look-at view.
                    rot = base_rot @ _rot_z(roll) @ _rot_y(yaw) @ _rot_x(pitch)
                    poses.append((_transform(rot, cam_pos), pose_kind))
        return poses

    def _frontier_frame(self, frontier: FrontierRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroid = frontier.centroid.astype(np.float64)
        normal = frontier.normal.astype(np.float64)
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tangent_y = np.cross(up, normal)
        if np.linalg.norm(tangent_y) < 1e-6:
            tangent_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_y /= np.linalg.norm(tangent_y)
        tangent_z = np.cross(normal, tangent_y)
        tangent_z /= max(np.linalg.norm(tangent_z), 1e-6)
        return centroid, normal, np.column_stack((tangent_y, tangent_z))

    def _candidate_camera_poses(self, frontier: FrontierRecord) -> list[tuple[np.ndarray, str]]:
        centroid, normal, tangents = self._frontier_frame(frontier)
        poses: list[tuple[np.ndarray, str]] = []
        for dist in self.stand_offs_m:
            base_pos = centroid - dist * normal
            for lateral in self.lateral_offsets_m:
                for vertical in self.vertical_offsets_m:
                    cam_pos = (
                        base_pos
                        + lateral * tangents[:, 0]
                        + vertical * tangents[:, 1]
                    )
                    poses.extend(self._oriented_camera_poses(cam_pos, centroid, "local"))
        recovery_standoffs = (0.60, 0.75)
        recovery_offsets = (-0.22, 0.0, 0.22)
        for dist in recovery_standoffs:
            base_pos = centroid - dist * normal
            for lateral in recovery_offsets:
                for vertical in recovery_offsets:
                    cam_pos = (
                        base_pos
                        + lateral * tangents[:, 0]
                        + vertical * tangents[:, 1]
                    )
                    poses.extend(self._oriented_camera_poses(cam_pos, centroid, "recovery"))
        return poses

    def _cached_candidate_camera_poses(self, frontier: FrontierRecord) -> list[tuple[np.ndarray, str]]:
        key = int(frontier.frontier_id)
        centroid = frontier.centroid.astype(np.float64)
        normal = frontier.normal.astype(np.float64)
        cached = self.frontier_pose_cache.get(key)
        if cached is not None:
            old_centroid = np.asarray(cached["centroid"], dtype=np.float64)
            old_normal = np.asarray(cached["normal"], dtype=np.float64)
            if (
                np.linalg.norm(old_centroid - centroid) < 0.04
                and np.linalg.norm(old_normal - normal) < 0.20
            ):
                return list(cached["poses"])
        poses = self._candidate_camera_poses(frontier)
        self.frontier_pose_cache[key] = {
            "centroid": centroid.copy(),
            "normal": normal.copy(),
            "poses": list(poses),
        }
        return poses

    def _ray_clearance_score(
        self,
        checker: UR5PointCloudCollisionChecker,
        origin: np.ndarray,
        target: np.ndarray,
        collision_radius_m: float = 0.06,
    ) -> float:
        if checker.kdtree is None or len(checker.occupied_points) == 0:
            return 1.0
        delta = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
        ray_len = float(np.linalg.norm(delta))
        if ray_len < 1e-6:
            return 0.0
        direction = delta / ray_len
        # Skip endpoints to avoid penalizing the frontier itself.
        alphas = np.linspace(0.15, 0.85, 6, dtype=np.float64)
        samples = origin[None, :] + alphas[:, None] * ray_len * direction[None, :]
        dists, _ = checker.kdtree.query(samples, k=1)
        min_dist = float(np.min(dists))
        if min_dist < collision_radius_m:
            return 0.0
        return min(1.0, (min_dist - collision_radius_m) / 0.12)

    def _candidate_gain(
        self,
        cam_pose: np.ndarray,
        cam_pos: np.ndarray,
        frontier: FrontierRecord,
        active_frontiers: list[FrontierRecord],
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None,
    ) -> float:
        focus = frontier.centroid.astype(np.float64)
        view_dir = focus - cam_pos
        view_norm = float(np.linalg.norm(view_dir))
        if view_norm < 1e-6:
            return -1e9
        view_dir /= view_norm
        gain = 0.0
        unknown_gain = 0.0
        counted_unknown: set[tuple[int, int, int]] = set()
        nbrs = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for rec in active_frontiers:
            target = rec.centroid.astype(np.float64)
            ray = target - cam_pos
            dist = float(np.linalg.norm(ray))
            if dist < 1e-6 or dist > 1.25:
                continue
            ray_dir = ray / dist
            facing = float(np.dot(ray_dir, rec.normal.astype(np.float64)))
            if facing < 0.15:
                continue
            optical_align = float(np.dot(view_dir, ray_dir))
            if optical_align < 0.85:
                continue
            proj = self._project_to_image(cam_pose, target, camera_info)
            if proj is None:
                continue
            _, _, depth = proj
            visibility = self._ray_clearance_score(checker, cam_pos, target)
            if visibility <= 0.0:
                continue
            depth_score = 1.0 / (1.0 + 0.6 * max(0.0, depth - 0.35))
            gain += visibility * optical_align * depth_score * (1.0 + 0.03 * rec.voxel_count)
            # Adapt the global_scene idea: reward candidate views that can expose
            # nearby unknown voxels at the frontier boundary, not just the centroid.
            voxel_stride = max(1, len(rec.voxels) // 24)
            for vx, vy, vz in rec.voxels[::voxel_stride]:
                for dx, dy, dz in nbrs:
                    unk = (vx + dx, vy + dy, vz + dz)
                    if unk in counted_unknown:
                        continue
                    if unk in checker.free_keys or unk in checker.occupied_keys:
                        continue
                    unk_center = checker.key_to_center(unk)
                    unk_ray = unk_center - cam_pos
                    unk_dist = float(np.linalg.norm(unk_ray))
                    if unk_dist < 1e-6 or unk_dist > 1.35:
                        continue
                    unk_dir = unk_ray / unk_dist
                    if float(np.dot(view_dir, unk_dir)) < 0.80:
                        continue
                    if self._project_to_image(cam_pose, unk_center, camera_info) is None:
                        continue
                    if float(np.dot(unk_dir, rec.normal.astype(np.float64))) < 0.05:
                        continue
                    unk_vis = self._ray_clearance_score(checker, cam_pos, unk_center, collision_radius_m=0.04)
                    if unk_vis <= 0.0:
                        continue
                    counted_unknown.add(unk)
                    unknown_gain += unk_vis
        return gain + 0.12 * unknown_gain

    def _target_neighborhood(
        self,
        frontier: FrontierRecord,
        active_frontiers: list[FrontierRecord],
    ) -> list[FrontierRecord]:
        center = frontier.centroid.astype(np.float64)
        neighborhood = [frontier]
        for rec in active_frontiers:
            if rec.frontier_id == frontier.frontier_id:
                continue
            dist = float(np.linalg.norm(rec.centroid.astype(np.float64) - center))
            if dist <= self.target_context_radius_m:
                neighborhood.append(rec)
        return neighborhood

    def _candidate_view_metrics(
        self,
        cam_pose: np.ndarray,
        cam_pos: np.ndarray,
        frontier: FrontierRecord,
        active_frontiers: list[FrontierRecord],
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None,
    ) -> dict[str, float]:
        focus = frontier.centroid.astype(np.float64)
        focus_ray = focus - cam_pos
        focus_dist = float(np.linalg.norm(focus_ray))
        if focus_dist < 1e-6:
            return {
                "visibility_score": -1e9,
                "local_coverage": 0.0,
                "view_alignment": -1.0,
                "target_visibility": 0.0,
                "global_context": 0.0,
            }
        view_dir = focus_ray / focus_dist
        target_visibility = self._ray_clearance_score(checker, cam_pos, focus)
        if target_visibility <= 0.0 or self._project_to_image(cam_pose, focus, camera_info) is None:
            return {
                "visibility_score": -1e9,
                "local_coverage": 0.0,
                "view_alignment": -1.0,
                "target_visibility": 0.0,
                "global_context": 0.0,
            }

        neighborhood = self._target_neighborhood(frontier, active_frontiers)
        visible_local = 0
        alignment_sum = 0.0
        total_local = max(1, len(neighborhood))
        for rec in neighborhood:
            target = rec.centroid.astype(np.float64)
            ray = target - cam_pos
            dist = float(np.linalg.norm(ray))
            if dist < 1e-6 or dist > 1.35:
                continue
            ray_dir = ray / dist
            optical_align = float(np.dot(view_dir, ray_dir))
            if optical_align < self.min_view_alignment:
                continue
            proj = self._project_to_image(cam_pose, target, camera_info)
            if proj is None:
                continue
            visibility = self._ray_clearance_score(checker, cam_pos, target)
            if visibility <= 0.0:
                continue
            visible_local += 1
            alignment_sum += optical_align
        local_coverage = float(visible_local) / float(total_local)
        mean_alignment = 0.0 if visible_local == 0 else alignment_sum / float(visible_local)

        visible_global = 0
        total_global = max(1, len(active_frontiers))
        for rec in active_frontiers:
            target = rec.centroid.astype(np.float64)
            ray = target - cam_pos
            dist = float(np.linalg.norm(ray))
            if dist < 1e-6 or dist > 1.35:
                continue
            ray_dir = ray / dist
            if float(np.dot(view_dir, ray_dir)) < self.min_view_alignment:
                continue
            if self._project_to_image(cam_pose, target, camera_info) is None:
                continue
            if self._ray_clearance_score(checker, cam_pos, target) <= 0.0:
                continue
            visible_global += 1
        global_context = float(visible_global) / float(total_global)

        gain_score = self._candidate_gain(cam_pose, cam_pos, frontier, active_frontiers, checker, camera_info)
        visibility_score = (
            0.65 * target_visibility
            + 0.35 * local_coverage
            + 0.10 * global_context
            + 0.10 * mean_alignment
            + 0.15 * max(0.0, gain_score)
        )
        return {
            "visibility_score": float(visibility_score),
            "local_coverage": float(local_coverage),
            "view_alignment": float(mean_alignment),
            "gain_score": float(gain_score),
            "target_visibility": float(target_visibility),
            "global_context": float(global_context),
            "visible_frontiers": float(visible_local),
        }

    def _project_to_image(
        self,
        cam_pose: np.ndarray,
        point_world: np.ndarray,
        camera_info: CameraInfo | None,
    ) -> tuple[float, float, float] | None:
        if camera_info is None:
            return None
        cam_from_world = np.linalg.inv(cam_pose)
        p_world = np.ones((4,), dtype=np.float64)
        p_world[:3] = np.asarray(point_world, dtype=np.float64)
        p_cam = cam_from_world @ p_world
        depth = float(p_cam[2])
        if depth <= 0.20 or depth >= 2.00:
            return None
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        px = fx * (p_cam[0] / depth) + cx
        py = fy * (p_cam[1] / depth) + cy
        if not (0.0 <= px < float(camera_info.width) and 0.0 <= py < float(camera_info.height)):
            return None
        return float(px), float(py), depth

    def ranked_candidates(
        self,
        frontiers: list[FrontierRecord],
        current_q: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None = None,
        current_camera_xyz: np.ndarray | None = None,
        step_idx: int | None = None,
        max_candidates: int = 12,
        progress_cb=None,
    ) -> list[ViewGoal]:
        current_q = np.asarray(current_q, dtype=np.float64)
        current_camera_xyz = None if current_camera_xyz is None else np.asarray(current_camera_xyz, dtype=np.float64)
        ranked: list[ViewGoal] = []
        ranked_frontiers = sorted(frontiers, key=lambda rec: rec.voxel_count, reverse=True)[:18]
        stats = {
            "frontiers_considered": len(ranked_frontiers),
            "candidates_total": 0,
            "rejected_gain": 0,
            "rejected_visibility": 0,
            "rejected_ik": 0,
            "rejected_clearance": 0,
            "rejected_same_pose": 0,
            "rejected_same_view": 0,
            "rejected_cooldown": 0,
            "rejected_orientation": 0,
            "rejected_self_occlusion": 0,
            "accepted": 0,
        }
        last_progress_count = 0
        for frontier in ranked_frontiers:
            if (
                step_idx is not None
                and self.frontier_reselect_cooldown_steps > 0
                and frontier.last_selected_step >= 0
                and (int(step_idx) - int(frontier.last_selected_step)) < self.frontier_reselect_cooldown_steps
            ):
                stats["rejected_cooldown"] += 1
                continue
            for cam_pose, pose_kind in self._cached_candidate_camera_poses(frontier):
                stats["candidates_total"] += 1
                cam_pos = cam_pose[:3, 3]
                if (
                    current_camera_xyz is not None
                    and float(np.linalg.norm(cam_pos - current_camera_xyz)) < self.min_camera_goal_delta_m
                ):
                    stats["rejected_same_view"] += 1
                    continue
                if self._project_to_image(cam_pose, frontier.centroid.astype(np.float64), camera_info) is None:
                    stats["rejected_gain"] += 1
                    continue
                view_metrics = self._candidate_view_metrics(
                    cam_pose,
                    cam_pos,
                    frontier,
                    frontiers,
                    checker,
                    camera_info,
                )
                if (
                    view_metrics["visibility_score"] < self.min_frontier_visibility_score
                    or view_metrics["local_coverage"] < self.min_roi_coverage_ratio
                ):
                    stats["rejected_visibility"] += 1
                    continue
                tool_pose = self.kinematics.camera_to_tool_pose(cam_pose, self.camera_in_tool)
                q_goal = self.kinematics.solve_ik(tool_pose, current_q)
                if q_goal is None:
                    stats["rejected_ik"] += 1
                    continue
                if float(np.max(np.abs(q_goal - current_q))) < self.min_goal_joint_delta_rad:
                    stats["rejected_same_pose"] += 1
                    continue
                actual_tool_pose = self.kinematics.fk(q_goal)
                actual_cam_pose = self.kinematics.tool_to_camera_pose(actual_tool_pose, self.camera_in_tool)
                actual_cam_pos = actual_cam_pose[:3, 3]
                target = frontier.centroid.astype(np.float64)
                actual_ray = target - actual_cam_pos
                actual_ray_norm = float(np.linalg.norm(actual_ray))
                if actual_ray_norm < 1e-6:
                    stats["rejected_orientation"] += 1
                    continue
                actual_alignment = float(np.dot(actual_cam_pose[:3, 2], actual_ray / actual_ray_norm))
                if actual_alignment < self.min_actual_view_alignment:
                    stats["rejected_orientation"] += 1
                    continue
                actual_view_metrics = self._candidate_view_metrics(
                    actual_cam_pose,
                    actual_cam_pos,
                    frontier,
                    frontiers,
                    checker,
                    camera_info,
                )
                if (
                    actual_view_metrics["visibility_score"] < self.min_frontier_visibility_score
                    or actual_view_metrics["local_coverage"] < self.min_roi_coverage_ratio
                ):
                    stats["rejected_visibility"] += 1
                    continue
                self_free_fraction = self._self_occlusion_free_fraction(
                    checker, q_goal, actual_cam_pos, frontier.centroid.reshape(1, 3)
                )
                if self_free_fraction < self.min_self_occlusion_free_fraction:
                    stats["rejected_self_occlusion"] += 1
                    continue
                clearance = checker.clearance(q_goal)
                if clearance <= 0.0:
                    stats["rejected_clearance"] += 1
                    continue
                move_cost = float(np.linalg.norm(q_goal - current_q))
                fallback_score = (
                    0.03 * frontier.voxel_count
                    - 0.30 * move_cost
                    + 0.8 * min(clearance, 0.25)
                    + 0.15 * actual_view_metrics["local_coverage"]
                )
                if pose_kind == "recovery":
                    fallback_score += 0.08
                if actual_view_metrics["gain_score"] <= 0.0:
                    score = fallback_score
                else:
                    score = (
                        actual_view_metrics["visibility_score"]
                        - 0.35 * move_cost
                        + 0.6 * min(clearance, 0.25)
                    )
                    if pose_kind == "recovery":
                        score += 0.10
                ranked.append(
                    ViewGoal(
                        frontier_id=frontier.frontier_id,
                        centroid=frontier.centroid.copy(),
                        camera_pose=actual_cam_pose.copy(),
                        tool_pose=actual_tool_pose.copy(),
                        q_goal=q_goal.copy(),
                        score=score,
                        pose_kind=pose_kind,
                        visibility_score=float(actual_view_metrics["visibility_score"]),
                        local_coverage=float(actual_view_metrics["local_coverage"]),
                        gain_score=float(actual_view_metrics["gain_score"]),
                        move_cost=move_cost,
                        clearance=float(clearance),
                    )
                )
                stats["accepted"] += 1
                if progress_cb is not None and (stats["candidates_total"] - last_progress_count) >= 20:
                    last_progress_count = int(stats["candidates_total"])
                    progress_cb(dict(stats))
        self.last_select_debug = stats
        ranked.sort(key=lambda goal: goal.score, reverse=True)
        return ranked[: max(1, int(max_candidates))]

    def select(
        self,
        frontiers: list[FrontierRecord],
        current_q: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None = None,
        current_camera_xyz: np.ndarray | None = None,
        step_idx: int | None = None,
    ) -> ViewGoal | None:
        ranked = self.ranked_candidates(
            frontiers,
            current_q,
            checker,
            camera_info=camera_info,
            current_camera_xyz=current_camera_xyz,
            step_idx=step_idx,
            max_candidates=1,
        )
        return ranked[0] if ranked else None
