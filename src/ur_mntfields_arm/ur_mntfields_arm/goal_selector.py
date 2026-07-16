from __future__ import annotations

import math
import numpy as np
from sensor_msgs.msg import CameraInfo

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.frontier_bank import FrontierRecord
from ur_mntfields_arm.roi_targeting import (
    distribute_candidate_budget,
    spatially_stratified_targets,
)
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
        min_actual_view_alignment: float = 0.85,
        target_context_radius_m: float = 0.35,
        yaw_offsets_deg: tuple[float, ...] = (-12.0, 0.0, 12.0),
        pitch_offsets_deg: tuple[float, ...] = (-10.0, 0.0, 10.0),
        roll_offsets_deg: tuple[float, ...] = (0.0,),
        fast_roi_nbv: bool = True,
        roi_nbv_max_pose_candidates: int = 18,
        enable_self_occlusion_filter: bool = True,
        min_self_occlusion_free_fraction: float = 0.98,
        self_occlusion_padding_m: float = 0.03,
        self_occlusion_ignore_near_origin_m: float = 0.06,
        self_occlusion_tool_radius_m: float = 0.08,
        self_occlusion_mount_radius_m: float = 0.07,
        frontier_pose_candidates_per_frontier: int = 15,
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
        # ROI targets are optional, but a zero-quality target should never
        # displace a valid frontier candidate.
        self.min_roi_visibility_score = max(0.01, self.min_frontier_visibility_score)
        self.min_roi_unknown_gain = max(0.01, self.min_roi_coverage_ratio)
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
        self.self_occlusion_ignore_near_origin_m = float(max(0.0, self_occlusion_ignore_near_origin_m))
        self.self_occlusion_tool_radius_m = float(max(0.0, self_occlusion_tool_radius_m))
        self.self_occlusion_mount_radius_m = float(max(0.0, self_occlusion_mount_radius_m))
        self.frontier_pose_candidates_per_frontier = max(3, int(frontier_pose_candidates_per_frontier))
        self.last_select_debug: dict[str, float | int] = {}
        self.frontier_pose_cache: dict[int, dict[str, object]] = {}

    def _tool_camera_extra_spheres(self, q_goal: np.ndarray) -> np.ndarray:
        del q_goal
        # The checker now contains wrist, bracket, and camera geometry. The
        # former 80 mm tool/70 mm mount proxies were useful while the camera
        # was absent, but with a top-mounted camera they intersect its forward
        # rays and falsely reject otherwise visible NBV poses.
        return np.zeros((0, 4), dtype=np.float64)

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
            ignore_near_origin_m=self.self_occlusion_ignore_near_origin_m,
            extra_spheres=self._tool_camera_extra_spheres(q_goal),
        )
        return float(np.clip(1.0 - occluded, 0.0, 1.0))

    def _actual_target_view_is_valid(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_goal: np.ndarray,
        actual_cam_pose: np.ndarray,
        target: np.ndarray,
        camera_info: CameraInfo | None,
    ) -> tuple[bool, str]:
        """Apply the final observation gate to the pose that IK actually reaches."""
        actual_cam_pos = np.asarray(actual_cam_pose[:3, 3], dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).reshape(3)
        ray = target - actual_cam_pos
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1.0e-6:
            return False, "orientation"
        alignment = float(np.dot(actual_cam_pose[:3, 2], ray / ray_norm))
        if alignment < self.min_actual_view_alignment:
            return False, "orientation"
        if self._project_to_image(actual_cam_pose, target, camera_info) is None:
            return False, "visibility"
        if self._ray_clearance_score(checker, actual_cam_pos, target) <= 0.0:
            return False, "visibility"
        target_free_fraction = self._self_occlusion_free_fraction(
            checker,
            q_goal,
            actual_cam_pos,
            target.reshape(1, 3),
        )
        if target_free_fraction < self.min_self_occlusion_free_fraction:
            return False, "self_occlusion"
        return True, ""

    def _roi_unknown_centers(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
        max_points: int | None = 300,
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
        centers = [checker.key_to_center(key) for key in keys]
        centers_arr = np.asarray(centers, dtype=np.float64)

        # The cabinet boxes are known collision geometry, not learnable empty
        # space. Keeping their interior as "unknown" biases ROI gain toward
        # wall/shelf volume and pulls the target toward the box center.
        boxes = np.asarray(getattr(checker, "box_obstacles", np.zeros((0, 6))), dtype=np.float64).reshape(-1, 6)
        if len(boxes):
            delta = np.abs(centers_arr[:, None, :] - boxes[None, :, :3])
            inside_known_geometry = np.any(np.all(delta <= 0.5 * boxes[None, :, 3:], axis=2), axis=1)
            centers_arr = centers_arr[~inside_known_geometry]
        if max_points is not None and int(max_points) > 0 and len(centers_arr) > int(max_points):
            stride = max(1, int(math.ceil(len(centers_arr) / int(max_points))))
            centers_arr = centers_arr[::stride]
        return centers_arr

    def _roi_stratified_targets(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        checker: UR5PointCloudCollisionChecker,
    ) -> np.ndarray:
        all_unknown = self._roi_unknown_centers(roi_min, roi_max, checker, max_points=None)
        target_limit = min(8, max(1, self.roi_nbv_max_pose_candidates // 2))
        return spatially_stratified_targets(
            all_unknown,
            roi_min,
            roi_max,
            max_targets=target_limit,
        )

    def _roi_candidate_camera_poses(
        self,
        roi_min: np.ndarray,
        roi_max: np.ndarray,
        current_camera_xyz: np.ndarray | None,
        target_xyz: np.ndarray | None = None,
        max_pose_candidates: int | None = None,
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
        pose_cap = self.roi_nbv_max_pose_candidates if max_pose_candidates is None else max(1, int(max_pose_candidates))
        lateral_nonzero = [value for value in lateral_offsets if abs(float(value)) > 1.0e-8]
        vertical_nonzero = [value for value in vertical_offsets if abs(float(value)) > 1.0e-8]
        offset_pairs = [(0.0, 0.0)]
        offset_pairs.extend((float(value), 0.0) for value in lateral_nonzero)
        offset_pairs.extend((0.0, float(value)) for value in vertical_nonzero)
        offset_pairs.extend(
            (float(lateral_offset), float(vertical_offset))
            for lateral_offset in lateral_nonzero
            for vertical_offset in vertical_nonzero
        )
        poses: list[tuple[np.ndarray, str]] = []
        for dist in standoffs:
            base_pos = target - dist * approach
            for lateral_offset, vertical_offset in offset_pairs:
                cam_pos = base_pos + lateral_offset * lateral + vertical_offset * vertical
                if self.fast_roi_nbv:
                    poses.append((_transform(look_at_rotation(cam_pos, target), cam_pos), "roi_fast"))
                else:
                    remaining = pose_cap - len(poses)
                    poses.extend(self._oriented_camera_poses(cam_pos, target, "roi")[:remaining])
                if len(poses) >= pose_cap:
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
        # ROI targets carry the coverage diversity. A compact metric set is
        # sufficient to rank them and avoids thousands of KD-tree rays before
        # a candidate has even passed IK.
        unknown_limit = 96 if self.fast_roi_nbv else 300
        unknown_centers = self._roi_unknown_centers(
            roi_min,
            roi_max,
            checker,
            max_points=unknown_limit,
        )
        roi_targets = self._roi_stratified_targets(roi_min, roi_max, checker)
        stats = {
            "candidates_total": 0,
            "unknown_points": int(len(unknown_centers)),
            "roi_targets": int(len(roi_targets)),
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
        pose_budget = distribute_candidate_budget(self.roi_nbv_max_pose_candidates, len(roi_targets))
        for roi_target, target_pose_budget in zip(roi_targets, pose_budget):
            for cam_pose, pose_kind in self._roi_candidate_camera_poses(
                roi_min,
                roi_max,
                current_camera_xyz,
                target_xyz=roi_target,
                max_pose_candidates=target_pose_budget,
            ):
                stats["candidates_total"] += 1
                cam_pos = cam_pose[:3, 3]
                if (
                    current_camera_xyz is not None
                    and float(np.linalg.norm(cam_pos - np.asarray(current_camera_xyz, dtype=np.float64))) < self.min_camera_goal_delta_m
                ):
                    stats["rejected_same_view"] += 1
                    continue
                # Only test the focus ray before IK. Full unknown-space gain
                # is intentionally deferred until the actual IK camera pose
                # has passed its target-observation gates.
                if (
                    self._project_to_image(cam_pose, roi_target, camera_info) is None
                    or self._ray_clearance_score(checker, cam_pos, roi_target, collision_radius_m=0.04) <= 0.0
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
                target_ok, reject_reason = self._actual_target_view_is_valid(
                    checker,
                    q_goal,
                    actual_cam_pose,
                    roi_target,
                    camera_info,
                )
                if not target_ok:
                    if reject_reason == "orientation":
                        stats["rejected_orientation"] += 1
                    elif reject_reason == "self_occlusion":
                        stats["rejected_self_occlusion"] += 1
                    else:
                        stats["rejected_visibility"] += 1
                    continue
                actual_metrics = self._roi_view_metrics(
                    actual_cam_pose, actual_cam_pos, roi_target, unknown_centers, checker, camera_info
                )
                if (
                    actual_metrics["visibility_score"] < self.min_roi_visibility_score
                    or actual_metrics["unknown_gain"] < self.min_roi_unknown_gain
                ):
                    stats["rejected_visibility"] += 1
                    continue
                target_bundle = unknown_centers
                if len(target_bundle) == 0:
                    target_bundle = roi_target.reshape(1, 3)
                else:
                    target_bundle = np.vstack((roi_target.reshape(1, 3), target_bundle))
                bundle_free_fraction = self._self_occlusion_free_fraction(
                    checker, q_goal, actual_cam_pos, target_bundle
                )
                if bundle_free_fraction < self.min_self_occlusion_free_fraction:
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
        lateral_offsets = (0.0,) + tuple(value for value in self.lateral_offsets_m if abs(value) > 1.0e-8)
        vertical_offsets = (0.0,) + tuple(value for value in self.vertical_offsets_m if abs(value) > 1.0e-8)
        for dist in self.stand_offs_m:
            # Frontier normals point from occupied geometry into observed free
            # space. Camera candidates must stay on that free-space side.
            base_pos = centroid + dist * normal
            for lateral in lateral_offsets:
                for vertical in vertical_offsets:
                    cam_pos = (
                        base_pos
                        + lateral * tangents[:, 0]
                        + vertical * tangents[:, 1]
                    )
                    # Position offsets cover faces and edges. The 3x3
                    # orientation cone only multiplied the expensive
                    # visibility pass while making no new target visible.
                    poses.append((_transform(look_at_rotation(cam_pos, centroid), cam_pos), "local"))
        recovery_standoffs = (0.60, 0.75)
        recovery_offsets = (0.0, -0.22, 0.22)
        for dist in recovery_standoffs:
            base_pos = centroid + dist * normal
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

    def _frontier_pose_subset(self, frontier: FrontierRecord) -> list[tuple[np.ndarray, str]]:
        """Retain near and recovery views without evaluating an unbounded cone."""
        poses = self._cached_candidate_camera_poses(frontier)
        local = [pose for pose in poses if pose[1] == "local"]
        recovery = [pose for pose in poses if pose[1] == "recovery"]
        local_budget = min(len(local), max(1, int(math.ceil(0.60 * self.frontier_pose_candidates_per_frontier))))
        recovery_budget = min(len(recovery), self.frontier_pose_candidates_per_frontier - local_budget)
        return local[:local_budget] + recovery[:recovery_budget]

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
        del cam_pose, cam_pos, active_frontiers, checker, camera_info
        # Candidate generation already covers local face/edge offsets.  The
        # former voxel-neighbour ray sweep made every pose O(frontiers *
        # voxels), turning NBV into the dominant online-training cost.
        return 0.25 + 0.75 * min(1.0, float(frontier.voxel_count) / 24.0)

    def _target_neighborhood(
        self,
        frontier: FrontierRecord,
        active_frontiers: list[FrontierRecord],
    ) -> list[FrontierRecord]:
        center = frontier.centroid.astype(np.float64)
        nearby: list[tuple[float, FrontierRecord]] = []
        for rec in active_frontiers:
            if rec.frontier_id == frontier.frontier_id:
                continue
            dist = float(np.linalg.norm(rec.centroid.astype(np.float64) - center))
            if dist <= self.target_context_radius_m:
                nearby.append((dist, rec))
        nearby.sort(key=lambda item: item[0])
        return [frontier] + [rec for _dist, rec in nearby[:7]]

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
        cam_from_world = np.linalg.inv(cam_pose)
        if target_visibility <= 0.0 or self._project_to_image(
            cam_pose, focus, camera_info, cam_from_world=cam_from_world
        ) is None:
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
            proj = self._project_to_image(cam_pose, target, camera_info, cam_from_world=cam_from_world)
            if proj is None:
                continue
            visibility = self._ray_clearance_score(checker, cam_pos, target)
            if visibility <= 0.0:
                continue
            visible_local += 1
            alignment_sum += optical_align
        local_coverage = float(visible_local) / float(total_local)
        mean_alignment = 0.0 if visible_local == 0 else alignment_sum / float(visible_local)

        gain_score = self._candidate_gain(cam_pose, cam_pos, frontier, active_frontiers, checker, camera_info)
        visibility_score = (
            0.65 * target_visibility
            + 0.35 * local_coverage
            + 0.10 * mean_alignment
            + 0.10 * gain_score
        )
        return {
            "visibility_score": float(visibility_score),
            "local_coverage": float(local_coverage),
            "view_alignment": float(mean_alignment),
            "gain_score": float(gain_score),
            "target_visibility": float(target_visibility),
            "global_context": 0.0,
            "visible_frontiers": float(visible_local),
        }

    def _project_to_image(
        self,
        cam_pose: np.ndarray,
        point_world: np.ndarray,
        camera_info: CameraInfo | None,
        *,
        cam_from_world: np.ndarray | None = None,
    ) -> tuple[float, float, float] | None:
        if camera_info is None:
            return None
        if cam_from_world is None:
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
        max_frontiers: int | None = None,
        progress_cb=None,
    ) -> list[ViewGoal]:
        current_q = np.asarray(current_q, dtype=np.float64)
        current_camera_xyz = None if current_camera_xyz is None else np.asarray(current_camera_xyz, dtype=np.float64)
        ranked: list[ViewGoal] = []
        frontier_limit = 12 if max_frontiers is None else max(1, int(max_frontiers))
        ordered_frontiers = sorted(frontiers, key=lambda rec: rec.voxel_count, reverse=True)
        if len(ordered_frontiers) <= frontier_limit:
            ranked_frontiers = ordered_frontiers
        else:
            # A small fallback budget should cover different cabinet faces,
            # rather than repeatedly scoring adjacent high-count voxels.
            ranked_frontiers = [ordered_frontiers[0]]
            remaining_frontiers = list(ordered_frontiers[1:])
            while remaining_frontiers and len(ranked_frontiers) < frontier_limit:
                best_index = max(
                    range(len(remaining_frontiers)),
                    key=lambda idx: (
                        min(
                            float(
                                np.linalg.norm(
                                    remaining_frontiers[idx].centroid.astype(np.float64)
                                    - selected.centroid.astype(np.float64)
                                )
                            )
                            for selected in ranked_frontiers
                        ),
                        int(remaining_frontiers[idx].voxel_count),
                    ),
                )
                ranked_frontiers.append(remaining_frontiers.pop(best_index))
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
            "rejected_frontier_ids": [],
        }
        last_progress_count = 0
        for frontier in ranked_frontiers:
            accepted_before_frontier = int(stats["accepted"])
            if (
                step_idx is not None
                and self.frontier_reselect_cooldown_steps > 0
                and frontier.last_selected_step >= 0
                and (int(step_idx) - int(frontier.last_selected_step)) < self.frontier_reselect_cooldown_steps
            ):
                stats["rejected_cooldown"] += 1
                continue
            for cam_pose, pose_kind in self._frontier_pose_subset(frontier):
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
                if self._ray_clearance_score(checker, cam_pos, frontier.centroid.astype(np.float64)) <= 0.0:
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
                target_ok, reject_reason = self._actual_target_view_is_valid(
                    checker,
                    q_goal,
                    actual_cam_pose,
                    target,
                    camera_info,
                )
                if not target_ok:
                    if reject_reason == "orientation":
                        stats["rejected_orientation"] += 1
                    elif reject_reason == "self_occlusion":
                        stats["rejected_self_occlusion"] += 1
                    else:
                        stats["rejected_visibility"] += 1
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
            if int(stats["accepted"]) == accepted_before_frontier:
                stats.setdefault("rejected_frontier_ids", []).append(int(frontier.frontier_id))
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
