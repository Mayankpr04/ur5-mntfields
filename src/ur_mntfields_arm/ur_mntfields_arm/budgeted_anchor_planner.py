from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ur_mntfields_arm.planner import ArmFieldPlanner
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


@dataclass(frozen=True)
class BudgetedAnchorConfig:
    """Configuration for a deliberately tiny, environment-derived anchor set."""

    anchor_budget: int = 1
    max_candidates: int = 8
    max_probes: int = 8
    workspace_shell_standoff_m: float = 0.25
    workspace_vertical_bias_fraction: float = -0.055
    anchor_clearance_min_m: float = 0.05
    probe_clearance_min_m: float = 0.02
    learned_min_speed: float = 0.20
    planning_edge_step_rad: float = 0.04
    enforce_learned_safety: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.anchor_budget) <= 2:
            raise ValueError("anchor_budget must be 1 or 2")


class BudgetedFieldAnchorPlanner:
    """Field planner with one opening-centred, environment-derived anchor.

    Scene bounds determine the camera pose and replay states supply only IK
    seeds. The anchor is used only when two endpoints are separated by an
    observed internal shelf plane. Runtime routing remains field-only; the
    geometric checker is not queried by :meth:`plan`.
    """

    def __init__(
        self,
        field_planner: ArmFieldPlanner,
        kinematics: UR5Kinematics,
        config: BudgetedAnchorConfig | None = None,
        camera_in_tool: np.ndarray | None = None,
    ):
        self.field_planner = field_planner
        self.kinematics = kinematics
        self.config = config or BudgetedAnchorConfig()
        self.camera_in_tool = (
            np.eye(4, dtype=np.float64)
            if camera_in_tool is None
            else np.asarray(camera_in_tool, dtype=np.float64).reshape(4, 4)
        )
        self.anchors = np.zeros((0, 6), dtype=np.float64)
        self.anchor_stats: list[dict[str, float]] = []
        self.last_debug: dict[str, object] = {}
        self.scene_bounds_min: np.ndarray | None = None
        self.scene_bounds_max: np.ndarray | None = None
        self.retreat_face_axis: int | None = None
        self.retreat_face_value: float | None = None
        self.retreat_outward_sign: float | None = None
        self.separator_planes: list[tuple[int, float]] = []
        self.candidate_generation_debug: dict[str, object] = {}

    def select_anchors(
        self,
        checker,
        candidate_qs: np.ndarray,
        probe_qs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select one opening-centred retreat pose and its best IK branch.

        The scene bounding box determines the workspace pose. Replay states are
        used only as IK seeds and as a weak joint-branch prior; they cannot move
        the anchor away from the opening. This keeps selection deterministic and
        avoids the expensive candidate-by-probe set-cover calculation.
        """

        started = time.perf_counter()
        supplied = np.asarray(candidate_qs, dtype=np.float64).reshape(-1, 6)
        probe_source = supplied if probe_qs is None else np.asarray(probe_qs, dtype=np.float64).reshape(-1, 6)
        raw_candidates, target_camera_xyz = self._opening_anchor_candidates(
            checker, supplied, probe_source
        )
        raw_clearance = (
            checker.clearance_batch(raw_candidates.astype(np.float32)).astype(np.float64)
            if len(raw_candidates)
            else np.zeros((0,), dtype=np.float64)
        )
        anchor_minimum = float(self.config.anchor_clearance_min_m)
        if bool(getattr(checker, "learned_only", False)):
            anchor_minimum = max(anchor_minimum, float(self.config.learned_min_speed))
        valid_anchor = np.isfinite(raw_clearance) & (raw_clearance >= anchor_minimum)
        candidates = raw_candidates[valid_anchor]
        probe_subset = self._hub_seed_subset(
            probe_source, limit=max(1, int(self.config.max_probes))
        )
        probes = self._valid_free_rows(checker, probe_subset, self.config.probe_clearance_min_m)
        if not len(candidates):
            self.anchors = np.zeros((0, 6), dtype=np.float64)
            self.anchor_stats = []
            self.last_debug = {
                "status": (
                    "no_opening_anchor_ik"
                    if not len(raw_candidates)
                    else "opening_anchor_clearance_rejected"
                ),
                "candidate_count": 0,
                "raw_candidate_count": int(len(raw_candidates)),
                "raw_candidate_clearance_m": np.round(raw_clearance, 4).tolist(),
                "probe_count": int(len(probes)),
                "target_camera_xyz": target_camera_xyz.tolist() if target_camera_xyz is not None else None,
                "candidate_generation": dict(self.candidate_generation_debug),
                "selection_ms": float((time.perf_counter() - started) * 1.0e3),
            }
            return self.anchors.copy()

        candidates = candidates[: int(self.config.max_candidates)]
        clearance = checker.clearance_batch(candidates.astype(np.float32)).astype(np.float64)
        reference = self._hub_seed_subset(probes if len(probes) else supplied, limit=32)
        reference_n = np.asarray([self.kinematics.normalize(q) for q in reference], dtype=np.float64)
        reference_center = np.median(reference_n, axis=0) if len(reference_n) else np.zeros((6,))
        candidate_n = np.asarray([self.kinematics.normalize(q) for q in candidates], dtype=np.float64)
        joint_distance = np.linalg.norm(candidate_n - reference_center[None, :], axis=1)
        clearance_scale = max(float(np.max(clearance)), 1.0e-6)
        distance_scale = max(float(np.max(joint_distance)), 1.0e-6)
        # Geometry decides the workspace anchor. This score only selects a
        # well-cleared, familiar IK branch for that exact camera pose.
        cost = (
            0.75 * joint_distance / distance_scale
            - 0.25 * clearance / clearance_scale
        )
        selected_idx = int(np.argmin(cost))
        self.anchors = candidates[selected_idx : selected_idx + 1].astype(np.float64, copy=True)
        stats = [
            {
                "selection_index": 0.0,
                "score": float(-cost[selected_idx]),
                "anchor_clearance_m": float(clearance[selected_idx]),
                "joint_branch_distance": float(joint_distance[selected_idx]),
            }
        ]
        self.anchor_stats = stats
        self.last_debug = {
            "status": "opening_anchor_selected",
            "anchor_count": 1,
            "anchor_budget": int(self.config.anchor_budget),
            "candidate_count": int(len(candidates)),
            "raw_candidate_count": int(len(raw_candidates)),
            "raw_candidate_clearance_m": np.round(raw_clearance, 4).tolist(),
            "candidate_source": "opening_center_ik",
            "probe_count": int(len(probes)),
            "target_camera_xyz": target_camera_xyz.tolist() if target_camera_xyz is not None else None,
            "candidate_generation": dict(self.candidate_generation_debug),
            "selection_ms": float((time.perf_counter() - started) * 1.0e3),
            "anchor_stats": list(stats),
        }
        return self.anchors.copy()

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float = 0.03,
        max_steps: int = 120,
        force_anchor: bool = False,
    ) -> np.ndarray:
        """Plan with the field directly, then through the fixed tiny anchor set."""

        started = time.perf_counter()
        q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)
        direct_edge_min_speed = float("nan")
        crosses_separator = self._crosses_scene_separator(q_start, q_goal)
        force_anchor = bool(force_anchor and len(self.anchors))
        if (force_anchor or crosses_separator) and len(self.anchors):
            routes: list[tuple[str, list[np.ndarray]]] = [
                (f"anchor_{idx + 1}", [anchor, q_goal])
                for idx, anchor in enumerate(self.anchors)
            ]
        else:
            routes = [("direct", [q_goal])]
        if crosses_separator and len(self.anchors) == 2:
            routes.extend(
                [
                    ("anchor_1_2", [self.anchors[0], self.anchors[1], q_goal]),
                    ("anchor_2_1", [self.anchors[1], self.anchors[0], q_goal]),
                ]
            )

        route_results: list[dict[str, object]] = []
        enforce_safety = bool(self.config.enforce_learned_safety)
        for route_name, targets in routes:
            cur = q_start
            pieces: list[np.ndarray] = []
            bottleneck = float("inf")
            valid = True
            statuses: list[str] = []
            for target in targets:
                if enforce_safety:
                    path = self.field_planner.plan_learned_speed_search(
                        cur,
                        target,
                        step_size_q,
                        max_steps,
                        min_predicted_speed=self.config.learned_min_speed,
                        allow_direct_edge=True,
                        mode="forward",
                    )
                else:
                    # Explicit simulation diagnostic for a partial checkpoint:
                    # preserve anchor topology but exercise the raw travel-time
                    # gradient, without claiming state-head certification.
                    path = self.field_planner.plan(
                        cur,
                        target,
                        step_size_q,
                        max_steps,
                        mode="forward",
                        allow_direct_edge=False,
                    )
                statuses.append(str(self.field_planner.last_debug.get("status", "")))
                if np.asarray(path).ndim != 2 or len(path) < 2:
                    valid = False
                    break
                piece = np.asarray(path, dtype=np.float64)
                if enforce_safety:
                    leg_debug = dict(self.field_planner.last_debug)
                    cached_edge_speed = leg_debug.get("direct_edge_min_speed")
                    leg_speed = (
                        float(cached_edge_speed)
                        if cached_edge_speed is not None and np.isfinite(float(cached_edge_speed))
                        else self._path_min_speed(piece, np.asarray(target, dtype=np.float64))
                    )
                else:
                    leg_speed = 1.0
                bottleneck = min(bottleneck, leg_speed)
                pieces.append(piece if not pieces else piece[1:])
                cur = np.asarray(target, dtype=np.float64)
            combined = np.concatenate(pieces, axis=0) if valid and pieces else np.zeros((0, 6), dtype=np.float64)
            length = (
                float(np.sum(np.linalg.norm(np.diff(combined, axis=0), axis=1)))
                if len(combined) > 1
                else float("inf")
            )
            route_results.append(
                {
                    "name": route_name,
                    "path": combined,
                    "valid": bool(valid and len(combined) >= 2),
                    "bottleneck_speed": float(bottleneck),
                    "length": float(length),
                    "statuses": statuses,
                }
            )
            if route_name == "direct" and np.isfinite(bottleneck):
                direct_edge_min_speed = float(bottleneck)

        viable = [
            result
            for result in route_results
            if result["valid"]
            and (
                not enforce_safety
                or float(result["bottleneck_speed"]) >= self.config.learned_min_speed
            )
        ]
        required_anchor_routes = [
            result
            for result in route_results
            if (force_anchor or crosses_separator)
            and result["name"] != "direct"
        ]
        if required_anchor_routes:
            required_viable = [
                result for result in required_anchor_routes
                if result["valid"]
                and (
                    not enforce_safety
                    or float(result["bottleneck_speed"]) >= self.config.learned_min_speed
                )
            ]
            if not required_viable:
                self.last_debug = {
                    "status": "budgeted_anchor_learned_safety_rejected",
                    "anchor_count": int(len(self.anchors)),
                    "forced_anchor": bool(force_anchor),
                    "crosses_scene_separator": bool(crosses_separator),
                    "routes": self._debug_routes(route_results),
                    "plan_ms": float((time.perf_counter() - started) * 1.0e3),
                }
                return np.zeros((0, 6), dtype=np.float32)
            best = min(
                required_viable,
                key=lambda result: (float(result["length"]), -float(result["bottleneck_speed"])),
            )
        elif viable:
            best = min(
                viable,
                key=lambda result: (float(result["length"]), -float(result["bottleneck_speed"])),
            )
        else:
            self.last_debug = {
                "status": "budgeted_anchor_learned_safety_rejected",
                "anchor_count": int(len(self.anchors)),
                "routes": self._debug_routes(route_results),
                "plan_ms": float((time.perf_counter() - started) * 1.0e3),
            }
            return np.zeros((0, 6), dtype=np.float32)
        self.last_debug = {
            "status": (
                "budgeted_anchor_reached"
                if enforce_safety
                else "budgeted_anchor_diagnostic_raw_field_reached"
            ),
            "selected_route": str(best["name"]),
            "used_anchor": str(best["name"]) != "direct",
            "fallback_below_threshold": not enforce_safety,
            "learned_safety_enforced": enforce_safety,
            "anchor_count": int(len(self.anchors)),
            "forced_anchor": bool(force_anchor),
            "crosses_scene_separator": bool(crosses_separator),
            "routing_reason": (
                "startup_anchor"
                if force_anchor and str(best["name"]) != "direct"
                else "scene_separator"
                if crosses_separator and str(best["name"]) != "direct"
                else "same_region"
            ),
            "direct_edge_min_speed": float(direct_edge_min_speed),
            "bottleneck_speed": float(best["bottleneck_speed"]),
            "path_length": float(best["length"]),
            "routes": self._debug_routes(route_results),
            "plan_ms": float((time.perf_counter() - started) * 1.0e3),
        }
        return np.asarray(best["path"], dtype=np.float32)

    def _valid_free_rows(self, checker, rows: np.ndarray, minimum: float) -> np.ndarray:
        q = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        if len(q) == 0:
            return q
        q = np.asarray([self.kinematics.clamp(row) for row in q], dtype=np.float64)
        qn = np.asarray([self.kinematics.normalize(row) for row in q], dtype=np.float32)
        _, unique_idx = np.unique(np.round(qn, 3), axis=0, return_index=True)
        q = q[np.sort(unique_idx)]
        clearance = checker.clearance_batch(q.astype(np.float32))
        required = float(minimum)
        if bool(getattr(checker, "learned_only", False)):
            required = max(required, float(self.config.learned_min_speed))
        valid = np.isfinite(clearance) & (clearance >= required)
        return q[valid]

    def _opening_anchor_candidates(
        self,
        checker,
        seed_rows: np.ndarray,
        workspace_reference_qs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Generate IK branches for one camera pose centred outside the opening."""

        boxes = np.asarray(getattr(checker, "box_obstacles", np.zeros((0, 6))), dtype=np.float64).reshape(-1, 6)
        support_count = int(getattr(checker, "support_box_count", 0))
        if support_count > 0 and len(boxes) >= support_count:
            boxes = boxes[:-support_count]
        occupied = np.asarray(getattr(checker, "occupied_points", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        bounds_min: np.ndarray
        bounds_max: np.ndarray
        if len(boxes):
            bounds_min = np.min(boxes[:, :3] - 0.5 * boxes[:, 3:], axis=0)
            bounds_max = np.max(boxes[:, :3] + 0.5 * boxes[:, 3:], axis=0)
        elif len(occupied):
            bounds_min = np.quantile(occupied, 0.02, axis=0)
            bounds_max = np.quantile(occupied, 0.98, axis=0)
        else:
            return np.zeros((0, 6), dtype=np.float64), None
        if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
            return np.zeros((0, 6), dtype=np.float64), None

        # The opening has a horizontal normal. Considering z here previously
        # selected the cabinet bottom as the "nearest face" for a raised robot.
        face_options: list[tuple[float, int, float, float]] = []
        for axis in (0, 1):
            if bounds_min[axis] > 0.0:
                face_options.append((float(bounds_min[axis]), axis, float(bounds_min[axis]), -1.0))
            elif bounds_max[axis] < 0.0:
                face_options.append((-float(bounds_max[axis]), axis, float(bounds_max[axis]), 1.0))
        if not face_options:
            for axis in (0, 1):
                face_options.append((abs(float(bounds_min[axis])), axis, float(bounds_min[axis]), -1.0))
                face_options.append((abs(float(bounds_max[axis])), axis, float(bounds_max[axis]), 1.0))
        _distance, face_axis, face_value, outward_sign = min(face_options, key=lambda item: item[0])
        self.scene_bounds_min = bounds_min.copy()
        self.scene_bounds_max = bounds_max.copy()
        self.retreat_face_axis = int(face_axis)
        self.retreat_face_value = float(face_value)
        self.retreat_outward_sign = float(outward_sign)
        self.separator_planes = self._find_separator_planes(boxes, bounds_min, bounds_max)
        extent = np.maximum(bounds_max - bounds_min, 1.0e-3)
        center = 0.5 * (bounds_min + bounds_max)
        center[2] += float(self.config.workspace_vertical_bias_fraction) * float(extent[2])
        lateral_axis = 1 if face_axis == 0 else 0
        reference_q = np.asarray(
            np.zeros((0, 6)) if workspace_reference_qs is None else workspace_reference_qs,
            dtype=np.float64,
        ).reshape(-1, 6)
        if len(reference_q):
            reference_camera_xyz = np.asarray(
                [
                    self.kinematics.tool_to_camera_pose(
                        self.kinematics.fk(q), self.camera_in_tool
                    )[:3, 3]
                    for q in reference_q
                ],
                dtype=np.float64,
            )
            natural_lateral = float(np.median(reference_camera_xyz[:, lateral_axis]))
            lateral_half_band = 0.10 * float(extent[lateral_axis])
            center[lateral_axis] = float(
                np.clip(
                    natural_lateral,
                    center[lateral_axis] - lateral_half_band,
                    center[lateral_axis] + lateral_half_band,
                )
            )
        standoff = max(0.05, float(self.config.workspace_shell_standoff_m))
        target = center.copy()
        target[face_axis] = face_value + outward_sign * standoff

        # Optical z faces into the structure, optical y remains gravity-down.
        forward = np.zeros((3,), dtype=np.float64)
        forward[face_axis] = -outward_sign
        image_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        image_right = np.cross(image_down, forward)
        image_right /= max(float(np.linalg.norm(image_right)), 1.0e-6)
        desired_camera = np.eye(4, dtype=np.float64)
        desired_camera[:3, :3] = np.column_stack((image_right, image_down, forward))
        desired_camera[:3, 3] = target
        desired_tool = self.kinematics.camera_to_tool_pose(desired_camera, self.camera_in_tool)

        seed_q = np.asarray(seed_rows, dtype=np.float64).reshape(-1, 6)
        seed_limit = max(2, int(self.config.max_candidates))
        tail_count = min(len(seed_q), max(1, seed_limit // 2))
        tail = seed_q[-tail_count:] if tail_count else np.zeros((0, 6), dtype=np.float64)
        hub = self._hub_seed_subset(seed_q, limit=max(1, seed_limit - tail_count))
        seeds = np.vstack((tail, hub)) if len(tail) or len(hub) else np.zeros((0, 6))
        candidates: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        solved_count = 0
        rejected_position = 0
        rejected_forward = 0
        rejected_down = 0
        for seed in seeds:
            q = self.kinematics.solve_ik_fast(desired_tool, seed)
            if q is None:
                continue
            solved_count += 1
            actual_camera = self.kinematics.tool_to_camera_pose(
                self.kinematics.fk(q), self.camera_in_tool
            )
            position_error = float(np.linalg.norm(actual_camera[:3, 3] - target))
            forward_alignment = float(np.dot(actual_camera[:3, 2], forward))
            down_alignment = float(np.dot(actual_camera[:3, 1], image_down))
            if position_error > 0.04:
                rejected_position += 1
                continue
            if forward_alignment < 0.95:
                rejected_forward += 1
                continue
            if down_alignment < 0.95:
                rejected_down += 1
                continue
            key = tuple(np.round(q, 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(np.asarray(q, dtype=np.float64))
        self.candidate_generation_debug = {
            "seed_count": int(len(seeds)),
            "ik_solved_count": int(solved_count),
            "rejected_position": int(rejected_position),
            "rejected_forward": int(rejected_forward),
            "rejected_down": int(rejected_down),
            "accepted_pose_count": int(len(candidates)),
        }
        rows = (
            np.asarray(candidates, dtype=np.float64).reshape(-1, 6)
            if candidates
            else np.zeros((0, 6), dtype=np.float64)
        )
        return rows, target

    @staticmethod
    def _find_separator_planes(
        boxes: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray
    ) -> list[tuple[int, float]]:
        """Find thin internal boxes that partition the scene into regions."""

        extent = np.maximum(np.asarray(bounds_max) - np.asarray(bounds_min), 1.0e-3)
        planes: list[tuple[int, float]] = []
        for box in np.asarray(boxes, dtype=np.float64).reshape(-1, 6):
            center, size = box[:3], box[3:]
            for axis in range(3):
                other_axes = [idx for idx in range(3) if idx != axis]
                thin = float(size[axis]) <= max(0.08, 0.12 * float(extent[axis]))
                spanning = all(float(size[idx]) >= 0.50 * float(extent[idx]) for idx in other_axes)
                relative = float((center[axis] - bounds_min[axis]) / extent[axis])
                internal = 0.10 < relative < 0.90
                if thin and spanning and internal:
                    planes.append((axis, float(center[axis])))
        unique: list[tuple[int, float]] = []
        for axis, value in sorted(planes):
            if not any(axis == old_axis and abs(value - old_value) < 0.03 for old_axis, old_value in unique):
                unique.append((axis, value))
        return unique

    def _crosses_scene_separator(self, q_start: np.ndarray, q_goal: np.ndarray) -> bool:
        """Return true when the endpoint cameras belong to different regions.

        Region membership is defined in camera workspace, because the camera is
        the task point used to enter and observe a shelf.  Endpoints in the
        opening approach band also belong to a shelf; requiring the camera to
        already be behind the opening plane incorrectly classified valid
        high-shelf poses as generic free-space poses.
        """

        if (
            not len(self.anchors)
            or self.scene_bounds_min is None
            or self.scene_bounds_max is None
            or self.retreat_face_axis is None
            or self.retreat_face_value is None
            or self.retreat_outward_sign is None
            or not self.separator_planes
        ):
            return False
        start_xyz = np.asarray(
            self.kinematics.tool_to_camera_pose(
                self.kinematics.fk(q_start), self.camera_in_tool
            )[:3, 3],
            dtype=np.float64,
        )
        goal_xyz = np.asarray(
            self.kinematics.tool_to_camera_pose(
                self.kinematics.fk(q_goal), self.camera_in_tool
            )[:3, 3],
            dtype=np.float64,
        )
        axis = int(self.retreat_face_axis)
        inward_sign = -float(self.retreat_outward_sign)
        tolerance = 0.02
        start_depth = inward_sign * (float(start_xyz[axis]) - float(self.retreat_face_value))
        goal_depth = inward_sign * (float(goal_xyz[axis]) - float(self.retreat_face_value))
        approach_depth = max(
            tolerance,
            float(self.config.workspace_shell_standoff_m) + tolerance,
        )
        if start_depth < -approach_depth or goal_depth < -approach_depth:
            return False
        # Reject points far outside the scene in the tangent directions. The
        # normal-direction approach band above deliberately retains camera
        # poses immediately in front of a compartment opening.
        extent = np.maximum(self.scene_bounds_max - self.scene_bounds_min, 1.0e-3)
        for tangent_axis in range(3):
            if tangent_axis == axis:
                continue
            margin = 0.10 * float(extent[tangent_axis])
            if not (
                self.scene_bounds_min[tangent_axis] - margin
                <= start_xyz[tangent_axis]
                <= self.scene_bounds_max[tangent_axis] + margin
                and self.scene_bounds_min[tangent_axis] - margin
                <= goal_xyz[tangent_axis]
                <= self.scene_bounds_max[tangent_axis] + margin
            ):
                return False
        for separator_axis, value in self.separator_planes:
            start_side = float(start_xyz[separator_axis] - value)
            goal_side = float(goal_xyz[separator_axis] - value)
            if start_side * goal_side < -(tolerance * tolerance):
                return True
        return False

    def _hub_seed_subset(self, rows: np.ndarray, limit: int) -> np.ndarray:
        q = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        if len(q) <= limit:
            return q.copy()
        qn = np.asarray([self.kinematics.normalize(row) for row in q], dtype=np.float32)
        centroid = np.median(qn, axis=0)
        order = np.argsort(np.linalg.norm(qn - centroid[None, :], axis=1), kind="stable")
        return q[order[:limit]]

    def _path_min_speed(self, path: np.ndarray, target: np.ndarray) -> float:
        values = []
        for qa, qb in zip(path[:-1], path[1:]):
            values.append(
                self.field_planner.learned_edge_min_speed(
                    qa, qb, target, max_step_rad=self.config.planning_edge_step_rad
                )
            )
        return float(min(values)) if values else -float("inf")

    @staticmethod
    def _debug_routes(routes: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "name": str(route["name"]),
                "valid": bool(route["valid"]),
                "bottleneck_speed": float(route["bottleneck_speed"]),
                "length": float(route["length"]),
                "statuses": list(route["statuses"]),
            }
            for route in routes
        ]
