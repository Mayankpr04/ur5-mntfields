from __future__ import annotations

import heapq
import time

import numpy as np
import torch

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


class _LearnedSpeedChecker:
    """Planner edge oracle backed only by batched neural speed inference."""

    def __init__(
        self,
        field_model: ArmFieldModel,
        kinematics: UR5Kinematics,
        q_target: np.ndarray,
        *,
        deadline_at: float | None = None,
    ):
        self.field_model = field_model
        self.kinematics = kinematics
        self.q_target_n = kinematics.normalize(np.asarray(q_target, dtype=np.float64)).astype(np.float32)
        self.query_calls = 0
        self.query_states = 0
        self.requested_states = 0
        self._speed_cache: dict[bytes, float] = {}
        self.rejected_unsafe = 0
        self.rejected_coverage = 0
        self.deadline_at = deadline_at
        self.deadline_rejections = 0

    @property
    def expired(self) -> bool:
        return self.deadline_at is not None and time.perf_counter() >= self.deadline_at

    def clearance_batch(self, q_batch: np.ndarray) -> np.ndarray:
        q = np.asarray(q_batch, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        if len(q) == 0:
            return np.zeros((0,), dtype=np.float32)
        qn = np.asarray([self.kinematics.normalize(row) for row in q], dtype=np.float32)
        return self.clearance_normalized_batch(qn)

    def clearance_normalized_batch(self, qn_batch: np.ndarray) -> np.ndarray:
        """Evaluate already-normalized states without Python FK conversions."""
        qn = np.asarray(qn_batch, dtype=np.float32).reshape(-1, 6)
        if len(qn) == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.expired:
            self.deadline_rejections += int(len(qn))
            return np.zeros((len(qn),), dtype=np.float32)
        qn = np.clip(qn, -0.5, 0.5)
        self.requested_states += int(len(qn))
        keys = [np.ascontiguousarray(row).tobytes() for row in qn]
        missing_rows: list[np.ndarray] = []
        missing_keys: list[bytes] = []
        seen_missing: set[bytes] = set()
        for key, row in zip(keys, qn):
            if key not in self._speed_cache and key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_rows.append(row)
        if missing_rows:
            query = np.asarray(missing_rows, dtype=np.float32)
            if hasattr(self.field_model, "predict_normalized_state_geometry"):
                speed_raw, unsafe_probability, speed = self.field_model.predict_normalized_state_geometry(query)
                speed = np.asarray(speed, dtype=np.float32)
                unsafe_probability = np.asarray(unsafe_probability, dtype=np.float32)
                unsafe_mask = unsafe_probability >= 0.10
                self.rejected_unsafe += int(np.count_nonzero(unsafe_mask))
                speed[unsafe_mask] = 0.0
                tree = getattr(self.field_model, "coverage_tree", None)
                if tree is None:
                    coverage_mask = np.ones(len(query), dtype=bool)
                else:
                    distances, _indices = tree.query(query, k=1)
                    shell_radius = float(getattr(self.field_model, "shell_coverage_radius", 0.0))
                    free_radius = float(getattr(self.field_model, "free_coverage_radius", 0.0))
                    radii = np.where(speed_raw < 0.5, shell_radius, free_radius)
                    coverage_mask = np.asarray(distances) > radii
                self.rejected_coverage += int(np.count_nonzero(coverage_mask))
                speed[coverage_mask] = 0.0
            else:
                # Diagnostic compatibility for v2 fields and lightweight test
                # doubles. Certified execution never takes this branch.
                goals = np.repeat(self.q_target_n[None, :], len(query), axis=0)
                speed, _other = self.field_model.predict_normalized_pair_speeds(query, goals)
            self.query_calls += 1
            self.query_states += int(len(query))
            for key, value in zip(missing_keys, np.asarray(speed, dtype=np.float32)):
                self._speed_cache[key] = float(value)
        return np.asarray([self._speed_cache[key] for key in keys], dtype=np.float32)

    def clearance(self, q: np.ndarray) -> float:
        values = self.clearance_batch(np.asarray(q, dtype=np.float64)[None, :])
        return float(values[0]) if len(values) else 0.0


class ArmFieldPlanner:
    def __init__(self, field_model: ArmFieldModel, kinematics: UR5Kinematics):
        self.field_model = field_model
        self.kinematics = kinematics
        self.last_debug: dict[str, object] = {}
        self.rng = np.random.default_rng()
        self.score_tau_weight = 0.55 #0.55
        self.score_goal_dist_weight = 5.0 #5.0
        self.score_depth_weight = 0.04
        self.score_clearance_weight = 0.0 #0.2
        self.path_joint_edge_weight = 0.15
        self.path_turn_weight = 0.25
        self.path_tool_edge_weight = 0.0
        self.path_tool_goal_weight = 0.0
        self.path_clearance_penalty_weight = 20.0
        self.path_clearance_soft_margin_m = 0.04
        self.path_cost_return_first_goal = True
        self.bidirectional_forward_probe_fraction = 0.15
        self.cartesian_candidate_count = 0
        self.cartesian_candidate_step_m = 0.05
        self.cartesian_candidate_damping = 1.0e-3
        self.cartesian_shortcut_enabled = False
        self.cartesian_shortcut_tool_weight = 1.0
        self.cartesian_shortcut_joint_weight = 0.10
        self.cartesian_shortcut_smoothness_weight = 0.0
        self.cartesian_shortcut_min_improvement = 0.01
        self.cartesian_shortcut_max_skip = 48
        self.cartesian_shortcut_try_reverse = False
        # Collision geometry is evaluated in physical joint space.  Keeping the
        # interpolation step independent of normalized joint spans prevents both
        # missed narrow collisions and excessive sampling on short local edges.
        self.planning_edge_step_rad = 0.10
        self.final_edge_step_rad = 0.02
        self.collision_smoothing_enabled = True
        self.collision_smoothing_passes = 3
        self.collision_smoothing_alpha = 0.35

    def set_score_weights(
        self,
        tau: float | None = None,
        goal_dist: float | None = None,
        depth: float | None = None,
        clearance: float | None = None,
        joint_edge: float | None = None,
        turn: float | None = None,
        tool_edge: float | None = None,
        tool_goal: float | None = None,
        clearance_penalty: float | None = None,
        clearance_soft_margin_m: float | None = None,
        return_first_goal: bool | None = None,
        forward_probe_fraction: float | None = None,
        cartesian_candidate_count: int | None = None,
        cartesian_candidate_step_m: float | None = None,
        cartesian_candidate_damping: float | None = None,
        cartesian_shortcut_enabled: bool | None = None,
        cartesian_shortcut_tool_weight: float | None = None,
        cartesian_shortcut_joint_weight: float | None = None,
        cartesian_shortcut_smoothness_weight: float | None = None,
        cartesian_shortcut_min_improvement: float | None = None,
        cartesian_shortcut_max_skip: int | None = None,
        cartesian_shortcut_try_reverse: bool | None = None,
    ) -> None:
        if tau is not None:
            self.score_tau_weight = float(tau)
        if goal_dist is not None:
            self.score_goal_dist_weight = float(goal_dist)
        if depth is not None:
            self.score_depth_weight = float(depth)
        if clearance is not None:
            self.score_clearance_weight = float(clearance)
        if joint_edge is not None:
            self.path_joint_edge_weight = float(joint_edge)
        if turn is not None:
            self.path_turn_weight = max(0.0, float(turn))
        if tool_edge is not None:
            self.path_tool_edge_weight = float(tool_edge)
            if tool_goal is None:
                self.path_tool_goal_weight = float(tool_edge)
        if tool_goal is not None:
            self.path_tool_goal_weight = float(tool_goal)
        if clearance_penalty is not None:
            self.path_clearance_penalty_weight = float(clearance_penalty)
        if clearance_soft_margin_m is not None:
            self.path_clearance_soft_margin_m = max(0.0, float(clearance_soft_margin_m))
        if return_first_goal is not None:
            self.path_cost_return_first_goal = bool(return_first_goal)
        if forward_probe_fraction is not None:
            self.bidirectional_forward_probe_fraction = float(np.clip(float(forward_probe_fraction), 0.0, 1.0))
        if cartesian_candidate_count is not None:
            self.cartesian_candidate_count = max(0, int(cartesian_candidate_count))
        if cartesian_candidate_step_m is not None:
            self.cartesian_candidate_step_m = max(1.0e-4, float(cartesian_candidate_step_m))
        if cartesian_candidate_damping is not None:
            self.cartesian_candidate_damping = max(1.0e-8, float(cartesian_candidate_damping))
        if cartesian_shortcut_enabled is not None:
            self.cartesian_shortcut_enabled = bool(cartesian_shortcut_enabled)
        if cartesian_shortcut_tool_weight is not None:
            self.cartesian_shortcut_tool_weight = max(0.0, float(cartesian_shortcut_tool_weight))
        if cartesian_shortcut_joint_weight is not None:
            self.cartesian_shortcut_joint_weight = max(0.0, float(cartesian_shortcut_joint_weight))
        if cartesian_shortcut_smoothness_weight is not None:
            self.cartesian_shortcut_smoothness_weight = max(0.0, float(cartesian_shortcut_smoothness_weight))
        if cartesian_shortcut_min_improvement is not None:
            self.cartesian_shortcut_min_improvement = max(0.0, float(cartesian_shortcut_min_improvement))
        if cartesian_shortcut_max_skip is not None:
            self.cartesian_shortcut_max_skip = max(2, int(cartesian_shortcut_max_skip))
        if cartesian_shortcut_try_reverse is not None:
            self.cartesian_shortcut_try_reverse = bool(cartesian_shortcut_try_reverse)

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        mode: str = "bidirectional",
        allow_direct_edge: bool = False,
        min_predicted_speed: float = 0.20,
        edge_check_step_rad: float = 0.04,
    ) -> np.ndarray:
        q_start_n = self.kinematics.normalize(q_start)
        q_goal_n = self.kinematics.normalize(q_goal)
        tol = max(0.01, 0.5 * float(step_size_q))
        mode_norm = str(mode).strip().lower()
        direct_min_speed = float("nan")
        direct_ms = 0.0
        if allow_direct_edge:
            direct_t0 = time.perf_counter()
            direct_min_speed = self.learned_edge_min_speed(
                q_start,
                q_goal,
                q_goal,
                max_step_rad=edge_check_step_rad,
            )
            direct_ms = (time.perf_counter() - direct_t0) * 1.0e3
        if allow_direct_edge and direct_min_speed >= float(min_predicted_speed):
            self.last_debug = {
                "status": "learned_direct_edge",
                "planner": "field_only",
                "search_direction": "direct",
                "steps": 0,
                "direct_edge_min_speed": float(direct_min_speed),
                "direct_edge_infer_ms": float(direct_ms),
            }
            return np.asarray([q_start, q_goal], dtype=np.float32)
        if mode_norm in ("goal_to_start", "reverse_only", "goal_to_start_only"):
            rollout_t0 = time.perf_counter()
            path_rev = self.field_model.gradient_rollout(
                q_goal_n, q_start_n, step_size=step_size_q, max_steps=max_steps, tol=tol
            )
            rollout_ms = (time.perf_counter() - rollout_t0) * 1.0e3
            if not self._path_reached_goal(path_rev, q_start_n, tol):
                self.last_debug = {
                    "status": "field_only_goal_to_start_failed",
                    "planner": "field_only",
                    "search_direction": "goal_to_start",
                    "steps": int(max(0, len(path_rev) - 1)),
                    "direct_edge_min_speed": float(direct_min_speed),
                    "direct_edge_infer_ms": float(direct_ms),
                    "gradient_rollout_ms": float(rollout_ms),
                    "last_goal_dist": float(
                        np.linalg.norm(np.asarray(path_rev[-1], dtype=np.float32) - q_start_n)
                    )
                    if len(path_rev)
                    else float("inf"),
                }
                return np.zeros((0, 6), dtype=np.float32)
            path_n = self._ensure_endpoints(path_rev, q_goal_n, q_start_n)[::-1]
            self.last_debug = {
                "status": "field_only_goal_to_start_reached",
                "planner": "field_only",
                "search_direction": "goal_to_start",
                "steps": int(max(0, len(path_n) - 1)),
                "last_goal_dist": 0.0,
                "direct_edge_min_speed": float(direct_min_speed),
                "direct_edge_infer_ms": float(direct_ms),
                "gradient_rollout_ms": float(rollout_ms),
            }
            return np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)
        forward_t0 = time.perf_counter()
        path_fwd = self.field_model.gradient_rollout(
            q_start_n, q_goal_n, step_size=step_size_q, max_steps=max_steps, tol=tol
        )
        forward_ms = (time.perf_counter() - forward_t0) * 1.0e3
        if mode_norm in ("forward", "forward_only", "start_to_goal"):
            if not self._path_reached_goal(path_fwd, q_goal_n, tol):
                self.last_debug = {
                    "status": "field_only_forward_failed",
                    "planner": "field_only",
                    "search_direction": "forward",
                    "steps": int(max(0, len(path_fwd) - 1)),
                    "direct_edge_min_speed": float(direct_min_speed),
                    "direct_edge_infer_ms": float(direct_ms),
                    "gradient_rollout_ms": float(forward_ms),
                }
                return np.zeros((0, 6), dtype=np.float32)
            path_n = self._ensure_endpoints(path_fwd, q_start_n, q_goal_n)
            self.last_debug = {
                "status": "field_only_forward_reached",
                "planner": "field_only",
                "search_direction": "forward",
                "steps": int(max(0, len(path_n) - 1)),
                "last_goal_dist": 0.0,
                "direct_edge_min_speed": float(direct_min_speed),
                "direct_edge_infer_ms": float(direct_ms),
                "gradient_rollout_ms": float(forward_ms),
            }
            return np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)
        if mode_norm not in ("bidirectional", "bi", "merge"):
            raise ValueError(f"Unsupported planner mode: {mode}")
        reverse_t0 = time.perf_counter()
        path_rev = self.field_model.gradient_rollout(
            q_goal_n, q_start_n, step_size=step_size_q, max_steps=max_steps, tol=tol
        )
        reverse_ms = (time.perf_counter() - reverse_t0) * 1.0e3
        path_n = self._merge_bidirectional_paths(path_fwd, path_rev, q_start_n, q_goal_n, bridge_tol=max(2.0 * tol, 1.5 * float(step_size_q)))
        if path_n.size == 0:
            self.last_debug = {
                "status": "field_only_bidirectional_failed",
                "planner": "field_only",
                "search_direction": "bidirectional",
                "direct_edge_min_speed": float(direct_min_speed),
                "direct_edge_infer_ms": float(direct_ms),
                "forward_rollout_ms": float(forward_ms),
                "reverse_rollout_ms": float(reverse_ms),
            }
            return np.zeros((0, 6), dtype=np.float32)
        self.last_debug = {
            "status": "field_only_bidirectional_reached",
            "planner": "field_only",
            "search_direction": "bidirectional",
            "steps": int(max(0, len(path_n) - 1)),
            "last_goal_dist": 0.0,
            "direct_edge_min_speed": float(direct_min_speed),
            "direct_edge_infer_ms": float(direct_ms),
            "forward_rollout_ms": float(forward_ms),
            "reverse_rollout_ms": float(reverse_ms),
        }
        return np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)

    def learned_edge_min_speed(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        q_goal: np.ndarray,
        max_step_rad: float = 0.04,
    ) -> float:
        """Return the minimum learned speed along a densely sampled joint edge."""

        qa = np.asarray(q_start, dtype=np.float64).reshape(6)
        qb = np.asarray(q_end, dtype=np.float64).reshape(6)
        max_delta = float(np.max(np.abs(qb - qa)))
        segments = max(1, int(np.ceil(max_delta / max(0.01, float(max_step_rad)))))
        edge = np.linspace(qa, qb, segments + 1, dtype=np.float64)
        edge_n = np.asarray([self.kinematics.normalize(row) for row in edge], dtype=np.float32)
        if hasattr(self.field_model, "predict_normalized_state_geometry"):
            oracle = _LearnedSpeedChecker(self.field_model, self.kinematics, q_goal)
            speed = oracle.clearance_batch(edge)
            if len(speed) != len(edge_n) or not np.all(np.isfinite(speed)):
                return float("-inf")
            return float(np.min(speed))
        goal_n = self.kinematics.normalize(np.asarray(q_goal, dtype=np.float64)).astype(np.float32)
        goals = np.repeat(goal_n[None, :], len(edge_n), axis=0)
        speed, _ = self.field_model.predict_normalized_pair_speeds(edge_n, goals, batch_size=2048)
        if len(speed) != len(edge_n) or not np.all(np.isfinite(speed)):
            return float("-inf")
        return float(np.min(speed))

    def learned_state_speeds(
        self,
        q_batch: np.ndarray,
        q_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the certified state/coverage oracle for physical states."""
        q = np.asarray(q_batch, dtype=np.float64).reshape(-1, 6)
        if len(q) == 0:
            return np.zeros((0,), dtype=np.float32)
        target = q[0] if q_target is None else np.asarray(q_target, dtype=np.float64)
        oracle = _LearnedSpeedChecker(self.field_model, self.kinematics, target)
        return oracle.clearance_batch(q)

    def _learned_replay_bridge(
        self,
        checker: _LearnedSpeedChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        *,
        threshold: float,
        step_size_q: float,
        max_candidates: int,
    ) -> np.ndarray:
        """Evaluate direct/two-leg supported routes in one neural batch."""
        q_start_n = self.kinematics.normalize(q_start).astype(np.float32)
        q_goal_n = self.kinematics.normalize(q_goal).astype(np.float32)
        midpoint = 0.5 * (q_start_n + q_goal_n)
        coverage = np.asarray(
            getattr(self.field_model, "coverage_states", np.zeros((0, 6))),
            dtype=np.float32,
        ).reshape(-1, 6)
        anchors = [midpoint]
        if len(coverage):
            # Retrieve a bounded local neighborhood from the replay index and
            # combine it with a small set of global cell landmarks. The old
            # implementation computed two norms across every 200k+ replay
            # state for every attempted route.
            tree = getattr(self.field_model, "coverage_tree", None)
            local_indices = np.zeros((0,), dtype=np.int64)
            if tree is not None:
                query_k = min(16, len(coverage))
                _distance, indices = tree.query(
                    np.stack((q_start_n, midpoint, q_goal_n)), k=query_k
                )
                local_indices = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
                local_indices = local_indices[local_indices < len(coverage)]
            landmarks = np.asarray(
                getattr(self.field_model, "coverage_landmarks", np.zeros((0, 6))),
                dtype=np.float32,
            ).reshape(-1, 6)
            if not len(landmarks):
                stride = max(1, int(np.ceil(len(coverage) / 64.0)))
                landmarks = coverage[::stride][:64]
            bounded = np.concatenate((coverage[local_indices], landmarks), axis=0)
            bounded = np.unique(np.round(bounded, decimals=6), axis=0).astype(np.float32)
            route_cost = (
                np.linalg.norm(bounded - q_start_n[None, :], axis=1)
                + np.linalg.norm(bounded - q_goal_n[None, :], axis=1)
            )
            take = min(max(1, int(max_candidates)), len(bounded))
            shortest_take = max(1, take // 2)
            shortest = np.argpartition(route_cost, shortest_take - 1)[:shortest_take]
            pool_take = min(len(bounded), max(take, 128))
            pool = np.argpartition(route_cost, pool_take - 1)[:pool_take]
            direction = q_goal_n - q_start_n
            direction_norm_sq = float(np.dot(direction, direction))
            if direction_norm_sq > 1.0e-8:
                rel = bounded[pool] - q_start_n[None, :]
                projection = (
                    (rel @ direction) / direction_norm_sq
                )[:, None] * direction[None, :]
                deviation = np.linalg.norm(rel - projection, axis=1)
            else:
                deviation = np.linalg.norm(bounded[pool] - midpoint[None, :], axis=1)
            detour_take = min(take - shortest_take, len(pool))
            detour = pool[np.argsort(deviation)[-detour_take:]] if detour_take else np.zeros(0, dtype=int)
            idx = np.unique(np.concatenate((shortest, detour)))
            anchors.extend(bounded[idx])
        anchors_n = np.unique(np.round(np.asarray(anchors), decimals=6), axis=0).astype(np.float32)

        # Coarse stage: reject unsafe/unsupported anchor states cheaply, then
        # densely validate only a handful of complete two-leg routes.  The
        # previous implementation expanded all 64 routes into thousands of
        # interpolation states before its first inference call.
        anchor_speed = checker.clearance_normalized_batch(anchors_n)
        if len(anchor_speed) != len(anchors_n) or checker.expired:
            return np.zeros((0, 6), dtype=np.float32)
        viable_anchor = np.flatnonzero(
            np.isfinite(anchor_speed) & (anchor_speed >= float(threshold))
        )
        if not len(viable_anchor):
            return np.zeros((0, 6), dtype=np.float32)
        coarse_cost = (
            np.linalg.norm(anchors_n[viable_anchor] - q_start_n[None, :], axis=1)
            + np.linalg.norm(anchors_n[viable_anchor] - q_goal_n[None, :], axis=1)
            - 0.05 * anchor_speed[viable_anchor]
        )
        dense_count = min(8, len(viable_anchor))
        short_count = min(max(1, dense_count // 2), len(viable_anchor))
        short_local = np.argsort(coarse_cost)[:short_count]
        direction = q_goal_n - q_start_n
        direction_norm_sq = float(np.dot(direction, direction))
        rel = anchors_n[viable_anchor] - q_start_n[None, :]
        if direction_norm_sq > 1.0e-8:
            projection = ((rel @ direction) / direction_norm_sq)[:, None] * direction[None, :]
            deviation = np.linalg.norm(rel - projection, axis=1)
        else:
            deviation = np.linalg.norm(rel, axis=1)
        detour_local = np.argsort(deviation)[-(dense_count - short_count):]
        selected_local = np.unique(np.concatenate((short_local, detour_local)))
        if len(selected_local) < dense_count:
            remainder = [i for i in np.argsort(coarse_cost) if i not in set(selected_local.tolist())]
            selected_local = np.concatenate(
                (selected_local, np.asarray(remainder[: dense_count - len(selected_local)], dtype=int))
            )
        selected_anchor_idx = viable_anchor[selected_local[:dense_count]]
        dense_anchors = anchors_n[selected_anchor_idx]

        rows: list[np.ndarray] = []
        spans: list[tuple[int, int, int, int]] = []
        cursor = 0
        step = max(0.01, float(step_size_q))
        for anchor in dense_anchors:
            n0 = max(1, int(np.ceil(np.max(np.abs(anchor - q_start_n)) / step)))
            n1 = max(1, int(np.ceil(np.max(np.abs(q_goal_n - anchor)) / step)))
            first = np.linspace(q_start_n, anchor, n0 + 1, dtype=np.float32)
            second = np.linspace(anchor, q_goal_n, n1 + 1, dtype=np.float32)
            rows.extend((first, second))
            spans.append((cursor, cursor + len(first), cursor + len(first), cursor + len(first) + len(second)))
            cursor += len(first) + len(second)
        if not rows or checker.expired:
            return np.zeros((0, 6), dtype=np.float32)
        qn = np.concatenate(rows, axis=0)
        speed = checker.clearance_normalized_batch(qn)
        if len(speed) != len(qn) or checker.expired:
            return np.zeros((0, 6), dtype=np.float32)
        viable: list[tuple[float, float, int]] = []
        for index, (a0, a1, b0, b1) in enumerate(spans):
            bottleneck = min(float(np.min(speed[a0:a1])), float(np.min(speed[b0:b1])))
            if np.isfinite(bottleneck) and bottleneck >= float(threshold):
                anchor = dense_anchors[index]
                length = float(
                    np.linalg.norm(anchor - q_start_n) + np.linalg.norm(q_goal_n - anchor)
                )
                viable.append((length, -bottleneck, index))
        if not viable:
            return np.zeros((0, 6), dtype=np.float32)
        _length, _negative_bottleneck, best = min(viable)
        anchor = dense_anchors[best]
        if np.max(np.abs(anchor - midpoint)) <= 1.0e-5:
            return np.asarray([q_start, q_goal], dtype=np.float32)
        return np.asarray(
            [q_start, self.kinematics.denormalize(anchor), q_goal], dtype=np.float32
        )

    def _path_reached_goal(self, path: np.ndarray, q_goal_n: np.ndarray, tol: float) -> bool:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            return False
        return bool(np.linalg.norm(pts[-1] - np.asarray(q_goal_n, dtype=np.float32)) <= max(1.0e-6, float(tol)))

    def plan_collision_aware(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        clearance_margin_m: float = 0.01,
        max_local_candidates: int = 32,
        allow_direct_edge: bool = True,
        shortcut_path: bool = True,
        mode: str = "bidirectional",
    ) -> np.ndarray:
        mode_norm = str(mode).strip().lower()
        if mode_norm in ("goal_to_start", "reverse_only", "goal_to_start_only"):
            reverse_path = self._plan_collision_aware_one_way(
                checker,
                q_goal,
                q_start,
                step_size_q,
                max_steps,
                clearance_margin_m=clearance_margin_m,
                max_local_candidates=max_local_candidates,
                search_label="goal_to_start",
                allow_direct_edge=allow_direct_edge,
            )
            reverse_debug = dict(self.last_debug)
            if np.asarray(reverse_path).ndim == 2 and len(reverse_path) > 0:
                path = np.asarray(reverse_path[::-1], dtype=np.float32)
                raw_waypoints = int(len(path))
                path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
                post_debug = {
                    key: value
                    for key, value in self.last_debug.items()
                    if str(key).startswith("cartesian_shortcut")
                }
                self.last_debug = {
                    "status": "goal_to_start_reached",
                    "search_direction": "goal_to_start",
                    "reverse_debug": reverse_debug,
                    "steps": reverse_debug.get("steps", 0),
                    "start_goal_dist": reverse_debug.get("start_goal_dist", -1.0),
                    "last_goal_dist": 0.0,
                    "valid_edge_count": int(reverse_debug.get("valid_edge_count", 0)),
                    "candidate_count": int(reverse_debug.get("candidate_count", 0)),
                    "best_candidate_goal_dist": float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
                    "best_edge_min_clearance": float(reverse_debug.get("best_edge_min_clearance", -1.0)),
                    "raw_waypoints": raw_waypoints,
                    "shortcut_waypoints": int(len(path)),
                    "shortcut_enabled": bool(shortcut_path),
                    **post_debug,
                }
                return path
            self.last_debug = {
                "status": "failed_goal_to_start_field_search",
                "search_direction": "goal_to_start",
                "reverse_debug": reverse_debug,
                "steps": int(reverse_debug.get("steps", 0)),
                "start_goal_dist": reverse_debug.get("start_goal_dist", -1.0),
                "last_goal_dist": reverse_debug.get("last_goal_dist", float("inf")),
                "valid_edge_count": int(reverse_debug.get("valid_edge_count", 0)),
                "candidate_count": int(reverse_debug.get("candidate_count", 0)),
                "best_candidate_goal_dist": float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
                "best_edge_min_clearance": float(reverse_debug.get("best_edge_min_clearance", -1.0)),
            }
            return np.zeros((0, 6), dtype=np.float32)

        if mode_norm in ("forward", "forward_only", "start_to_goal"):
            return self._plan_collision_aware_one_way(
                checker,
                q_start,
                q_goal,
                step_size_q,
                max_steps,
                clearance_margin_m=clearance_margin_m,
                max_local_candidates=max_local_candidates,
                search_label="forward_only",
                allow_direct_edge=allow_direct_edge,
            )
        if mode_norm not in ("bidirectional", "bi", "merge"):
            raise ValueError(f"Unsupported collision-aware planner mode: {mode}")

        probe_steps = int(max_steps)
        path_cost_mode = (
            self.path_joint_edge_weight != 0.0
            or self.path_tool_edge_weight != 0.0
            or self.path_tool_goal_weight != 0.0
        )
        if path_cost_mode and int(max_steps) > 30:
            probe_steps = max(20, int(round(float(max_steps) * float(self.bidirectional_forward_probe_fraction))))
        path = self._plan_collision_aware_one_way(
            checker,
            q_start,
            q_goal,
            step_size_q,
            probe_steps,
            clearance_margin_m=clearance_margin_m,
            max_local_candidates=max_local_candidates,
            search_label="forward",
            allow_direct_edge=allow_direct_edge,
        )
        forward_probe_debug = dict(self.last_debug)
        if np.asarray(path).ndim == 2 and len(path) > 0:
            raw_waypoints = int(len(path))
            path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
            forward_post_debug = {
                key: value
                for key, value in self.last_debug.items()
                if str(key).startswith("cartesian_shortcut")
            }
            selected_path = path
            reverse_comp_debug: dict[str, object] = {"enabled": False}
            if self.cartesian_shortcut_enabled and self.cartesian_shortcut_try_reverse:
                reverse_comp_t0 = time.perf_counter()
                reverse_raw = self._plan_collision_aware_one_way(
                    checker,
                    q_goal,
                    q_start,
                    step_size_q,
                    max_steps,
                    clearance_margin_m=clearance_margin_m,
                    max_local_candidates=max_local_candidates,
                    search_label="reverse_compare",
                    allow_direct_edge=allow_direct_edge,
                )
                reverse_debug = dict(self.last_debug)
                reverse_comp_debug = {
                    "enabled": True,
                    "status": reverse_debug.get("status", ""),
                    "raw_waypoints": int(len(reverse_raw)) if np.asarray(reverse_raw).ndim == 2 else 0,
                }
                if np.asarray(reverse_raw).ndim == 2 and len(reverse_raw) > 0:
                    reverse_candidate = np.asarray(reverse_raw[::-1], dtype=np.float32)
                    reverse_candidate = self._postprocess_collision_path(
                        checker, reverse_candidate, clearance_margin_m, step_size_q, shortcut_path
                    )
                    forward_cost = self._cartesian_path_quality_cost(path)
                    reverse_cost = self._cartesian_path_quality_cost(reverse_candidate)
                    improvement = 0.0
                    if np.isfinite(forward_cost) and np.isfinite(reverse_cost):
                        improvement = (forward_cost - reverse_cost) / max(1.0e-6, abs(forward_cost))
                    reverse_comp_debug.update(
                        {
                            "candidate_waypoints": int(len(reverse_candidate)),
                            "forward_cost": float(forward_cost),
                            "reverse_cost": float(reverse_cost),
                            "improvement": float(improvement),
                        }
                    )
                    if reverse_cost < forward_cost and improvement >= float(self.cartesian_shortcut_min_improvement):
                        selected_path = reverse_candidate
                        reverse_comp_debug["accepted"] = True
                    else:
                        reverse_comp_debug["accepted"] = False
                else:
                    reverse_comp_debug["accepted"] = False
                reverse_comp_debug["ms"] = (time.perf_counter() - reverse_comp_t0) * 1e3
                self.last_debug = dict(forward_probe_debug)
                self.last_debug.update(forward_post_debug)
            path = selected_path
            self.last_debug["search_direction"] = "forward"
            self.last_debug["forward_probe_steps"] = int(probe_steps)
            self.last_debug["raw_waypoints"] = raw_waypoints
            self.last_debug["shortcut_waypoints"] = int(len(path))
            self.last_debug["shortcut_enabled"] = bool(shortcut_path)
            self.last_debug["cartesian_reverse_compare"] = reverse_comp_debug
            return path

        reverse_path = self._plan_collision_aware_one_way(
            checker,
            q_goal,
            q_start,
            step_size_q,
            max_steps,
            clearance_margin_m=clearance_margin_m,
            max_local_candidates=max_local_candidates,
            search_label="reverse",
            allow_direct_edge=allow_direct_edge,
        )
        reverse_debug = dict(self.last_debug)
        if np.asarray(reverse_path).ndim == 2 and len(reverse_path) > 0:
            path = np.asarray(reverse_path[::-1], dtype=np.float32)
            path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
            post_debug = {
                key: value
                for key, value in self.last_debug.items()
                if str(key).startswith("cartesian_shortcut")
            }
            self.last_debug = {
                "status": "reverse_reached",
                "search_direction": "reverse",
                "forward_debug": forward_probe_debug,
                "reverse_debug": reverse_debug,
                "steps": reverse_debug.get("steps", 0),
                "start_goal_dist": forward_probe_debug.get("start_goal_dist", reverse_debug.get("start_goal_dist", -1.0)),
                "last_goal_dist": 0.0,
                "valid_edge_count": int(forward_probe_debug.get("valid_edge_count", 0)) + int(reverse_debug.get("valid_edge_count", 0)),
                "candidate_count": int(forward_probe_debug.get("candidate_count", 0)) + int(reverse_debug.get("candidate_count", 0)),
                "best_candidate_goal_dist": min(
                    float(forward_probe_debug.get("best_candidate_goal_dist", float("inf"))),
                    float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
                ),
                "best_edge_min_clearance": max(
                    float(forward_probe_debug.get("best_edge_min_clearance", -1.0)),
                    float(reverse_debug.get("best_edge_min_clearance", -1.0)),
                ),
                "raw_waypoints": int(len(reverse_path)),
                "forward_probe_steps": int(probe_steps),
                "shortcut_waypoints": int(len(path)),
                "shortcut_enabled": bool(shortcut_path),
                **post_debug,
            }
            return path

        forward_debug = forward_probe_debug
        if probe_steps < int(max_steps):
            path = self._plan_collision_aware_one_way(
                checker,
                q_start,
                q_goal,
                step_size_q,
                max_steps,
                clearance_margin_m=clearance_margin_m,
                max_local_candidates=max_local_candidates,
                search_label="forward_full",
                allow_direct_edge=allow_direct_edge,
            )
            forward_debug = dict(self.last_debug)
            if np.asarray(path).ndim == 2 and len(path) > 0:
                raw_waypoints = int(len(path))
                path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
                self.last_debug["search_direction"] = "forward_full"
                self.last_debug["forward_probe_debug"] = forward_probe_debug
                self.last_debug["reverse_debug"] = reverse_debug
                self.last_debug["raw_waypoints"] = raw_waypoints
                self.last_debug["shortcut_waypoints"] = int(len(path))
                self.last_debug["shortcut_enabled"] = bool(shortcut_path)
                return path

        self.last_debug = {
            "status": "failed_bidirectional_field_search",
            "search_direction": "bidirectional",
            "forward_probe_debug": forward_probe_debug,
            "forward_debug": forward_debug,
            "reverse_debug": reverse_debug,
            "steps": int(forward_debug.get("steps", 0)) + int(reverse_debug.get("steps", 0)),
            "start_goal_dist": forward_debug.get("start_goal_dist", reverse_debug.get("start_goal_dist", -1.0)),
            "last_goal_dist": min(
                float(forward_debug.get("last_goal_dist", float("inf"))),
                float(reverse_debug.get("last_goal_dist", float("inf"))),
            ),
            "valid_edge_count": int(forward_debug.get("valid_edge_count", 0)) + int(reverse_debug.get("valid_edge_count", 0)),
            "candidate_count": int(forward_debug.get("candidate_count", 0)) + int(reverse_debug.get("candidate_count", 0)),
            "best_candidate_goal_dist": min(
                float(forward_debug.get("best_candidate_goal_dist", float("inf"))),
                float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
            ),
            "best_edge_min_clearance": max(
                float(forward_debug.get("best_edge_min_clearance", -1.0)),
                float(reverse_debug.get("best_edge_min_clearance", -1.0)),
            ),
        }
        experimental_enabled = (
            self.path_joint_edge_weight != 0.0
            or self.path_tool_edge_weight != 0.0
            or self.path_tool_goal_weight != 0.0
            or self.cartesian_candidate_count > 0
        )
        if experimental_enabled:
            failed_debug = dict(self.last_debug)
            saved_weights = (
                self.path_joint_edge_weight,
                self.path_tool_edge_weight,
                self.path_tool_goal_weight,
                self.cartesian_candidate_count,
            )
            try:
                self.path_joint_edge_weight = 0.0
                self.path_tool_edge_weight = 0.0
                self.path_tool_goal_weight = 0.0
                self.cartesian_candidate_count = 0
                fallback = self.plan_collision_aware(
                    checker,
                    q_start,
                    q_goal,
                    step_size_q,
                    max_steps,
                    clearance_margin_m=clearance_margin_m,
                    max_local_candidates=max_local_candidates,
                    allow_direct_edge=allow_direct_edge,
                    shortcut_path=shortcut_path,
                )
                fallback_debug = dict(self.last_debug)
            finally:
                (
                    self.path_joint_edge_weight,
                    self.path_tool_edge_weight,
                    self.path_tool_goal_weight,
                    self.cartesian_candidate_count,
                ) = saved_weights
            if np.asarray(fallback).ndim == 2 and len(fallback) > 0:
                self.last_debug = dict(fallback_debug)
                self.last_debug["status"] = "experimental_fallback_reached"
                self.last_debug["experimental_failed_debug"] = failed_debug
                self.last_debug["experimental_fallback_used"] = True
                return np.asarray(fallback, dtype=np.float32)
            self.last_debug = failed_debug
            self.last_debug["experimental_fallback_debug"] = fallback_debug
            self.last_debug["experimental_fallback_used"] = False
        return np.zeros((0, 6), dtype=np.float32)

    def plan_learned_speed_search(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        min_predicted_speed: float = 0.20,
        max_local_candidates: int = 32,
        allow_direct_edge: bool = False,
        mode: str = "bidirectional",
        time_budget_ms: float = 90.0,
    ) -> np.ndarray:
        """Search using neural speed as the only edge-validity signal.

        This performs no point-cloud, SDF, geometry, or robot collision query.
        Geometric validation remains an offline evaluation concern.
        """
        mode_norm = str(mode).strip().lower()
        threshold = float(np.clip(min_predicted_speed, 0.0, 1.0))
        deadline_at = time.perf_counter() + max(1.0, float(time_budget_ms)) * 1.0e-3

        bridge_checker = _LearnedSpeedChecker(
            self.field_model, self.kinematics, q_goal, deadline_at=deadline_at
        )
        bridge = self._learned_replay_bridge(
            bridge_checker,
            q_start,
            q_goal,
            threshold=threshold,
            step_size_q=step_size_q,
            max_candidates=min(64, max(8, 2 * int(max_local_candidates))),
        )
        if len(bridge):
            self.last_debug = {
                "status": "learned_replay_bridge",
                "planner": "learned_speed_search",
                "steps": 0,
                "bridge_waypoints": int(len(bridge)),
                "geometric_collision_queries": 0,
                "learned_speed_query_calls": int(bridge_checker.query_calls),
                "learned_speed_query_states": int(bridge_checker.query_states),
                "learned_unsafe_rejections": int(bridge_checker.rejected_unsafe),
                "learned_coverage_rejections": int(bridge_checker.rejected_coverage),
                "time_budget_ms": float(time_budget_ms),
            }
            return bridge

        def _run(direction: str) -> tuple[np.ndarray, dict[str, object], _LearnedSpeedChecker]:
            reverse = direction == "goal_to_start"
            target = q_start if reverse else q_goal
            learned = _LearnedSpeedChecker(
                self.field_model, self.kinematics, target, deadline_at=deadline_at
            )
            path = self.plan_collision_aware(
                learned,
                q_start,
                q_goal,
                step_size_q,
                max_steps,
                clearance_margin_m=threshold,
                max_local_candidates=max_local_candidates,
                allow_direct_edge=allow_direct_edge,
                shortcut_path=False,
                mode="goal_to_start" if reverse else "forward",
            )
            debug = dict(self.last_debug)
            debug.update(
                {
                    "planner": "learned_speed_search",
                    "geometric_collision_queries": 0,
                    "learned_speed_query_calls": int(learned.query_calls),
                    "learned_speed_query_states": int(learned.query_states),
                    "learned_speed_requested_states": int(learned.requested_states),
                    "learned_speed_cache_hits": int(learned.requested_states - learned.query_states),
                    "min_predicted_speed": threshold,
                    "learned_unsafe_rejections": int(learned.rejected_unsafe),
                    "learned_coverage_rejections": int(learned.rejected_coverage),
                    "learned_deadline_rejections": int(learned.deadline_rejections),
                    "time_budget_ms": float(time_budget_ms),
                }
            )
            return path, debug, learned

        # A binary speed threshold alone makes every edge above the threshold
        # equally attractive and degenerates to a nearly straight goal search.
        # Add a learned travel-risk cost for this planner only.  It still makes
        # zero geometric queries: ``edge_mins`` are dense network speed queries.
        saved_costs = (
            self.path_joint_edge_weight,
            self.path_clearance_penalty_weight,
            self.path_clearance_soft_margin_m,
            self.path_cost_return_first_goal,
        )
        try:
            if self.path_joint_edge_weight == 0.0:
                self.path_joint_edge_weight = 0.15
            if self.path_clearance_penalty_weight == 0.0:
                self.path_clearance_penalty_weight = 20.0
                self.path_clearance_soft_margin_m = max(self.path_clearance_soft_margin_m, 0.18)
            self.path_cost_return_first_goal = False
            if mode_norm in ("goal_to_start", "reverse_only", "goal_to_start_only"):
                path, debug, _learned = _run("goal_to_start")
                self.last_debug = debug
                return path
            if mode_norm in ("forward", "forward_only", "start_to_goal"):
                path, debug, _learned = _run("forward")
                self.last_debug = debug
                return path
            if mode_norm not in ("bidirectional", "bi", "merge"):
                raise ValueError(f"Unsupported learned-speed search mode: {mode}")
            forward, forward_debug, _forward_checker = _run("forward")
            if np.asarray(forward).ndim == 2 and len(forward) > 0:
                self.last_debug = forward_debug
                self.last_debug["search_direction"] = "forward"
                return forward
            if time.perf_counter() >= deadline_at:
                self.last_debug = forward_debug
                self.last_debug["status"] = "learned_time_budget"
                self.last_debug["search_direction"] = "forward_timeout"
                return np.zeros((0, 6), dtype=np.float32)
            reverse, reverse_debug, _reverse_checker = _run("goal_to_start")
        finally:
            (
                self.path_joint_edge_weight,
                self.path_clearance_penalty_weight,
                self.path_clearance_soft_margin_m,
                self.path_cost_return_first_goal,
            ) = saved_costs
        self.last_debug = {
            **reverse_debug,
            "search_direction": "goal_to_start" if len(reverse) else "failed_bidirectional",
            "forward_debug": forward_debug,
        }
        return reverse

    def plan_sampled_rollout(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        clearance_margin_m: float = 0.01,
        sample_count: int = 256,
        horizon: int = 8,
        iterations: int = 2,
        noise_scale: float = 1.0,
        temperature: float = 35.0,
        tau_weight: float = 0.55,
        goal_dist_weight: float = 3.0,
        joint_path_weight: float = 0.20,
        tool_path_weight: float = 1.0,
        clearance_penalty_weight: float = 8.0,
        topk_edge_checks: int = 64,
        allow_direct_edge: bool = True,
        shortcut_path: bool = True,
    ) -> np.ndarray:
        q_start = self.kinematics.clamp(np.asarray(q_start, dtype=np.float64).reshape(6))
        q_goal = self.kinematics.clamp(np.asarray(q_goal, dtype=np.float64).reshape(6))
        q_cur_n = self.kinematics.normalize(q_start).astype(np.float32)
        q_goal_n = self.kinematics.normalize(q_goal).astype(np.float32)
        tol = max(0.01, 0.5 * float(step_size_q))
        step_size_q = float(max(1.0e-4, step_size_q))
        max_steps = max(1, int(max_steps))
        sample_count = max(8, int(sample_count))
        horizon = max(2, int(horizon))
        iterations = max(1, int(iterations))
        topk_edge_checks = max(1, min(int(topk_edge_checks), sample_count))

        start_goal_dist = float(np.linalg.norm(q_cur_n - q_goal_n))
        start_clearance = float(checker.clearance_batch(np.asarray([q_start], dtype=np.float32))[0])
        self.last_debug = {
            "status": "started",
            "planner": "sampled_rollout",
            "steps": 0,
            "start_goal_dist": start_goal_dist,
            "last_goal_dist": start_goal_dist,
            "start_clearance": start_clearance,
            "sample_count": sample_count,
            "horizon": horizon,
            "iterations": iterations,
            "rollout_samples": 0,
            "rollout_states": 0,
            "direct_edge_min_clearance": -1.0,
            "direct_edge_valid": False,
            "topk_edge_checks": topk_edge_checks,
        }
        if start_goal_dist <= tol:
            self.last_debug.update({"status": "already_at_goal", "last_goal_dist": 0.0})
            return np.asarray([q_start, q_goal], dtype=np.float32)

        direct_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], step_size_q)
        if direct_edge.size > 0:
            self.last_debug["direct_edge_min_clearance"] = float(direct_edge[0])
            self.last_debug["direct_edge_valid"] = bool(float(direct_edge[0]) >= float(clearance_margin_m))
        if allow_direct_edge and direct_edge.size > 0 and float(direct_edge[0]) >= float(clearance_margin_m):
            self.last_debug.update(
                {"status": "direct_edge", "last_goal_dist": 0.0, "steps": 0, "shortcut_enabled": bool(shortcut_path)}
            )
            return np.asarray([q_start, q_goal], dtype=np.float32)

        path_n = [q_cur_n.copy()]
        failed_edge_selections = 0

        for step_idx in range(max_steps):
            goal_dist = float(np.linalg.norm(q_cur_n - q_goal_n))
            if goal_dist <= tol:
                break

            prior = self._goal_directed_prior_controls(q_cur_n, q_goal_n, step_size_q, horizon)
            best_cost: np.ndarray | None = None
            best_rollouts: np.ndarray | None = None

            for _ in range(iterations):
                controls = self._sample_rollout_controls(prior, sample_count, step_size_q, float(noise_scale))
                rollouts = self._integrate_normalized_controls(q_cur_n, controls)
                cost = self._sampled_rollout_costs(
                    checker,
                    q_cur_n,
                    rollouts,
                    q_goal_n,
                    clearance_margin_m=float(clearance_margin_m),
                    tau_weight=float(tau_weight),
                    goal_dist_weight=float(goal_dist_weight),
                    joint_path_weight=float(joint_path_weight),
                    tool_path_weight=float(tool_path_weight),
                    clearance_penalty_weight=float(clearance_penalty_weight),
                )
                self._inc_debug("rollout_samples", sample_count)
                self._inc_debug("rollout_states", sample_count * horizon)
                if best_cost is None or float(np.nanmin(cost)) < float(np.nanmin(best_cost)):
                    best_cost = cost
                    best_rollouts = rollouts
                prior = self._weighted_control_update(controls, cost, float(temperature), fallback=prior)

            if best_cost is None or best_rollouts is None:
                self.last_debug.update({"status": "no_rollouts", "steps": int(step_idx), "last_goal_dist": goal_dist})
                break

            finite_cost = np.where(np.isfinite(best_cost), best_cost, np.inf)
            order = np.argsort(finite_cost)
            chosen_next = None
            checked = 0
            progress_candidates = 0
            for cand_idx in order:
                q_next_n = best_rollouts[int(cand_idx), 0].astype(np.float32)
                next_goal_dist = float(np.linalg.norm(q_next_n - q_goal_n))
                # MPPI noise can otherwise pick safe but sideways first steps forever.
                if next_goal_dist > goal_dist + max(0.005, 0.25 * step_size_q):
                    continue
                progress_candidates += 1
                if checked >= topk_edge_checks:
                    continue
                edge = self._edge_min_clearances(checker, q_cur_n, q_next_n[None, :], step_size_q)
                checked += 1
                if edge.size > 0 and float(edge[0]) >= float(clearance_margin_m):
                    chosen_next = q_next_n
                    break
            self._inc_debug("sampled_topk_edge_checks_done", checked)
            self._inc_debug("sampled_progress_candidates", progress_candidates)
            if chosen_next is None:
                failed_edge_selections += 1
                self.last_debug.update(
                    {
                        "status": "no_valid_sampled_step",
                        "steps": int(step_idx),
                        "last_goal_dist": goal_dist,
                        "failed_edge_selections": int(failed_edge_selections),
                    }
                )
                return np.zeros((0, 6), dtype=np.float32)

            q_cur_n = chosen_next
            path_n.append(q_cur_n.copy())
            self.last_debug["last_goal_dist"] = float(np.linalg.norm(q_cur_n - q_goal_n))
            self.last_debug["steps"] = int(step_idx + 1)

            if float(self.last_debug["last_goal_dist"]) <= max(tol, 1.25 * step_size_q):
                final_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], step_size_q)
                if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                    path_n.append(q_goal_n.copy())
                    self.last_debug.update({"status": "reached", "last_goal_dist": 0.0})
                    break
        else:
            self.last_debug.update({"status": "max_steps", "last_goal_dist": float(np.linalg.norm(q_cur_n - q_goal_n))})

        self.last_debug["failed_edge_selections"] = int(failed_edge_selections)
        if len(path_n) < 2:
            return np.zeros((0, 6), dtype=np.float32)
        if np.linalg.norm(path_n[-1] - q_goal_n) > max(tol, 1.25 * step_size_q):
            final_edge = self._edge_min_clearances(checker, path_n[-1], q_goal_n[None, :], step_size_q)
            if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                path_n.append(q_goal_n.copy())
                self.last_debug.update({"status": "reached", "last_goal_dist": 0.0})
        if np.linalg.norm(path_n[-1] - q_goal_n) > max(tol, 1.25 * step_size_q):
            self.last_debug["status"] = str(self.last_debug.get("status", "failed_to_reach"))
            return np.zeros((0, 6), dtype=np.float32)
        if str(self.last_debug.get("status", "")) in ("started", "max_steps"):
            self.last_debug.update({"status": "reached", "last_goal_dist": 0.0})
        path = np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)
        if shortcut_path and len(path) > 2:
            path = self._shortcut_collision_path(checker, path, float(clearance_margin_m), float(step_size_q))
        self.last_debug["shortcut_waypoints"] = int(len(path))
        self.last_debug["shortcut_enabled"] = bool(shortcut_path)
        return path

    def plan_cartesian_graph(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        clearance_margin_m: float = 0.01,
        max_local_candidates: int = 32,
        allow_direct_edge: bool = True,
        shortcut_path: bool = True,
        mode: str = "forward",
        tool_edge_weight: float = 1.0,
        tool_goal_weight: float = 2.0,
        joint_edge_weight: float = 0.05,
        joint_goal_weight: float = 0.25,
        tau_weight: float = 0.0,
        clearance_penalty_weight: float = 0.0,
        clearance_soft_margin_m: float = 0.04,
    ) -> np.ndarray:
        mode_norm = str(mode).strip().lower()
        if mode_norm in ("goal_to_start", "reverse_only", "goal_to_start_only"):
            reverse_path = self._plan_cartesian_graph_one_way(
                checker,
                q_goal,
                q_start,
                step_size_q,
                max_steps,
                clearance_margin_m=clearance_margin_m,
                max_local_candidates=max_local_candidates,
                allow_direct_edge=allow_direct_edge,
                search_label="cartesian_graph_goal_to_start",
                tool_edge_weight=tool_edge_weight,
                tool_goal_weight=tool_goal_weight,
                joint_edge_weight=joint_edge_weight,
                joint_goal_weight=joint_goal_weight,
                tau_weight=tau_weight,
                clearance_penalty_weight=clearance_penalty_weight,
                clearance_soft_margin_m=clearance_soft_margin_m,
            )
            reverse_debug = dict(self.last_debug)
            if np.asarray(reverse_path).ndim == 2 and len(reverse_path) > 0:
                path = np.asarray(reverse_path[::-1], dtype=np.float32)
                raw_waypoints = int(len(path))
                path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
                post_debug = {
                    key: value
                    for key, value in self.last_debug.items()
                    if str(key).startswith("cartesian_shortcut")
                }
                self.last_debug = {
                    "status": "cartesian_graph_goal_to_start_reached",
                    "planner": "cartesian_graph",
                    "search_direction": "goal_to_start",
                    "reverse_debug": reverse_debug,
                    "steps": int(reverse_debug.get("steps", 0)),
                    "start_goal_dist": reverse_debug.get("start_goal_dist", -1.0),
                    "last_goal_dist": 0.0,
                    "last_tool_goal_dist_m": 0.0,
                    "valid_edge_count": int(reverse_debug.get("valid_edge_count", 0)),
                    "candidate_count": int(reverse_debug.get("candidate_count", 0)),
                    "best_candidate_goal_dist": float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
                    "best_edge_min_clearance": float(reverse_debug.get("best_edge_min_clearance", -1.0)),
                    "raw_waypoints": raw_waypoints,
                    "shortcut_waypoints": int(len(path)),
                    "shortcut_enabled": bool(shortcut_path),
                    **post_debug,
                }
                return path
            self.last_debug = {
                "status": "failed_cartesian_graph_goal_to_start",
                "planner": "cartesian_graph",
                "search_direction": "goal_to_start",
                "reverse_debug": reverse_debug,
                "steps": int(reverse_debug.get("steps", 0)),
                "start_goal_dist": reverse_debug.get("start_goal_dist", -1.0),
                "last_goal_dist": reverse_debug.get("last_goal_dist", float("inf")),
                "valid_edge_count": int(reverse_debug.get("valid_edge_count", 0)),
                "candidate_count": int(reverse_debug.get("candidate_count", 0)),
                "best_candidate_goal_dist": float(reverse_debug.get("best_candidate_goal_dist", float("inf"))),
                "best_edge_min_clearance": float(reverse_debug.get("best_edge_min_clearance", -1.0)),
            }
            return np.zeros((0, 6), dtype=np.float32)

        path = self._plan_cartesian_graph_one_way(
            checker,
            q_start,
            q_goal,
            step_size_q,
            max_steps,
            clearance_margin_m=clearance_margin_m,
            max_local_candidates=max_local_candidates,
            allow_direct_edge=allow_direct_edge,
            search_label="cartesian_graph_forward",
            tool_edge_weight=tool_edge_weight,
            tool_goal_weight=tool_goal_weight,
            joint_edge_weight=joint_edge_weight,
            joint_goal_weight=joint_goal_weight,
            tau_weight=tau_weight,
            clearance_penalty_weight=clearance_penalty_weight,
            clearance_soft_margin_m=clearance_soft_margin_m,
        )
        raw_debug = dict(self.last_debug)
        if np.asarray(path).ndim == 2 and len(path) > 0:
            raw_waypoints = int(len(path))
            path = self._postprocess_collision_path(checker, path, clearance_margin_m, step_size_q, shortcut_path)
            raw_debug.update(
                {
                    "raw_waypoints": raw_waypoints,
                    "shortcut_waypoints": int(len(path)),
                    "shortcut_enabled": bool(shortcut_path),
                }
            )
            raw_debug.update(
                {
                    key: value
                    for key, value in self.last_debug.items()
                    if str(key).startswith("cartesian_shortcut")
                }
            )
            self.last_debug = raw_debug
        return path

    def _plan_cartesian_graph_one_way(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        clearance_margin_m: float,
        max_local_candidates: int,
        allow_direct_edge: bool,
        search_label: str,
        tool_edge_weight: float,
        tool_goal_weight: float,
        joint_edge_weight: float,
        joint_goal_weight: float,
        tau_weight: float,
        clearance_penalty_weight: float,
        clearance_soft_margin_m: float,
    ) -> np.ndarray:
        q_start_n = self.kinematics.normalize(q_start).astype(np.float32)
        q_goal_n = self.kinematics.normalize(q_goal).astype(np.float32)
        tol = max(0.01, 0.5 * float(step_size_q))
        start_goal_dist = float(np.linalg.norm(q_start_n - q_goal_n))
        start_tool = np.asarray(self.kinematics.fk(q_start)[:3, 3], dtype=np.float32)
        goal_tool = np.asarray(self.kinematics.fk(q_goal)[:3, 3], dtype=np.float32)
        start_tool_goal_dist = float(np.linalg.norm(start_tool - goal_tool))
        start_clearance = float(checker.clearance_batch(np.asarray([q_start], dtype=np.float32))[0])
        self.last_debug = {
            "status": "started",
            "planner": "cartesian_graph",
            "steps": 0,
            "start_goal_dist": start_goal_dist,
            "start_tool_goal_dist_m": start_tool_goal_dist,
            "last_goal_dist": start_goal_dist,
            "last_tool_goal_dist_m": start_tool_goal_dist,
            "start_clearance": start_clearance,
            "valid_edge_count": 0,
            "candidate_count": 0,
            "best_candidate_clearance": -1.0,
            "best_edge_min_clearance": -1.0,
            "best_candidate_goal_dist": start_goal_dist,
            "best_candidate_tool_goal_dist_m": start_tool_goal_dist,
            "expanded_nodes": 0,
            "queued_nodes": 1,
            "search_direction": search_label,
            "direct_edge_min_clearance": -1.0,
            "direct_edge_valid": False,
            "first_expansion_top_candidates": [],
            "cartesian_graph_tool_edge_weight": float(tool_edge_weight),
            "cartesian_graph_tool_goal_weight": float(tool_goal_weight),
            "cartesian_graph_joint_edge_weight": float(joint_edge_weight),
            "cartesian_graph_joint_goal_weight": float(joint_goal_weight),
            "cartesian_graph_tau_weight": float(tau_weight),
        }

        if start_goal_dist <= tol:
            self.last_debug.update({"status": "already_at_goal", "steps": 0, "last_goal_dist": 0.0})
            return np.asarray([q_start, q_goal], dtype=np.float32)

        direct_edge = self._edge_min_clearances(checker, q_start_n, q_goal_n[None, :], float(step_size_q))
        if direct_edge.size > 0:
            self.last_debug["direct_edge_min_clearance"] = float(direct_edge[0])
            self.last_debug["direct_edge_valid"] = bool(float(direct_edge[0]) >= float(clearance_margin_m))
        if allow_direct_edge and direct_edge.size > 0 and float(direct_edge[0]) >= float(clearance_margin_m):
            self.last_debug.update(
                {
                    "status": "cartesian_graph_direct_edge",
                    "steps": 0,
                    "last_goal_dist": 0.0,
                    "last_tool_goal_dist_m": 0.0,
                    "valid_edge_count": 1,
                    "best_edge_min_clearance": float(direct_edge[0]),
                }
            )
            return np.asarray([q_start, q_goal], dtype=np.float32)

        max_expansions = max(1, int(max_steps))
        max_candidates = max(4, int(max_local_candidates))
        max_nodes = max(max_candidates + 1, max_expansions * max_candidates)
        nodes: list[np.ndarray] = [q_start_n.copy()]
        parents: list[int] = [-1]
        depths: list[int] = [0]
        costs: list[float] = [0.0]
        tools: list[np.ndarray] = [start_tool.copy()]
        closed: set[int] = set()
        best_seen: dict[tuple[int, ...], float] = {self._search_key(q_start_n, step_size_q): 0.0}
        heap: list[tuple[float, int, int]] = [
            (
                self._cartesian_graph_heuristic(
                    q_start_n, start_tool, q_goal_n, goal_tool, joint_goal_weight, tool_goal_weight, tau_weight
                ),
                0,
                0,
            )
        ]
        push_count = 1
        best_goal_dist = start_goal_dist
        best_tool_goal_dist = start_tool_goal_dist
        best_idx = 0

        for expansion_idx in range(max_expansions):
            if bool(getattr(checker, "expired", False)):
                self.last_debug.update(
                    {
                        "status": "learned_time_budget",
                        "steps": int(expansion_idx),
                        "last_goal_dist": best_goal_dist,
                    }
                )
                return np.zeros((0, 6), dtype=np.float32)
            if not heap:
                self.last_debug.update({"status": "cartesian_graph_frontier_empty", "steps": int(expansion_idx)})
                break
            _, _, node_idx = heapq.heappop(heap)
            if node_idx in closed:
                continue
            closed.add(node_idx)
            q_cur_n = nodes[node_idx]
            q_cur_tool = tools[node_idx]
            node_goal_dist = float(np.linalg.norm(q_cur_n - q_goal_n))
            node_tool_goal_dist = float(np.linalg.norm(q_cur_tool - goal_tool))
            if node_goal_dist < best_goal_dist:
                best_goal_dist = node_goal_dist
                best_idx = node_idx
            best_tool_goal_dist = min(best_tool_goal_dist, node_tool_goal_dist)
            self.last_debug["expanded_nodes"] = int(self.last_debug.get("expanded_nodes", 0)) + 1

            if (allow_direct_edge or node_idx != 0) and node_goal_dist < start_goal_dist - 1.0e-6:
                final_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], float(step_size_q))
                if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                    goal_cost = float(costs[node_idx]) + self._cartesian_graph_edge_cost(
                        q_cur_n,
                        q_cur_tool,
                        q_goal_n,
                        goal_tool,
                        float(final_edge[0]),
                        clearance_margin_m,
                        tool_edge_weight,
                        joint_edge_weight,
                        clearance_penalty_weight,
                        clearance_soft_margin_m,
                    )
                    goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), node_idx)
                    costs.append(goal_cost)
                    tools.append(goal_tool.copy())
                    self.last_debug.update(
                        {
                            "status": "cartesian_graph_goal_edge_reached",
                            "steps": int(expansion_idx + 1),
                            "last_goal_dist": 0.0,
                            "last_tool_goal_dist_m": 0.0,
                            "best_edge_min_clearance": max(
                                float(self.last_debug.get("best_edge_min_clearance", -1.0)), float(final_edge[0])
                            ),
                            "goal_edge_min_clearance": float(final_edge[0]),
                            "best_goal_path_cost": goal_cost,
                        }
                    )
                    return self._search_path_to_array(nodes, parents, goal_idx)

            grad_np = self._field_gradient(q_cur_n, q_goal_n)
            candidates = self._local_step_candidates(q_cur_n, q_goal_n, grad_np, float(step_size_q), max_candidates)
            candidates = self._append_cartesian_step_candidates(candidates, q_cur_n, q_goal_n, float(step_size_q))
            if len(candidates) == 0:
                continue
            edge_mins, clearances = self._edge_min_clearances(
                checker, q_cur_n, candidates, float(step_size_q), return_endpoints=True
            )
            if bool(getattr(checker, "expired", False)):
                self.last_debug.update(
                    {
                        "status": "learned_time_budget",
                        "steps": int(expansion_idx + 1),
                        "last_goal_dist": best_goal_dist,
                    }
                )
                return np.zeros((0, 6), dtype=np.float32)
            edge_ok = edge_mins >= float(clearance_margin_m)
            q_phys = np.asarray([self.kinematics.denormalize(qn) for qn in candidates], dtype=np.float32)
            tool_t0 = time.perf_counter()
            cand_tools = np.asarray([self.kinematics.fk(q)[:3, 3] for q in q_phys], dtype=np.float32)
            self._add_debug_ms("cartesian_graph_candidate_fk_ms", tool_t0)
            self._inc_debug("cartesian_graph_candidate_fk_states", len(q_phys))
            goal_dists_all = np.linalg.norm(candidates - q_goal_n[None, :], axis=1)
            tool_goal_dists_all = np.linalg.norm(cand_tools - goal_tool[None, :], axis=1)
            self.last_debug["candidate_count"] = int(self.last_debug.get("candidate_count", 0)) + int(len(candidates))
            self.last_debug["valid_edge_count"] = int(self.last_debug.get("valid_edge_count", 0)) + int(np.count_nonzero(edge_ok))
            if len(clearances):
                self.last_debug["best_candidate_clearance"] = max(
                    float(self.last_debug.get("best_candidate_clearance", -1.0)), float(np.nanmax(clearances))
                )
            if len(edge_mins):
                self.last_debug["best_edge_min_clearance"] = max(
                    float(self.last_debug.get("best_edge_min_clearance", -1.0)), float(np.nanmax(edge_mins))
                )
            if len(goal_dists_all):
                self.last_debug["best_candidate_goal_dist"] = min(
                    float(self.last_debug.get("best_candidate_goal_dist", float("inf"))), float(np.nanmin(goal_dists_all))
                )
            if len(tool_goal_dists_all):
                self.last_debug["best_candidate_tool_goal_dist_m"] = min(
                    float(self.last_debug.get("best_candidate_tool_goal_dist_m", float("inf"))),
                    float(np.nanmin(tool_goal_dists_all)),
                )
            if not np.any(edge_ok):
                continue

            valid_idx = np.flatnonzero(edge_ok)
            valid_candidates = candidates[valid_idx]
            valid_tools = cand_tools[valid_idx]
            valid_clearances = clearances[valid_idx]
            valid_edge_mins = edge_mins[valid_idx]
            valid_goal_dists = goal_dists_all[valid_idx]
            valid_tool_goal_dists = tool_goal_dists_all[valid_idx]
            edge_costs = np.asarray(
                [
                    self._cartesian_graph_edge_cost(
                        q_cur_n,
                        q_cur_tool,
                        cand,
                        tool,
                        float(edge_min),
                        clearance_margin_m,
                        tool_edge_weight,
                        joint_edge_weight,
                        clearance_penalty_weight,
                        clearance_soft_margin_m,
                    )
                    for cand, tool, edge_min in zip(valid_candidates, valid_tools, valid_edge_mins)
                ],
                dtype=np.float32,
            )
            heuristics = np.asarray(
                [
                    self._cartesian_graph_heuristic(
                        cand, tool, q_goal_n, goal_tool, joint_goal_weight, tool_goal_weight, tau_weight
                    )
                    for cand, tool in zip(valid_candidates, valid_tools)
                ],
                dtype=np.float32,
            )
            candidate_g_costs = float(costs[node_idx]) + edge_costs
            valid_scores = candidate_g_costs + heuristics
            if expansion_idx == 0:
                top_idx = np.argsort(valid_scores)[: min(8, len(valid_scores))]
                self.last_debug["first_expansion_top_candidates"] = [
                    {
                        "score": float(valid_scores[int(i)]),
                        "g_cost": float(candidate_g_costs[int(i)]),
                        "heuristic": float(heuristics[int(i)]),
                        "edge_cost": float(edge_costs[int(i)]),
                        "goal_dist": float(valid_goal_dists[int(i)]),
                        "tool_goal_m": float(valid_tool_goal_dists[int(i)]),
                        "tool_step_m": float(np.linalg.norm(valid_tools[int(i)] - q_cur_tool)),
                        "clearance": float(valid_clearances[int(i)]),
                        "edge_min_clearance": float(valid_edge_mins[int(i)]),
                        "step_norm": float(np.linalg.norm(valid_candidates[int(i)] - q_cur_n)),
                    }
                    for i in top_idx
                ]

            for local_i in np.argsort(valid_scores):
                if len(nodes) >= max_nodes:
                    break
                cand = valid_candidates[int(local_i)].astype(np.float32)
                key = self._search_key(cand, step_size_q)
                child_cost = float(candidate_g_costs[int(local_i)])
                prev_cost = best_seen.get(key)
                if prev_cost is not None and child_cost >= prev_cost - 1.0e-6:
                    continue
                best_seen[key] = child_cost
                child_idx = self._append_search_node(nodes, parents, depths, cand, node_idx)
                costs.append(child_cost)
                tools.append(valid_tools[int(local_i)].astype(np.float32))
                push_count += 1
                heapq.heappush(heap, (float(valid_scores[int(local_i)]), push_count, child_idx))
                cand_goal_dist = float(valid_goal_dists[int(local_i)])
                cand_tool_goal_dist = float(valid_tool_goal_dists[int(local_i)])
                if cand_goal_dist < best_goal_dist:
                    best_goal_dist = cand_goal_dist
                    best_idx = child_idx
                best_tool_goal_dist = min(best_tool_goal_dist, cand_tool_goal_dist)
                if cand_goal_dist <= tol:
                    final_edge = self._edge_min_clearances(checker, cand, q_goal_n[None, :], float(step_size_q))
                    if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                        goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), child_idx)
                        costs.append(child_cost)
                        tools.append(goal_tool.copy())
                        self.last_debug.update(
                            {
                                "status": "cartesian_graph_reached",
                                "steps": int(expansion_idx + 1),
                                "last_goal_dist": 0.0,
                                "last_tool_goal_dist_m": 0.0,
                            }
                        )
                        return self._search_path_to_array(nodes, parents, goal_idx)
            self.last_debug["queued_nodes"] = int(len(heap))
            if len(nodes) >= max_nodes:
                self.last_debug.update({"status": "cartesian_graph_node_budget", "steps": int(expansion_idx + 1)})
                break
        else:
            self.last_debug.update({"status": "cartesian_graph_max_steps", "steps": int(max_steps)})

        self.last_debug["last_goal_dist"] = float(best_goal_dist)
        self.last_debug["last_tool_goal_dist_m"] = float(best_tool_goal_dist)
        self.last_debug["best_node_idx"] = int(best_idx)
        return np.zeros((0, 6), dtype=np.float32)

    def _cartesian_graph_edge_cost(
        self,
        q_from_n: np.ndarray,
        tool_from: np.ndarray,
        q_to_n: np.ndarray,
        tool_to: np.ndarray,
        edge_min_clearance: float,
        clearance_margin_m: float,
        tool_edge_weight: float,
        joint_edge_weight: float,
        clearance_penalty_weight: float,
        clearance_soft_margin_m: float,
    ) -> float:
        tool_edge = float(np.linalg.norm(np.asarray(tool_to, dtype=np.float32) - np.asarray(tool_from, dtype=np.float32)))
        joint_edge = float(np.linalg.norm(np.asarray(q_to_n, dtype=np.float32) - np.asarray(q_from_n, dtype=np.float32)))
        soft_threshold = float(clearance_margin_m) + float(clearance_soft_margin_m)
        soft_violation = max(0.0, soft_threshold - float(edge_min_clearance))
        return float(
            float(tool_edge_weight) * tool_edge
            + float(joint_edge_weight) * joint_edge
            + float(clearance_penalty_weight) * soft_violation * soft_violation
        )

    def _cartesian_graph_heuristic(
        self,
        q_n: np.ndarray,
        tool_xyz: np.ndarray,
        q_goal_n: np.ndarray,
        goal_tool_xyz: np.ndarray,
        joint_goal_weight: float,
        tool_goal_weight: float,
        tau_weight: float,
    ) -> float:
        tool_goal = float(np.linalg.norm(np.asarray(tool_xyz, dtype=np.float32) - np.asarray(goal_tool_xyz, dtype=np.float32)))
        joint_goal = float(np.linalg.norm(np.asarray(q_n, dtype=np.float32) - np.asarray(q_goal_n, dtype=np.float32)))
        tau_cost = 0.0
        if float(tau_weight) != 0.0:
            q_pair_goal = np.asarray(q_goal_n, dtype=np.float32)[None, :]
            q_pair = np.asarray(q_n, dtype=np.float32)[None, :]
            infer_t0 = time.perf_counter()
            tau_cost = float(self.field_model.predict_travel_times(q_pair, q_pair_goal)[0])
            self._add_debug_ms("cartesian_graph_tau_infer_ms", infer_t0)
            self._inc_debug("cartesian_graph_tau_infer_calls", 1)
            self._inc_debug("cartesian_graph_tau_infer_pairs", 1)
        return float(
            float(tool_goal_weight) * tool_goal
            + float(joint_goal_weight) * joint_goal
            + float(tau_weight) * tau_cost
        )

    def _plan_collision_aware_one_way(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        clearance_margin_m: float = 0.01,
        max_local_candidates: int = 32,
        search_label: str = "forward",
        allow_direct_edge: bool = True,
    ) -> np.ndarray:
        q_start_n = self.kinematics.normalize(q_start).astype(np.float32)
        q_goal_n = self.kinematics.normalize(q_goal).astype(np.float32)
        tol = max(0.01, 0.5 * float(step_size_q))
        start_goal_dist = float(np.linalg.norm(q_start_n - q_goal_n))
        start_clearance = float(checker.clearance_batch(np.asarray([q_start], dtype=np.float32))[0])
        self.last_debug = {
            "status": "started",
            "steps": 0,
            "start_goal_dist": start_goal_dist,
            "last_goal_dist": start_goal_dist,
            "start_clearance": start_clearance,
            "valid_edge_count": 0,
            "candidate_count": 0,
            "best_candidate_clearance": -1.0,
            "best_edge_min_clearance": -1.0,
            "best_candidate_goal_dist": start_goal_dist,
            "expanded_nodes": 0,
            "queued_nodes": 1,
            "search_direction": search_label,
            "direct_edge_min_clearance": -1.0,
            "direct_edge_valid": False,
            "first_expansion_top_candidates": [],
        }

        if start_goal_dist <= tol:
            self.last_debug.update({"status": "already_at_goal", "steps": 0, "last_goal_dist": 0.0})
            return np.asarray([q_start, q_goal], dtype=np.float32)

        direct_edge = self._edge_min_clearances(checker, q_start_n, q_goal_n[None, :], float(step_size_q))
        if direct_edge.size > 0:
            self.last_debug["direct_edge_min_clearance"] = float(direct_edge[0])
            self.last_debug["direct_edge_valid"] = bool(float(direct_edge[0]) >= float(clearance_margin_m))
        if allow_direct_edge and direct_edge.size > 0 and float(direct_edge[0]) >= float(clearance_margin_m):
            self.last_debug.update(
                {
                    "status": "direct_edge",
                    "steps": 0,
                    "last_goal_dist": 0.0,
                    "valid_edge_count": 1,
                    "best_edge_min_clearance": float(direct_edge[0]),
                }
            )
            return np.asarray([q_start, q_goal], dtype=np.float32)

        max_expansions = max(1, int(max_steps))
        max_candidates = max(4, int(max_local_candidates))
        max_nodes = max(max_candidates + 1, max_expansions * max_candidates)

        nodes: list[np.ndarray] = [q_start_n.copy()]
        parents: list[int] = [-1]
        depths: list[int] = [0]
        costs: list[float] = [0.0]
        is_goal_node: list[bool] = [False]
        closed: set[int] = set()
        best_seen: dict[tuple[int, ...], float] = {self._search_key(q_start_n, step_size_q): 0.0}
        heap: list[tuple[float, int, int]] = [(self._field_search_priority(q_start_n, q_goal_n, 0, start_clearance, clearance_margin_m), 0, 0)]
        push_count = 1
        best_goal_dist = start_goal_dist
        best_idx = 0
        path_cost_mode = (
            self.path_joint_edge_weight != 0.0
            or self.path_tool_edge_weight != 0.0
            or self.path_tool_goal_weight != 0.0
        )

        for expansion_idx in range(max_expansions):
            if bool(getattr(checker, "expired", False)):
                self.last_debug.update(
                    {
                        "status": "learned_time_budget",
                        "steps": int(expansion_idx),
                        "last_goal_dist": best_goal_dist,
                    }
                )
                return np.zeros((0, 6), dtype=np.float32)
            if not heap:
                self.last_debug.update({"status": "frontier_empty", "steps": int(expansion_idx), "last_goal_dist": best_goal_dist})
                break
            _, _, node_idx = heapq.heappop(heap)
            if node_idx in closed:
                continue
            closed.add(node_idx)
            q_cur_n = nodes[node_idx]
            if is_goal_node[node_idx]:
                self.last_debug.update({"status": "reached", "steps": int(expansion_idx), "last_goal_dist": 0.0})
                return self._search_path_to_array(nodes, parents, node_idx)
            node_goal_dist = float(np.linalg.norm(q_cur_n - q_goal_n))
            if node_goal_dist < best_goal_dist:
                best_goal_dist = node_goal_dist
                best_idx = node_idx
            self.last_debug["expanded_nodes"] = int(self.last_debug.get("expanded_nodes", 0)) + 1

            if (allow_direct_edge or node_idx != 0) and node_goal_dist < start_goal_dist - 1.0e-6:
                final_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], float(step_size_q))
                if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                    goal_edge_cost = float(
                        self._edge_path_costs(q_cur_n, q_goal_n[None, :], final_edge, clearance_margin_m)[0]
                    )
                    goal_cost = float(costs[node_idx]) + goal_edge_cost
                    goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), node_idx)
                    costs.append(goal_cost)
                    is_goal_node.append(True)
                    push_count += 1
                    heapq.heappush(heap, (goal_cost, push_count, goal_idx))
                    self.last_debug.update(
                        {
                            "status": "goal_edge_reached"
                            if path_cost_mode and self.path_cost_return_first_goal
                            else "goal_edge_queued",
                            "steps": int(expansion_idx + 1),
                            "last_goal_dist": min(float(self.last_debug.get("last_goal_dist", node_goal_dist)), node_goal_dist),
                            "best_edge_min_clearance": max(
                                float(self.last_debug.get("best_edge_min_clearance", -1.0)),
                                float(final_edge[0]),
                            ),
                            "goal_edge_min_clearance": float(final_edge[0]),
                            "best_goal_path_cost": min(
                                float(self.last_debug.get("best_goal_path_cost", float("inf"))), goal_cost
                            ),
                        }
                    )
                    if path_cost_mode and self.path_cost_return_first_goal:
                        self.last_debug["last_goal_dist"] = 0.0
                        return self._search_path_to_array(nodes, parents, goal_idx)

            if node_goal_dist <= max(tol, 1.25 * float(step_size_q)):
                final_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], float(step_size_q))
                if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                    goal_edge_cost = float(
                        self._edge_path_costs(q_cur_n, q_goal_n[None, :], final_edge, clearance_margin_m)[0]
                    )
                    goal_cost = float(costs[node_idx]) + goal_edge_cost
                    goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), node_idx)
                    costs.append(goal_cost)
                    is_goal_node.append(True)
                    push_count += 1
                    heapq.heappush(heap, (goal_cost, push_count, goal_idx))
                    self.last_debug.update(
                        {
                            "status": "goal_edge_reached"
                            if path_cost_mode and self.path_cost_return_first_goal
                            else "goal_edge_queued",
                            "steps": int(expansion_idx + 1),
                            "last_goal_dist": min(float(self.last_debug.get("last_goal_dist", node_goal_dist)), node_goal_dist),
                            "best_edge_min_clearance": max(
                                float(self.last_debug.get("best_edge_min_clearance", -1.0)),
                                float(final_edge[0]),
                            ),
                            "best_goal_path_cost": min(
                                float(self.last_debug.get("best_goal_path_cost", float("inf"))), goal_cost
                            ),
                        }
                    )
                    if path_cost_mode and self.path_cost_return_first_goal:
                        self.last_debug["last_goal_dist"] = 0.0
                        return self._search_path_to_array(nodes, parents, goal_idx)
                    continue

            grad_np = self._field_gradient(q_cur_n, q_goal_n)
            candidates = self._local_step_candidates(q_cur_n, q_goal_n, grad_np, float(step_size_q), max_candidates)
            candidates = self._append_cartesian_step_candidates(candidates, q_cur_n, q_goal_n, float(step_size_q))
            if len(candidates) == 0:
                continue
            edge_mins, clearances = self._edge_min_clearances(
                checker, q_cur_n, candidates, float(step_size_q), return_endpoints=True
            )
            if bool(getattr(checker, "expired", False)):
                self.last_debug.update(
                    {
                        "status": "learned_time_budget",
                        "steps": int(expansion_idx + 1),
                        "last_goal_dist": best_goal_dist,
                    }
                )
                return np.zeros((0, 6), dtype=np.float32)
            edge_ok = edge_mins >= float(clearance_margin_m)
            q_phys = np.asarray([self.kinematics.denormalize(qn) for qn in candidates], dtype=np.float32)
            goal_dists_all = np.linalg.norm(candidates - q_goal_n[None, :], axis=1)
            if len(clearances):
                self.last_debug["best_candidate_clearance"] = max(
                    float(self.last_debug.get("best_candidate_clearance", -1.0)),
                    float(np.nanmax(clearances)),
                )
            if len(edge_mins):
                self.last_debug["best_edge_min_clearance"] = max(
                    float(self.last_debug.get("best_edge_min_clearance", -1.0)),
                    float(np.nanmax(edge_mins)),
                )
            if len(goal_dists_all):
                self.last_debug["best_candidate_goal_dist"] = min(
                    float(self.last_debug.get("best_candidate_goal_dist", float("inf"))),
                    float(np.nanmin(goal_dists_all)),
                )
            self.last_debug["candidate_count"] = int(self.last_debug.get("candidate_count", 0)) + int(len(candidates))
            self.last_debug["valid_edge_count"] = int(self.last_debug.get("valid_edge_count", 0)) + int(np.count_nonzero(edge_ok))
            if not np.any(edge_ok):
                continue

            valid_candidates = candidates[edge_ok]
            valid_clearances = clearances[edge_ok]
            valid_edge_mins = edge_mins[edge_ok]
            valid_goal_dists = goal_dists_all[edge_ok]
            heuristic_scores = self._field_search_scores(
                valid_candidates,
                q_goal_n,
                depths[node_idx] + 1,
                valid_clearances,
                clearance_margin_m,
            )
            edge_costs = self._edge_path_costs(q_cur_n, valid_candidates, valid_edge_mins, clearance_margin_m)
            if self.path_turn_weight != 0.0 and int(parents[node_idx]) >= 0:
                edge_costs += self._turn_path_costs(nodes[int(parents[node_idx])], q_cur_n, valid_candidates)
            candidate_g_costs = float(costs[node_idx]) + edge_costs
            valid_scores = candidate_g_costs + heuristic_scores
            if expansion_idx == 0:
                top_idx = np.argsort(valid_scores)[: min(8, len(valid_scores))]
                self.last_debug["first_expansion_top_candidates"] = [
                    {
                        "score": float(valid_scores[int(i)]),
                        "g_cost": float(candidate_g_costs[int(i)]),
                        "heuristic": float(heuristic_scores[int(i)]),
                        "edge_cost": float(edge_costs[int(i)]),
                        "goal_dist": float(valid_goal_dists[int(i)]),
                        "clearance": float(valid_clearances[int(i)]),
                        "edge_min_clearance": float(valid_edge_mins[int(i)]),
                        "step_norm": float(np.linalg.norm(valid_candidates[int(i)] - q_cur_n)),
                    }
                    for i in top_idx
                ]
            order = np.argsort(valid_scores)
            for local_i in order:
                if len(nodes) >= max_nodes:
                    break
                cand = valid_candidates[int(local_i)].astype(np.float32)
                key = self._search_key(cand, step_size_q)
                child_cost = float(candidate_g_costs[int(local_i)])
                priority = float(valid_scores[int(local_i)])
                prev_cost = best_seen.get(key)
                if prev_cost is not None and child_cost >= prev_cost - 1.0e-6:
                    continue
                best_seen[key] = child_cost
                child_idx = self._append_search_node(nodes, parents, depths, cand, node_idx)
                costs.append(child_cost)
                is_goal_node.append(False)
                push_count += 1
                heapq.heappush(heap, (priority, push_count, child_idx))
                cand_goal_dist = float(valid_goal_dists[int(local_i)])
                if cand_goal_dist < best_goal_dist:
                    best_goal_dist = cand_goal_dist
                    best_idx = child_idx
                if cand_goal_dist <= tol:
                    final_edge = self._edge_min_clearances(checker, cand, q_goal_n[None, :], float(step_size_q))
                    if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                        goal_edge_cost = float(
                            self._edge_path_costs(cand, q_goal_n[None, :], final_edge, clearance_margin_m)[0]
                        )
                        goal_cost = child_cost + goal_edge_cost
                        goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), child_idx)
                        costs.append(goal_cost)
                        is_goal_node.append(True)
                        push_count += 1
                        heapq.heappush(heap, (goal_cost, push_count, goal_idx))
                        self.last_debug.update(
                            {
                                "status": "reached"
                                if path_cost_mode and self.path_cost_return_first_goal
                                else "goal_edge_queued",
                                "steps": int(expansion_idx + 1),
                                "last_goal_dist": 0.0,
                                "best_goal_path_cost": min(
                                    float(self.last_debug.get("best_goal_path_cost", float("inf"))), goal_cost
                                ),
                            }
                        )
                        if path_cost_mode and self.path_cost_return_first_goal:
                            return self._search_path_to_array(nodes, parents, goal_idx)
            self.last_debug["queued_nodes"] = int(len(heap))
            if len(nodes) >= max_nodes:
                self.last_debug.update({"status": "node_budget", "steps": int(expansion_idx + 1), "last_goal_dist": best_goal_dist})
                break
        else:
            self.last_debug.update({"status": "max_steps", "steps": int(max_steps), "last_goal_dist": best_goal_dist})
        self.last_debug["best_node_idx"] = int(best_idx)
        return np.zeros((0, 6), dtype=np.float32)

    def _append_search_node(
        self,
        nodes: list[np.ndarray],
        parents: list[int],
        depths: list[int],
        q_n: np.ndarray,
        parent_idx: int,
    ) -> int:
        nodes.append(np.asarray(q_n, dtype=np.float32).copy())
        parents.append(int(parent_idx))
        depths.append(int(depths[parent_idx]) + 1 if parent_idx >= 0 else 0)
        return len(nodes) - 1

    def _search_path_to_array(self, nodes: list[np.ndarray], parents: list[int], idx: int) -> np.ndarray:
        out = []
        cur = int(idx)
        while cur >= 0:
            out.append(nodes[cur])
            cur = int(parents[cur])
        out.reverse()
        return np.asarray([self.kinematics.denormalize(qn) for qn in out], dtype=np.float32)

    def _search_key(self, q_n: np.ndarray, step_size_q: float) -> tuple[int, ...]:
        cell = max(0.01, 0.5 * float(step_size_q))
        return tuple(np.round(np.asarray(q_n, dtype=np.float32) / cell).astype(np.int32).tolist())

    def _edge_path_costs(
        self,
        q_cur_n: np.ndarray,
        candidates_n: np.ndarray,
        edge_mins: np.ndarray,
        clearance_margin_m: float,
    ) -> np.ndarray:
        candidates_n = np.asarray(candidates_n, dtype=np.float32)
        if candidates_n.ndim != 2 or len(candidates_n) == 0:
            return np.zeros((0,), dtype=np.float32)
        out = np.zeros((len(candidates_n),), dtype=np.float32)
        if self.path_joint_edge_weight != 0.0:
            joint_edge = np.linalg.norm(candidates_n - np.asarray(q_cur_n, dtype=np.float32)[None, :], axis=1)
            out += float(self.path_joint_edge_weight) * joint_edge.astype(np.float32)
        if self.path_tool_edge_weight != 0.0:
            tool_t0 = time.perf_counter()
            cur_tool = self._tool_xyz_from_normalized(np.asarray(q_cur_n, dtype=np.float32)[None, :])[0]
            cand_tool = self._tool_xyz_from_normalized(candidates_n)
            tool_edge = np.linalg.norm(cand_tool - cur_tool[None, :], axis=1)
            out += float(self.path_tool_edge_weight) * tool_edge.astype(np.float32)
            self._add_debug_ms("path_tool_edge_fk_ms", tool_t0)
            self._inc_debug("path_tool_edge_fk_states", len(candidates_n) + 1)
        if self.path_clearance_penalty_weight != 0.0:
            soft_threshold = float(clearance_margin_m) + float(self.path_clearance_soft_margin_m)
            soft_violation = np.maximum(soft_threshold - np.asarray(edge_mins, dtype=np.float32), 0.0)
            out += float(self.path_clearance_penalty_weight) * (soft_violation * soft_violation).astype(np.float32)
        return out

    def _turn_path_costs(
        self,
        q_parent_n: np.ndarray,
        q_cur_n: np.ndarray,
        candidates_n: np.ndarray,
    ) -> np.ndarray:
        incoming = np.asarray(q_cur_n, dtype=np.float32) - np.asarray(q_parent_n, dtype=np.float32)
        incoming_norm = float(np.linalg.norm(incoming))
        candidates = np.asarray(candidates_n, dtype=np.float32).reshape(-1, 6)
        if incoming_norm <= 1.0e-8 or len(candidates) == 0:
            return np.zeros((len(candidates),), dtype=np.float32)
        outgoing = candidates - np.asarray(q_cur_n, dtype=np.float32)[None, :]
        outgoing_norm = np.linalg.norm(outgoing, axis=1)
        denom = np.maximum(incoming_norm * outgoing_norm, 1.0e-8)
        cosine = np.clip((outgoing @ incoming) / denom, -1.0, 1.0)
        return (float(self.path_turn_weight) * (1.0 - cosine)).astype(np.float32)

    def _tool_xyz_from_normalized(self, q_n: np.ndarray) -> np.ndarray:
        q_n = np.asarray(q_n, dtype=np.float32)
        if q_n.ndim == 1:
            q_n = q_n[None, :]
        q_phys = np.asarray([self.kinematics.denormalize(q) for q in q_n], dtype=np.float32)
        return np.asarray([self.kinematics.fk(q)[:3, 3] for q in q_phys], dtype=np.float32)

    def _field_gradient(self, q_cur_n: np.ndarray, q_goal_n: np.ndarray) -> np.ndarray:
        xp = torch.from_numpy(np.concatenate((q_cur_n, q_goal_n), axis=0)[None, :].astype(np.float32)).to(
            self.field_model.device
        )
        infer_t0 = time.perf_counter()
        grad = self.field_model.model.function.Gradient(xp)
        grad_np = grad[0, :6].detach().cpu().numpy().astype(np.float32)
        self._add_debug_ms("gradient_infer_ms", infer_t0)
        self._inc_debug("gradient_infer_calls", 1)
        self._inc_debug("gradient_infer_pairs", 1)
        return grad_np

    def _field_search_priority(
        self,
        q_n: np.ndarray,
        q_goal_n: np.ndarray,
        depth: int,
        clearance: float,
        clearance_margin_m: float,
    ) -> float:
        return float(self._field_search_scores(
            np.asarray(q_n, dtype=np.float32)[None, :],
            np.asarray(q_goal_n, dtype=np.float32),
            depth,
            np.asarray([clearance], dtype=np.float32),
            clearance_margin_m,
        )[0])

    def _field_search_scores(
        self,
        candidates_n: np.ndarray,
        q_goal_n: np.ndarray,
        depth: int,
        clearances: np.ndarray,
        clearance_margin_m: float,
    ) -> np.ndarray:
        candidates_n = np.asarray(candidates_n, dtype=np.float32)
        if candidates_n.ndim != 2 or len(candidates_n) == 0:
            return np.zeros((0,), dtype=np.float32)
        path_cost_mode = (
            self.path_joint_edge_weight != 0.0
            or self.path_turn_weight != 0.0
            or self.path_tool_edge_weight != 0.0
            or self.path_tool_goal_weight != 0.0
        )
        if path_cost_mode:
            tau = np.zeros((len(candidates_n),), dtype=np.float32)
        else:
            q_goals = np.repeat(np.asarray(q_goal_n, dtype=np.float32)[None, :], len(candidates_n), axis=0)
            infer_t0 = time.perf_counter()
            tau = self.field_model.predict_travel_times(candidates_n, q_goals)
            self._add_debug_ms("tau_infer_ms", infer_t0)
            self._inc_debug("tau_infer_calls", 1)
            self._inc_debug("tau_infer_pairs", len(candidates_n))
        goal_dist = np.linalg.norm(candidates_n - q_goal_n[None, :], axis=1)
        clearance_bonus = np.clip(np.asarray(clearances, dtype=np.float32) - float(clearance_margin_m), 0.0, 0.20)
        tool_goal = np.zeros((len(candidates_n),), dtype=np.float32)
        if self.path_tool_goal_weight != 0.0:
            tool_t0 = time.perf_counter()
            cand_tool = self._tool_xyz_from_normalized(candidates_n)
            goal_tool = self._tool_xyz_from_normalized(np.asarray(q_goal_n, dtype=np.float32)[None, :])[0]
            tool_goal = np.linalg.norm(cand_tool - goal_tool[None, :], axis=1).astype(np.float32)
            self._add_debug_ms("path_tool_goal_fk_ms", tool_t0)
            self._inc_debug("path_tool_goal_fk_states", len(candidates_n) + 1)
        if path_cost_mode:
            # Path-cost mode: accumulated edge cost is the objective, so the
            # heuristic must stay geometric. Keep both tool-space and joint-space
            # goal progress, otherwise tool-only IK-equivalent states can look
            # cheap while remaining hard to connect to a fixed joint goal.
            score = (
                self.path_tool_goal_weight * tool_goal
                + self.score_goal_dist_weight * goal_dist
                - self.score_clearance_weight * clearance_bonus
            )
        else:
            # Best-first field search: the learned travel time ranks branches,
            # geometric distance prevents metric basins from dominating, and
            # depth keeps the search from preferring long wandering paths.
            score = (
                self.score_tau_weight * tau
                + self.score_goal_dist_weight * goal_dist
                + self.score_depth_weight * float(depth)
                - self.score_clearance_weight * clearance_bonus
            )
        return np.where(np.isfinite(score), score, np.inf).astype(np.float32)

    def _postprocess_collision_path(
        self,
        checker: UR5PointCloudCollisionChecker,
        path: np.ndarray,
        clearance_margin_m: float,
        step_size_q: float,
        shortcut_path: bool,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) <= 2:
            self.last_debug["cartesian_shortcut_enabled"] = bool(self.cartesian_shortcut_enabled)
            self.last_debug["cartesian_shortcut_accepted"] = False
            return pts

        baseline = self._shortcut_collision_path(checker, pts, clearance_margin_m, step_size_q) if shortcut_path else pts
        if not self.cartesian_shortcut_enabled:
            self.last_debug["cartesian_shortcut_enabled"] = False
            self.last_debug["cartesian_shortcut_accepted"] = False
            return self._smooth_collision_path(checker, baseline, clearance_margin_m)

        opt_t0 = time.perf_counter()
        candidate = self._cartesian_shortcut_path(checker, pts, clearance_margin_m, step_size_q)
        base_cost = self._cartesian_path_quality_cost(baseline)
        cand_cost = self._cartesian_path_quality_cost(candidate)
        improvement = 0.0
        accepted = False
        out = baseline
        if len(candidate) > 0 and np.isfinite(cand_cost) and np.isfinite(base_cost):
            improvement = (base_cost - cand_cost) / max(1.0e-6, abs(base_cost))
            if cand_cost < base_cost and improvement >= float(self.cartesian_shortcut_min_improvement):
                out = candidate
                accepted = True
        self._add_debug_ms("cartesian_shortcut_ms", opt_t0)
        self.last_debug["cartesian_shortcut_enabled"] = True
        self.last_debug["cartesian_shortcut_accepted"] = bool(accepted)
        self.last_debug["cartesian_shortcut_base_cost"] = float(base_cost)
        self.last_debug["cartesian_shortcut_candidate_cost"] = float(cand_cost)
        self.last_debug["cartesian_shortcut_improvement"] = float(improvement)
        self.last_debug["cartesian_shortcut_candidate_waypoints"] = int(len(candidate))
        return self._smooth_collision_path(checker, np.asarray(out, dtype=np.float32), clearance_margin_m)

    def _cartesian_path_quality_cost(self, path: np.ndarray) -> float:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            return float("inf")
        if len(pts) == 1:
            return 0.0
        qn = np.asarray([self.kinematics.normalize(q) for q in pts], dtype=np.float32)
        joint_len = float(np.sum(np.linalg.norm(np.diff(qn, axis=0), axis=1)))
        tool_xyz = np.asarray([self.kinematics.fk(q)[:3, 3] for q in pts], dtype=np.float32)
        tool_len = float(np.sum(np.linalg.norm(np.diff(tool_xyz, axis=0), axis=1)))
        smoothness = 0.0
        if len(tool_xyz) >= 3 and self.cartesian_shortcut_smoothness_weight != 0.0:
            smoothness = float(np.sum(np.linalg.norm(tool_xyz[2:] - 2.0 * tool_xyz[1:-1] + tool_xyz[:-2], axis=1)))
        return (
            float(self.cartesian_shortcut_tool_weight) * tool_len
            + float(self.cartesian_shortcut_joint_weight) * joint_len
            + float(self.cartesian_shortcut_smoothness_weight) * smoothness
        )

    def _cartesian_segment_quality_cost(
        self,
        qn_from: np.ndarray,
        q_from: np.ndarray,
        tool_from: np.ndarray,
        qn_to: np.ndarray,
        q_to: np.ndarray,
        tool_to: np.ndarray,
    ) -> np.ndarray:
        qn_to = np.asarray(qn_to, dtype=np.float32)
        if qn_to.ndim == 1:
            qn_to = qn_to[None, :]
        tool_to = np.asarray(tool_to, dtype=np.float32)
        if tool_to.ndim == 1:
            tool_to = tool_to[None, :]
        joint_len = np.linalg.norm(qn_to - np.asarray(qn_from, dtype=np.float32)[None, :], axis=1)
        tool_len = np.linalg.norm(tool_to - np.asarray(tool_from, dtype=np.float32)[None, :], axis=1)
        return (
            float(self.cartesian_shortcut_tool_weight) * tool_len
            + float(self.cartesian_shortcut_joint_weight) * joint_len
        ).astype(np.float32)

    def _cartesian_shortcut_path(
        self,
        checker: UR5PointCloudCollisionChecker,
        path: np.ndarray,
        clearance_margin_m: float,
        step_size_q: float,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) <= 2:
            return pts
        qn = np.asarray([self.kinematics.normalize(q) for q in pts], dtype=np.float32)
        tool_xyz = np.asarray([self.kinematics.fk(q)[:3, 3] for q in pts], dtype=np.float32)
        n = int(len(pts))
        best_cost = np.full((n,), np.inf, dtype=np.float64)
        best_next = np.full((n,), -1, dtype=np.int32)
        best_cost[-1] = 0.0
        max_skip = max(2, int(self.cartesian_shortcut_max_skip))
        edge_checks = 0
        valid_edges = 0

        for idx in range(n - 2, -1, -1):
            end = min(n, idx + max_skip + 1)
            cand_indices = np.arange(idx + 1, end, dtype=np.int32)
            if cand_indices.size == 0:
                continue
            edge_clearances = self._edge_min_clearances(checker, qn[idx], qn[cand_indices], float(step_size_q))
            edge_checks += int(cand_indices.size)
            valid = edge_clearances >= float(clearance_margin_m)
            valid &= np.isfinite(best_cost[cand_indices])
            if not np.any(valid):
                continue
            valid_indices = cand_indices[valid]
            segment_cost = self._cartesian_segment_quality_cost(
                qn[idx],
                pts[idx],
                tool_xyz[idx],
                qn[valid_indices],
                pts[valid_indices],
                tool_xyz[valid_indices],
            ).astype(np.float64)
            total_cost = segment_cost + best_cost[valid_indices]
            best_local = int(np.argmin(total_cost))
            best_cost[idx] = float(total_cost[best_local])
            best_next[idx] = int(valid_indices[best_local])
            valid_edges += int(np.count_nonzero(valid))

        if best_next[0] <= 0:
            self.last_debug["cartesian_shortcut_edge_checks"] = int(edge_checks)
            self.last_debug["cartesian_shortcut_valid_edges"] = int(valid_edges)
            return pts

        out = [pts[0].copy()]
        cur = 0
        visited = {0}
        while cur < n - 1:
            nxt = int(best_next[cur])
            if nxt <= cur or nxt in visited:
                return pts
            out.append(pts[nxt].copy())
            visited.add(nxt)
            cur = nxt
        self.last_debug["cartesian_shortcut_edge_checks"] = int(edge_checks)
        self.last_debug["cartesian_shortcut_valid_edges"] = int(valid_edges)
        return np.asarray(out, dtype=np.float32)

    def _shortcut_collision_path(
        self,
        checker: UR5PointCloudCollisionChecker,
        path: np.ndarray,
        clearance_margin_m: float,
        step_size_q: float,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) <= 2:
            return pts
        out = [pts[0].copy()]
        cur = 0
        while cur < len(pts) - 1:
            # Test a window of possible shortcuts in one checker batch.  The old
            # farthest-first loop launched one GPU query for every failed edge.
            end = min(len(pts), cur + max(2, int(getattr(self, "cartesian_shortcut_max_skip", 48))) + 1)
            cand_indices = np.arange(cur + 1, end, dtype=np.int32)
            q_cur_n = self.kinematics.normalize(pts[cur])
            next_idx = cur + 1
            # Check farthest candidates first, eight at a time.  This retains
            # GPU batching without evaluating every long edge when an early
            # far shortcut succeeds.
            descending = cand_indices[::-1]
            for chunk_start in range(0, len(descending), 8):
                chunk = descending[chunk_start : chunk_start + 8]
                q_cand_n = np.asarray([self.kinematics.normalize(pts[i]) for i in chunk], dtype=np.float32)
                edge = self._edge_min_clearances(checker, q_cur_n, q_cand_n, float(step_size_q))
                valid = np.flatnonzero(edge >= float(clearance_margin_m))
                if len(valid):
                    next_idx = int(chunk[int(valid[0])])
                    break
            out.append(pts[next_idx].copy())
            cur = next_idx
        return np.asarray(out, dtype=np.float32)

    def _smooth_collision_path(
        self,
        checker: UR5PointCloudCollisionChecker,
        path: np.ndarray,
        clearance_margin_m: float,
    ) -> np.ndarray:
        """Remove sharp joint-space corners without relaxing the safety margin."""
        pts = np.asarray(path, dtype=np.float32)
        if (
            pts.ndim != 2
            or len(pts) <= 2
            or not bool(getattr(self, "collision_smoothing_enabled", True))
        ):
            return pts
        original = pts.copy()
        out = pts.copy()
        alpha = float(np.clip(getattr(self, "collision_smoothing_alpha", 0.35), 0.0, 1.0))
        passes = max(0, int(getattr(self, "collision_smoothing_passes", 8)))
        accepted = 0
        smooth_t0 = time.perf_counter()
        for _pass in range(passes):
            changed = False
            # Alternating non-adjacent corners means every accepted connecting
            # segment was present in the same batched validation query.
            for parity in (0, 1):
                indices = np.arange(1 + parity, len(out) - 1, 2, dtype=np.int32)
                if len(indices) == 0:
                    continue
                current = out[indices]
                proposal = (1.0 - alpha) * current + alpha * 0.5 * (out[indices - 1] + out[indices + 1])
                starts = np.vstack((out[indices - 1], proposal)).astype(np.float32)
                ends = np.vstack((proposal, out[indices + 1])).astype(np.float32)
                mins, _endpoints = self._physical_segments_min_clearances(checker, starts, ends)
                n = len(indices)
                safe = (mins[:n] >= float(clearance_margin_m)) & (mins[n:] >= float(clearance_margin_m))
                old_curve = np.linalg.norm(out[indices - 1] - 2.0 * current + out[indices + 1], axis=1)
                new_curve = np.linalg.norm(out[indices - 1] - 2.0 * proposal + out[indices + 1], axis=1)
                improve = new_curve < old_curve - 1.0e-7
                take = safe & improve
                if np.any(take):
                    out[indices[take]] = proposal[take]
                    accepted += int(np.count_nonzero(take))
                    changed = True
            if not changed:
                break

        # A single batched whole-path check is the final invariant.  Any numeric
        # or checker inconsistency falls back to the already-valid shortcut path.
        if len(out) > 1:
            final_step = float(getattr(self, "final_edge_step_rad", 0.02))
            final_mins, _ = self._physical_segments_min_clearances(
                checker, out[:-1], out[1:], max_step_rad=final_step
            )
            if len(final_mins) != len(out) - 1 or np.any(final_mins < float(clearance_margin_m)):
                original_mins, _ = self._physical_segments_min_clearances(
                    checker, original[:-1], original[1:], max_step_rad=final_step
                )
                out = original if np.all(original_mins >= float(clearance_margin_m)) else np.zeros((0, 6), dtype=np.float32)
                accepted = 0
        self._add_debug_ms("collision_smoothing_ms", smooth_t0)
        self.last_debug["collision_smoothing_accepted_corners"] = int(accepted)
        self.last_debug["collision_smoothing_enabled"] = True
        return np.asarray(out, dtype=np.float32)

    def _edge_min_clearances(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_cur_n: np.ndarray,
        candidates_n: np.ndarray,
        step_size_q: float,
        *,
        return_endpoints: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        del step_size_q  # search resolution and collision resolution are independent
        q_cur_n = np.asarray(q_cur_n, dtype=np.float32)
        candidates_n = np.asarray(candidates_n, dtype=np.float32)
        if candidates_n.ndim == 1:
            candidates_n = candidates_n[None, :]
        if len(candidates_n) == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return (empty, empty) if return_endpoints else empty
        start = np.asarray(self.kinematics.denormalize(q_cur_n), dtype=np.float32)
        ends = np.asarray([self.kinematics.denormalize(row) for row in candidates_n], dtype=np.float32)
        starts = np.repeat(start[None, :], len(ends), axis=0)
        mins, endpoint_clearances = self._physical_segments_min_clearances(checker, starts, ends)
        return (mins, endpoint_clearances) if return_endpoints else mins

    def _physical_segments_min_clearances(
        self,
        checker: UR5PointCloudCollisionChecker,
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        max_step_rad: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(starts, dtype=np.float32).reshape(-1, 6)
        ends = np.asarray(ends, dtype=np.float32).reshape(-1, 6)
        if len(starts) != len(ends):
            raise ValueError("starts and ends must contain the same number of segments")
        if len(starts) == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty
        max_step_rad = max(
            1.0e-3,
            float(
                getattr(self, "planning_edge_step_rad", 0.10)
                if max_step_rad is None
                else max_step_rad
            ),
        )
        max_deltas = np.max(np.abs(ends - starts), axis=1)
        nsegs = np.maximum(1, np.ceil(max_deltas / max_step_rad).astype(np.int32))
        edges = []
        edge_ids = []
        endpoint_rows = []
        row_offset = 0
        for edge_idx, (q_start, q_end, nseg) in enumerate(zip(starts, ends, nsegs)):
            alpha = (np.arange(int(nseg) + 1, dtype=np.float32) / float(nseg))[:, None]
            pts = q_start[None, :] + alpha * (q_end[None, :] - q_start[None, :])
            edges.append(pts)
            edge_ids.extend([edge_idx] * len(pts))
            endpoint_rows.append(row_offset + len(pts) - 1)
            row_offset += len(pts)
        all_pts = np.vstack(edges).astype(np.float32)
        edge_t0 = time.perf_counter()
        all_clearances = checker.clearance_batch(all_pts)
        self._add_debug_ms("edge_check_ms", edge_t0)
        self._inc_debug("edge_check_calls", 1)
        self._inc_debug("edge_check_edges", len(starts))
        self._inc_debug("edge_check_states", len(all_pts))
        out = np.full((len(starts),), np.inf, dtype=np.float32)
        for edge_idx, clearance in zip(edge_ids, all_clearances):
            out[int(edge_idx)] = min(float(out[int(edge_idx)]), float(clearance))
        endpoint_clearances = np.asarray(all_clearances, dtype=np.float32)[np.asarray(endpoint_rows, dtype=np.int32)]
        return out.astype(np.float32), endpoint_clearances.astype(np.float32)

    def _local_step_candidates(
        self,
        q_cur_n: np.ndarray,
        q_goal_n: np.ndarray,
        grad: np.ndarray,
        step_size_q: float,
        max_local_candidates: int,
    ) -> np.ndarray:
        q_cur_n = np.asarray(q_cur_n, dtype=np.float32)
        q_goal_n = np.asarray(q_goal_n, dtype=np.float32)
        directions: list[np.ndarray] = []
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1.0e-8 and np.all(np.isfinite(grad)):
            directions.append(np.asarray(grad / grad_norm, dtype=np.float32))
        goal_delta = q_goal_n - q_cur_n
        goal_norm = float(np.linalg.norm(goal_delta))
        if goal_norm > 1.0e-8:
            directions.append(np.asarray(goal_delta / goal_norm, dtype=np.float32))
        for axis in range(6):
            basis = np.zeros((6,), dtype=np.float32)
            basis[axis] = 1.0
            directions.append(basis)
            directions.append(-basis)
            if goal_norm > 1.0e-8:
                for sign in (-1.0, 1.0):
                    mixed = goal_delta / goal_norm + sign * 0.75 * basis
                    mixed_norm = float(np.linalg.norm(mixed))
                    if mixed_norm > 1.0e-8:
                        directions.append(np.asarray(mixed / mixed_norm, dtype=np.float32))
            if grad_norm > 1.0e-8 and np.all(np.isfinite(grad)):
                for sign in (-1.0, 1.0):
                    mixed = grad / grad_norm + sign * 0.75 * basis
                    mixed_norm = float(np.linalg.norm(mixed))
                    if mixed_norm > 1.0e-8:
                        directions.append(np.asarray(mixed / mixed_norm, dtype=np.float32))
        if not directions:
            return np.zeros((0, 6), dtype=np.float32)
        steps = (1.0, 0.5, 1.5)
        out: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for direction in directions:
            for step in steps:
                cand = np.clip(q_cur_n + float(step_size_q) * float(step) * direction, -0.5, 0.5).astype(np.float32)
                key = tuple(np.round(cand, 4).tolist())
                if key in seen or np.max(np.abs(cand - q_cur_n)) < 1.0e-5:
                    continue
                seen.add(key)
                out.append(cand)
                if len(out) >= max(1, max_local_candidates):
                    return np.asarray(out, dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _append_cartesian_step_candidates(
        self,
        candidates_n: np.ndarray,
        q_cur_n: np.ndarray,
        q_goal_n: np.ndarray,
        step_size_q: float,
    ) -> np.ndarray:
        extra = self._cartesian_step_candidates(q_cur_n, q_goal_n, step_size_q)
        if len(extra) == 0:
            return np.asarray(candidates_n, dtype=np.float32)
        base = np.asarray(candidates_n, dtype=np.float32)
        if base.ndim != 2 or len(base) == 0:
            out = extra
        else:
            out = np.vstack((base, extra)).astype(np.float32)
        deduped: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for cand in out:
            key = tuple(np.round(cand, 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(np.asarray(cand, dtype=np.float32))
        return np.asarray(deduped, dtype=np.float32)

    def _cartesian_step_candidates(
        self,
        q_cur_n: np.ndarray,
        q_goal_n: np.ndarray,
        step_size_q: float,
    ) -> np.ndarray:
        count = max(0, int(self.cartesian_candidate_count))
        if count <= 0:
            return np.zeros((0, 6), dtype=np.float32)
        t0 = time.perf_counter()
        q_cur_n = np.asarray(q_cur_n, dtype=np.float32).reshape(6)
        q_goal_n = np.asarray(q_goal_n, dtype=np.float32).reshape(6)
        q_cur = self.kinematics.denormalize(q_cur_n)
        q_goal = self.kinematics.denormalize(q_goal_n)
        cur_xyz = self.kinematics.fk(q_cur)[:3, 3]
        goal_xyz = self.kinematics.fk(q_goal)[:3, 3]
        delta_xyz = goal_xyz - cur_xyz
        dist = float(np.linalg.norm(delta_xyz))
        if not np.isfinite(dist) or dist <= 1.0e-8:
            return np.zeros((0, 6), dtype=np.float32)

        desired_step = delta_xyz * (min(float(self.cartesian_candidate_step_m), dist) / dist)
        jac_pos = self.kinematics.numerical_jacobian(q_cur)[:3, :]
        if not np.all(np.isfinite(jac_pos)):
            return np.zeros((0, 6), dtype=np.float32)

        damping = float(self.cartesian_candidate_damping)
        lhs = jac_pos @ jac_pos.T + damping * np.eye(3, dtype=np.float64)
        try:
            dq = jac_pos.T @ np.linalg.solve(lhs, desired_step)
        except np.linalg.LinAlgError:
            return np.zeros((0, 6), dtype=np.float32)
        if not np.all(np.isfinite(dq)) or float(np.linalg.norm(dq)) <= 1.0e-10:
            return np.zeros((0, 6), dtype=np.float32)

        out: list[np.ndarray] = []
        scales = (1.0, 0.5, 1.5, 2.0, -0.5, -1.0)
        joint_ranges = self.kinematics.joint_max - self.kinematics.joint_min
        base_n = self.kinematics.normalize(self.kinematics.clamp(q_cur + dq)).astype(np.float32)
        base_delta_n = base_n - q_cur_n
        base_delta_norm = float(np.linalg.norm(base_delta_n))
        if base_delta_norm <= 1.0e-8:
            return np.zeros((0, 6), dtype=np.float32)
        base_direction_n = (base_delta_n / base_delta_norm).astype(np.float32)
        for scale in scales:
            if len(out) >= count:
                break
            cand_n = np.clip(
                q_cur_n + float(step_size_q) * float(scale) * base_direction_n,
                -0.5,
                0.5,
            ).astype(np.float32)
            if float(np.max(np.abs(cand_n - q_cur_n))) < 1.0e-5:
                continue
            out.append(cand_n)

        # Add a translational-Jacobian transpose direction as a robust fallback
        # when the damped least-squares step is nearly singular.
        if len(out) < count:
            grad_q = jac_pos.T @ (delta_xyz / dist)
            grad_n = (grad_q * joint_ranges).astype(np.float64)
            grad_norm = float(np.linalg.norm(grad_n))
            if np.isfinite(grad_norm) and grad_norm > 1.0e-8:
                direction = (grad_n / grad_norm).astype(np.float32)
                for scale in (1.0, 0.5, 1.5):
                    if len(out) >= count:
                        break
                    cand_n = np.clip(q_cur_n + float(step_size_q) * float(scale) * direction, -0.5, 0.5).astype(np.float32)
                    if float(np.max(np.abs(cand_n - q_cur_n))) >= 1.0e-5:
                        out.append(cand_n)

        self._add_debug_ms("cartesian_candidate_fk_ms", t0)
        self._inc_debug("cartesian_candidate_calls", 1)
        self._inc_debug("cartesian_candidate_count", len(out))
        return np.asarray(out, dtype=np.float32)

    def _goal_directed_prior_controls(
        self,
        q_cur_n: np.ndarray,
        q_goal_n: np.ndarray,
        step_size_q: float,
        horizon: int,
    ) -> np.ndarray:
        controls = np.zeros((max(1, int(horizon)), 6), dtype=np.float32)
        q_tmp = np.asarray(q_cur_n, dtype=np.float32).copy()
        q_goal_n = np.asarray(q_goal_n, dtype=np.float32)
        for idx in range(len(controls)):
            delta = q_goal_n - q_tmp
            dist = float(np.linalg.norm(delta))
            if dist <= 1.0e-8:
                break
            step = min(float(step_size_q), dist)
            controls[idx] = (step / dist) * delta
            q_tmp = np.clip(q_tmp + controls[idx], -0.5, 0.5).astype(np.float32)
        return controls

    def _sample_rollout_controls(
        self,
        prior: np.ndarray,
        sample_count: int,
        step_size_q: float,
        noise_scale: float,
    ) -> np.ndarray:
        prior = np.asarray(prior, dtype=np.float32)
        sample_count = max(1, int(sample_count))
        noise_std = max(0.0, float(noise_scale)) * float(step_size_q)
        controls = prior[None, :, :] + self.rng.normal(
            0.0, noise_std, size=(sample_count, prior.shape[0], prior.shape[1])
        ).astype(np.float32)
        controls[0] = prior
        max_norm = max(float(step_size_q), 1.5 * float(step_size_q))
        norms = np.linalg.norm(controls, axis=2, keepdims=True)
        controls = controls * np.minimum(1.0, max_norm / np.maximum(norms, 1.0e-8))
        return controls.astype(np.float32)

    def _integrate_normalized_controls(self, q_cur_n: np.ndarray, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float32)
        q = np.repeat(np.asarray(q_cur_n, dtype=np.float32)[None, :], controls.shape[0], axis=0)
        states = []
        for t_idx in range(controls.shape[1]):
            q = np.clip(q + controls[:, t_idx, :], -0.5, 0.5).astype(np.float32)
            states.append(q.copy())
        return np.stack(states, axis=1).astype(np.float32)

    def _sampled_rollout_costs(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_cur_n: np.ndarray,
        rollouts_n: np.ndarray,
        q_goal_n: np.ndarray,
        clearance_margin_m: float,
        tau_weight: float,
        goal_dist_weight: float,
        joint_path_weight: float,
        tool_path_weight: float,
        clearance_penalty_weight: float,
    ) -> np.ndarray:
        rollouts_n = np.asarray(rollouts_n, dtype=np.float32)
        q_cur_n = np.asarray(q_cur_n, dtype=np.float32).reshape(6)
        q_goal_n = np.asarray(q_goal_n, dtype=np.float32).reshape(6)
        if rollouts_n.ndim != 3 or rollouts_n.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        sample_count, horizon, _ = rollouts_n.shape

        final_n = rollouts_n[:, -1, :]
        q_goals = np.repeat(q_goal_n[None, :], sample_count, axis=0)
        tau_t0 = time.perf_counter()
        tau = np.asarray(self.field_model.predict_travel_times(final_n, q_goals), dtype=np.float32).reshape(-1)
        self._add_debug_ms("sampled_tau_infer_ms", tau_t0)
        self._inc_debug("sampled_tau_infer_calls", 1)
        self._inc_debug("sampled_tau_infer_pairs", sample_count)

        pts_n = np.concatenate(
            [np.repeat(q_cur_n[None, None, :], sample_count, axis=0), rollouts_n],
            axis=1,
        )
        deltas_n = np.diff(pts_n, axis=1)
        joint_path = np.sum(np.linalg.norm(deltas_n, axis=2), axis=1).astype(np.float32)
        goal_dist = np.linalg.norm(final_n - q_goal_n[None, :], axis=1).astype(np.float32)

        flat_n = pts_n.reshape(-1, 6)
        q_flat = np.asarray([self.kinematics.denormalize(qn) for qn in flat_n], dtype=np.float32)

        fk_t0 = time.perf_counter()
        tool_xyz = np.asarray([self.kinematics.fk(q)[:3, 3] for q in q_flat], dtype=np.float32)
        tool_xyz = tool_xyz.reshape(sample_count, horizon + 1, 3)
        tool_path = np.sum(np.linalg.norm(np.diff(tool_xyz, axis=1), axis=2), axis=1).astype(np.float32)
        self._add_debug_ms("sampled_fk_tool_ms", fk_t0)
        self._inc_debug("sampled_fk_tool_states", len(q_flat))

        clearance_t0 = time.perf_counter()
        clearances = np.asarray(checker.clearance_batch(q_flat), dtype=np.float32).reshape(sample_count, horizon + 1)
        self._add_debug_ms("sampled_clearance_ms", clearance_t0)
        self._inc_debug("sampled_clearance_calls", 1)
        self._inc_debug("sampled_clearance_states", len(q_flat))
        violation = np.maximum(float(clearance_margin_m) - clearances, 0.0)
        clearance_penalty = np.sum(violation * violation, axis=1).astype(np.float32)
        min_clearance = np.min(clearances, axis=1)
        valid_rollout = min_clearance >= float(clearance_margin_m)

        cost = (
            float(tau_weight) * tau
            + float(goal_dist_weight) * goal_dist
            + float(joint_path_weight) * joint_path
            + float(tool_path_weight) * tool_path
            + float(clearance_penalty_weight) * clearance_penalty
        )
        cost = np.where(valid_rollout, cost, cost + 1.0e3).astype(np.float32)
        self._inc_debug("sampled_valid_rollouts", int(np.count_nonzero(valid_rollout)))
        self._inc_debug("sampled_invalid_rollouts", int(len(valid_rollout) - np.count_nonzero(valid_rollout)))
        if len(cost) > 0:
            best_idx = int(np.nanargmin(np.where(np.isfinite(cost), cost, np.inf)))
            self.last_debug["sampled_best_cost"] = float(cost[best_idx])
            self.last_debug["sampled_best_goal_dist"] = float(goal_dist[best_idx])
            self.last_debug["sampled_best_tau"] = float(tau[best_idx])
            self.last_debug["sampled_best_joint_path"] = float(joint_path[best_idx])
            self.last_debug["sampled_best_tool_path_m"] = float(tool_path[best_idx])
            self.last_debug["sampled_best_min_clearance_m"] = float(min_clearance[best_idx])
        return cost.astype(np.float32)

    def _weighted_control_update(
        self,
        controls: np.ndarray,
        costs: np.ndarray,
        temperature: float,
        fallback: np.ndarray,
    ) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32).reshape(-1)
        finite = np.isfinite(costs)
        if controls.ndim != 3 or not np.any(finite):
            return np.asarray(fallback, dtype=np.float32)
        finite_costs = costs[finite]
        finite_controls = controls[finite]
        logits = -max(1.0e-6, float(temperature)) * (finite_costs - float(np.min(finite_costs)))
        logits = logits - float(np.max(logits))
        weights = np.exp(logits).astype(np.float32)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1.0e-12:
            return np.asarray(fallback, dtype=np.float32)
        weights /= weight_sum
        return np.einsum("n,nhd->hd", weights, finite_controls).astype(np.float32)

    def _add_debug_ms(self, key: str, start_time_s: float) -> None:
        if not isinstance(getattr(self, "last_debug", None), dict):
            return
        prev = float(self.last_debug.get(key, 0.0))
        self.last_debug[key] = prev + (time.perf_counter() - float(start_time_s)) * 1.0e3

    def _inc_debug(self, key: str, value: int | float) -> None:
        if not isinstance(getattr(self, "last_debug", None), dict):
            return
        self.last_debug[key] = int(self.last_debug.get(key, 0)) + int(value)

    def _valid_step_mask(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_cur_n: np.ndarray,
        candidates_n: np.ndarray,
        clearance_margin_m: float,
        step_size_q: float,
    ) -> np.ndarray:
        q_cur = self.kinematics.denormalize(q_cur_n)
        out = []
        max_delta_step = max(0.01, 0.5 * step_size_q)
        for cand_n in np.asarray(candidates_n, dtype=np.float32):
            q_cand = self.kinematics.denormalize(cand_n)
            max_delta = float(np.max(np.abs(q_cand - q_cur)))
            nseg = max(1, int(np.ceil(max_delta / max_delta_step)))
            edge = np.asarray(
                [q_cur + (float(i) / float(nseg)) * (q_cand - q_cur) for i in range(nseg + 1)],
                dtype=np.float32,
            )
            clearances = checker.clearance_batch(edge)
            out.append(bool(clearances.size > 0 and float(np.min(clearances)) >= clearance_margin_m))
        return np.asarray(out, dtype=bool)

    def _best_local_candidate(
        self,
        checker: UR5PointCloudCollisionChecker,
        candidates_n: np.ndarray,
        q_goal_n: np.ndarray,
        clearance_margin_m: float,
    ) -> np.ndarray | None:
        candidates_n = np.asarray(candidates_n, dtype=np.float32)
        if candidates_n.ndim != 2 or len(candidates_n) == 0:
            return None
        q_goals = np.repeat(np.asarray(q_goal_n, dtype=np.float32)[None, :], len(candidates_n), axis=0)
        tau = self.field_model.predict_travel_times(candidates_n, q_goals)
        goal_dist = np.linalg.norm(candidates_n - q_goal_n[None, :], axis=1)
        q_phys = np.asarray([self.kinematics.denormalize(qn) for qn in candidates_n], dtype=np.float32)
        clearances = checker.clearance_batch(q_phys)
        clearance_bonus = np.clip(clearances - clearance_margin_m, 0.0, 0.20)
        # The learned tau is still useful, but early online fields can have
        # local basins. Keep geometric goal progress dominant and use tau as a
        # secondary tie-breaker, otherwise rollouts can loop or move away from
        # reachable goals even when safe progress steps exist.
        score = 1.0 * goal_dist + 0.20 * tau - 0.25 * clearance_bonus
        finite = np.isfinite(score)
        if not np.any(finite):
            return None
        idx = int(np.argmin(np.where(finite, score, np.inf)))
        return candidates_n[idx].copy()

    def _ensure_endpoints(
        self,
        path: np.ndarray,
        q_start_n: np.ndarray,
        q_goal_n: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            pts = np.asarray([q_start_n, q_goal_n], dtype=np.float32)
        if np.linalg.norm(pts[0] - q_start_n) > 1e-5:
            pts = np.vstack((q_start_n.astype(np.float32), pts))
        if np.linalg.norm(pts[-1] - q_goal_n) > 1e-5:
            pts = np.vstack((pts, q_goal_n.astype(np.float32)))
        dedup = [pts[0]]
        for q in pts[1:]:
            if np.max(np.abs(q - dedup[-1])) > 1e-5:
                dedup.append(q)
        return np.asarray(dedup, dtype=np.float32)

    def _merge_bidirectional_paths(
        self,
        path_fwd: np.ndarray,
        path_rev: np.ndarray,
        q_start_n: np.ndarray,
        q_goal_n: np.ndarray,
        bridge_tol: float,
    ) -> np.ndarray:
        fwd = np.asarray(path_fwd, dtype=np.float32)
        rev = np.asarray(path_rev, dtype=np.float32)
        if fwd.ndim != 2 or len(fwd) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        if rev.ndim != 2 or len(rev) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        rev_forward = rev[::-1].copy()
        best_i = 0
        best_j = 0
        best_d = float("inf")
        for i in range(len(fwd)):
            diffs = rev_forward - fwd[i][None, :]
            dists = np.linalg.norm(diffs, axis=1)
            j = int(np.argmin(dists))
            d = float(dists[j])
            if d < best_d:
                best_d = d
                best_i = i
                best_j = j
        if best_d > max(1.0e-6, float(bridge_tol)):
            return np.zeros((0, 6), dtype=np.float32)

        meet_fwd = fwd[best_i].copy()
        meet_rev = rev_forward[best_j].copy()
        meet_mid = 0.5 * (meet_fwd + meet_rev)

        merged_parts = [fwd[:best_i].copy(), meet_mid[None, :], rev_forward[best_j + 1 :].copy()]
        merged = np.vstack([part for part in merged_parts if len(part) > 0])
        if len(merged) == 0:
            merged = np.asarray([q_start_n, q_goal_n], dtype=np.float32)
        if np.linalg.norm(merged[0] - q_start_n) > 1e-5:
            merged = np.vstack((q_start_n.astype(np.float32), merged))
        if np.linalg.norm(merged[-1] - q_goal_n) > 1e-5:
            merged = np.vstack((merged, q_goal_n.astype(np.float32)))
        dedup = [merged[0]]
        for q in merged[1:]:
            if np.max(np.abs(q - dedup[-1])) > 1e-5:
                dedup.append(q)
        return np.asarray(dedup, dtype=np.float32)


class JointSpaceRRTConnectPlanner:
    def __init__(self, kinematics: UR5Kinematics, rng: np.random.Generator | None = None):
        self.kinematics = kinematics
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_debug: dict[str, object] = {}
        self._edge_query_calls = 0
        self._edge_query_states = 0

    def _edge_is_valid(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_from: np.ndarray,
        q_to: np.ndarray,
        clearance_margin_m: float,
        max_delta_step: float,
    ) -> bool:
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        max_delta = float(np.max(np.abs(q_to - q_from)))
        nseg = max(1, int(np.ceil(max_delta / max(1.0e-3, max_delta_step))))
        pts = np.asarray(
            [q_from + (float(i) / float(nseg)) * (q_to - q_from) for i in range(nseg + 1)],
            dtype=np.float32,
        )
        clearances = checker.clearance_batch(pts)
        self._edge_query_calls += 1
        self._edge_query_states += int(len(pts))
        return bool(clearances.size > 0 and float(np.min(clearances)) >= clearance_margin_m)

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray, step_size_q: float) -> np.ndarray:
        delta = np.asarray(q_to, dtype=np.float64) - np.asarray(q_from, dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        if dist <= max(1.0e-9, step_size_q):
            return self.kinematics.clamp(np.asarray(q_to, dtype=np.float64))
        return self.kinematics.clamp(np.asarray(q_from, dtype=np.float64) + (step_size_q / dist) * delta)

    def _nearest_idx(self, nodes: list[np.ndarray], q: np.ndarray) -> int:
        dists = [float(np.linalg.norm(np.asarray(n, dtype=np.float64) - q)) for n in nodes]
        return int(np.argmin(np.asarray(dists, dtype=np.float64)))

    def _trace_path(self, nodes: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        cur = int(idx)
        while cur >= 0:
            out.append(np.asarray(nodes[cur], dtype=np.float64))
            cur = int(parents[cur])
        out.reverse()
        return out

    def _extend_tree(
        self,
        checker: UR5PointCloudCollisionChecker,
        nodes: list[np.ndarray],
        parents: list[int],
        q_target: np.ndarray,
        step_size_q: float,
        clearance_margin_m: float,
        edge_check_step_rad: float,
    ) -> tuple[int | None, str]:
        nearest = self._nearest_idx(nodes, q_target)
        q_near = nodes[nearest]
        q_new = self._steer(q_near, q_target, step_size_q)
        if np.max(np.abs(q_new - q_near)) < 1.0e-6:
            return None, "trapped"
        if not self._edge_is_valid(checker, q_near, q_new, clearance_margin_m, max_delta_step=edge_check_step_rad):
            return None, "trapped"
        nodes.append(q_new.copy())
        parents.append(nearest)
        if np.linalg.norm(q_new - q_target) <= step_size_q and self._edge_is_valid(
            checker, q_new, q_target, clearance_margin_m, max_delta_step=edge_check_step_rad
        ):
            nodes.append(np.asarray(q_target, dtype=np.float64).copy())
            parents.append(len(nodes) - 2)
            return len(nodes) - 1, "reached"
        return len(nodes) - 1, "advanced"

    def plan(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_iters: int = 4000,
        goal_bias: float = 0.2,
        clearance_margin_m: float = 0.01,
        edge_check_step_rad: float = 0.04,
    ) -> np.ndarray:
        self._edge_query_calls = 0
        self._edge_query_states = 0
        self.last_debug = {
            "planner": "rrt_connect",
            "status": "started",
            "iterations": 0,
            "edge_query_calls": 0,
            "edge_query_states": 0,
            "clearance_margin_m": float(clearance_margin_m),
            "edge_check_step_rad": float(edge_check_step_rad),
        }
        q_start = self.kinematics.clamp(np.asarray(q_start, dtype=np.float64))
        q_goal = self.kinematics.clamp(np.asarray(q_goal, dtype=np.float64))
        edge_check_step_rad = max(1.0e-3, float(edge_check_step_rad))
        if self._edge_is_valid(checker, q_start, q_goal, clearance_margin_m, max_delta_step=edge_check_step_rad):
            self.last_debug.update(
                {
                    "status": "direct_edge",
                    "edge_query_calls": self._edge_query_calls,
                    "edge_query_states": self._edge_query_states,
                    "start_tree_nodes": 1,
                    "goal_tree_nodes": 1,
                }
            )
            return np.asarray([q_start, q_goal], dtype=np.float32)

        start_nodes = [q_start.copy()]
        start_parents = [-1]
        goal_nodes = [q_goal.copy()]
        goal_parents = [-1]

        for it in range(max(1, int(max_iters))):
            grow_start = (it % 2) == 0
            tree_a, parents_a = (start_nodes, start_parents) if grow_start else (goal_nodes, goal_parents)
            tree_b, parents_b = (goal_nodes, goal_parents) if grow_start else (start_nodes, start_parents)

            if float(self.rng.uniform()) < float(goal_bias):
                q_rand = tree_b[0]
            else:
                q_rand = self.rng.uniform(self.kinematics.joint_min, self.kinematics.joint_max).astype(np.float64)
            idx_new, status = self._extend_tree(
                checker,
                tree_a,
                parents_a,
                q_rand,
                step_size_q=step_size_q,
                clearance_margin_m=clearance_margin_m,
                edge_check_step_rad=edge_check_step_rad,
            )
            if idx_new is None:
                continue
            q_new = tree_a[idx_new]
            while True:
                idx_other, status_other = self._extend_tree(
                    checker,
                    tree_b,
                    parents_b,
                    q_new,
                    step_size_q=step_size_q,
                    clearance_margin_m=clearance_margin_m,
                    edge_check_step_rad=edge_check_step_rad,
                )
                if idx_other is None:
                    break
                q_other = tree_b[idx_other]
                if np.linalg.norm(q_other - q_new) <= 1.0e-6:
                    if grow_start:
                        path_a = self._trace_path(start_nodes, start_parents, idx_new)
                        path_b = self._trace_path(goal_nodes, goal_parents, idx_other)
                        merged = path_a + list(reversed(path_b[:-1]))
                    else:
                        path_a = self._trace_path(start_nodes, start_parents, idx_other)
                        path_b = self._trace_path(goal_nodes, goal_parents, idx_new)
                        merged = path_a + list(reversed(path_b[:-1]))
                    self.last_debug.update(
                        {
                            "status": "reached",
                            "iterations": int(it + 1),
                            "edge_query_calls": self._edge_query_calls,
                            "edge_query_states": self._edge_query_states,
                            "start_tree_nodes": len(start_nodes),
                            "goal_tree_nodes": len(goal_nodes),
                        }
                    )
                    return np.asarray(merged, dtype=np.float32)
                if status_other != "advanced":
                    break
        self.last_debug.update(
            {
                "status": "max_iterations",
                "iterations": max(1, int(max_iters)),
                "edge_query_calls": self._edge_query_calls,
                "edge_query_states": self._edge_query_states,
                "start_tree_nodes": len(start_nodes),
                "goal_tree_nodes": len(goal_nodes),
            }
        )
        return np.zeros((0, 6), dtype=np.float32)
