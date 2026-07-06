from __future__ import annotations

import heapq

import numpy as np
import torch

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


class ArmFieldPlanner:
    def __init__(self, field_model: ArmFieldModel, kinematics: UR5Kinematics):
        self.field_model = field_model
        self.kinematics = kinematics
        self.last_debug: dict[str, object] = {}
        self.score_tau_weight = 0.55
        self.score_goal_dist_weight = 3.0
        self.score_depth_weight = 0.04
        self.score_clearance_weight = 0.0

    def set_score_weights(
        self,
        tau: float | None = None,
        goal_dist: float | None = None,
        depth: float | None = None,
        clearance: float | None = None,
    ) -> None:
        if tau is not None:
            self.score_tau_weight = float(tau)
        if goal_dist is not None:
            self.score_goal_dist_weight = float(goal_dist)
        if depth is not None:
            self.score_depth_weight = float(depth)
        if clearance is not None:
            self.score_clearance_weight = float(clearance)

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size_q: float,
        max_steps: int,
        mode: str = "bidirectional",
    ) -> np.ndarray:
        q_start_n = self.kinematics.normalize(q_start)
        q_goal_n = self.kinematics.normalize(q_goal)
        tol = max(0.01, 0.5 * float(step_size_q))
        path_fwd = self.field_model.gradient_rollout(
            q_start_n, q_goal_n, step_size=step_size_q, max_steps=max_steps, tol=tol
        )
        mode_norm = str(mode).strip().lower()
        if mode_norm in ("forward", "forward_only", "start_to_goal"):
            if not self._path_reached_goal(path_fwd, q_goal_n, tol):
                return np.zeros((0, 6), dtype=np.float32)
            path_n = self._ensure_endpoints(path_fwd, q_start_n, q_goal_n)
            return np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)
        if mode_norm not in ("bidirectional", "bi", "merge"):
            raise ValueError(f"Unsupported planner mode: {mode}")
        path_rev = self.field_model.gradient_rollout(
            q_goal_n, q_start_n, step_size=step_size_q, max_steps=max_steps, tol=tol
        )
        path_n = self._merge_bidirectional_paths(path_fwd, path_rev, q_start_n, q_goal_n, bridge_tol=max(2.0 * tol, 1.5 * float(step_size_q)))
        if path_n.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray([self.kinematics.denormalize(qn) for qn in path_n], dtype=np.float32)

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
    ) -> np.ndarray:
        path = self._plan_collision_aware_one_way(
            checker,
            q_start,
            q_goal,
            step_size_q,
            max_steps,
            clearance_margin_m=clearance_margin_m,
            max_local_candidates=max_local_candidates,
            search_label="forward",
            allow_direct_edge=allow_direct_edge,
        )
        forward_debug = dict(self.last_debug)
        if np.asarray(path).ndim == 2 and len(path) > 0:
            if shortcut_path:
                path = self._shortcut_collision_path(checker, path, clearance_margin_m, step_size_q)
            self.last_debug["search_direction"] = "forward"
            self.last_debug["shortcut_waypoints"] = int(len(path))
            self.last_debug["shortcut_enabled"] = bool(shortcut_path)
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
            if shortcut_path:
                path = self._shortcut_collision_path(checker, path, clearance_margin_m, step_size_q)
            self.last_debug = {
                "status": "reverse_reached",
                "search_direction": "reverse",
                "forward_debug": forward_debug,
                "reverse_debug": reverse_debug,
                "steps": reverse_debug.get("steps", 0),
                "start_goal_dist": forward_debug.get("start_goal_dist", reverse_debug.get("start_goal_dist", -1.0)),
                "last_goal_dist": 0.0,
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
                "raw_waypoints": int(len(reverse_path)),
                "shortcut_waypoints": int(len(path)),
                "shortcut_enabled": bool(shortcut_path),
            }
            return path

        self.last_debug = {
            "status": "failed_bidirectional_field_search",
            "search_direction": "bidirectional",
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
        return np.zeros((0, 6), dtype=np.float32)

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
        closed: set[int] = set()
        best_seen: dict[tuple[int, ...], float] = {self._search_key(q_start_n, step_size_q): 0.0}
        heap: list[tuple[float, int, int]] = [(self._field_search_priority(q_start_n, q_goal_n, 0, start_clearance, clearance_margin_m), 0, 0)]
        push_count = 1
        best_goal_dist = start_goal_dist
        best_idx = 0

        for expansion_idx in range(max_expansions):
            if not heap:
                self.last_debug.update({"status": "frontier_empty", "steps": int(expansion_idx), "last_goal_dist": best_goal_dist})
                break
            _, _, node_idx = heapq.heappop(heap)
            if node_idx in closed:
                continue
            closed.add(node_idx)
            q_cur_n = nodes[node_idx]
            node_goal_dist = float(np.linalg.norm(q_cur_n - q_goal_n))
            if node_goal_dist < best_goal_dist:
                best_goal_dist = node_goal_dist
                best_idx = node_idx
            self.last_debug["expanded_nodes"] = int(self.last_debug.get("expanded_nodes", 0)) + 1

            if node_goal_dist <= max(tol, 1.25 * float(step_size_q)):
                final_edge = self._edge_min_clearances(checker, q_cur_n, q_goal_n[None, :], float(step_size_q))
                if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                    goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), node_idx)
                    self.last_debug.update(
                        {
                            "status": "reached",
                            "steps": int(expansion_idx + 1),
                            "last_goal_dist": 0.0,
                            "best_edge_min_clearance": max(
                                float(self.last_debug.get("best_edge_min_clearance", -1.0)),
                                float(final_edge[0]),
                            ),
                        }
                    )
                    return self._search_path_to_array(nodes, parents, goal_idx)

            grad_np = self._field_gradient(q_cur_n, q_goal_n)
            candidates = self._local_step_candidates(q_cur_n, q_goal_n, grad_np, float(step_size_q), max_candidates)
            if len(candidates) == 0:
                continue
            edge_mins = self._edge_min_clearances(checker, q_cur_n, candidates, float(step_size_q))
            edge_ok = edge_mins >= float(clearance_margin_m)
            q_phys = np.asarray([self.kinematics.denormalize(qn) for qn in candidates], dtype=np.float32)
            clearances = checker.clearance_batch(q_phys)
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
            valid_goal_dists = goal_dists_all[edge_ok]
            valid_scores = self._field_search_scores(
                valid_candidates,
                q_goal_n,
                depths[node_idx] + 1,
                valid_clearances,
                clearance_margin_m,
            )
            if expansion_idx == 0:
                top_idx = np.argsort(valid_scores)[: min(8, len(valid_scores))]
                self.last_debug["first_expansion_top_candidates"] = [
                    {
                        "score": float(valid_scores[int(i)]),
                        "goal_dist": float(valid_goal_dists[int(i)]),
                        "clearance": float(valid_clearances[int(i)]),
                        "edge_min_clearance": float(edge_mins[edge_ok][int(i)]),
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
                score = float(valid_scores[int(local_i)])
                prev_score = best_seen.get(key)
                if prev_score is not None and score >= prev_score - 1.0e-6:
                    continue
                best_seen[key] = score
                child_idx = self._append_search_node(nodes, parents, depths, cand, node_idx)
                push_count += 1
                heapq.heappush(heap, (score, push_count, child_idx))
                cand_goal_dist = float(valid_goal_dists[int(local_i)])
                if cand_goal_dist < best_goal_dist:
                    best_goal_dist = cand_goal_dist
                    best_idx = child_idx
                if cand_goal_dist <= tol:
                    final_edge = self._edge_min_clearances(checker, cand, q_goal_n[None, :], float(step_size_q))
                    if final_edge.size > 0 and float(final_edge[0]) >= float(clearance_margin_m):
                        goal_idx = self._append_search_node(nodes, parents, depths, q_goal_n.copy(), child_idx)
                        self.last_debug.update({"status": "reached", "steps": int(expansion_idx + 1), "last_goal_dist": 0.0})
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

    def _field_gradient(self, q_cur_n: np.ndarray, q_goal_n: np.ndarray) -> np.ndarray:
        xp = torch.from_numpy(np.concatenate((q_cur_n, q_goal_n), axis=0)[None, :].astype(np.float32)).to(
            self.field_model.device
        )
        grad = self.field_model.model.function.Gradient(xp)
        return grad[0, :6].detach().cpu().numpy().astype(np.float32)

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
        q_goals = np.repeat(np.asarray(q_goal_n, dtype=np.float32)[None, :], len(candidates_n), axis=0)
        tau = self.field_model.predict_travel_times(candidates_n, q_goals)
        goal_dist = np.linalg.norm(candidates_n - q_goal_n[None, :], axis=1)
        clearance_bonus = np.clip(np.asarray(clearances, dtype=np.float32) - float(clearance_margin_m), 0.0, 0.20)
        # Best-first field search: the learned travel time ranks branches,
        # geometric distance prevents metric basins from dominating, and depth
        # keeps the search from preferring long wandering paths.
        score = (
            self.score_tau_weight * tau
            + self.score_goal_dist_weight * goal_dist
            + self.score_depth_weight * float(depth)
            - self.score_clearance_weight * clearance_bonus
        )
        return np.where(np.isfinite(score), score, np.inf).astype(np.float32)

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
            next_idx = cur + 1
            for cand_idx in range(len(pts) - 1, cur, -1):
                q_cur_n = self.kinematics.normalize(pts[cur])
                q_cand_n = self.kinematics.normalize(pts[cand_idx])
                edge = self._edge_min_clearances(checker, q_cur_n, q_cand_n[None, :], float(step_size_q))
                if edge.size > 0 and float(edge[0]) >= float(clearance_margin_m):
                    next_idx = cand_idx
                    break
            out.append(pts[next_idx].copy())
            cur = next_idx
        return np.asarray(out, dtype=np.float32)

    def _edge_min_clearances(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_cur_n: np.ndarray,
        candidates_n: np.ndarray,
        step_size_q: float,
    ) -> np.ndarray:
        q_cur = self.kinematics.denormalize(q_cur_n)
        max_delta_step = max(0.01, 0.5 * step_size_q)
        candidates_n = np.asarray(candidates_n, dtype=np.float32)
        if candidates_n.ndim == 1:
            candidates_n = candidates_n[None, :]
        if len(candidates_n) == 0:
            return np.zeros((0,), dtype=np.float32)
        q_cands = np.asarray([self.kinematics.denormalize(cand_n) for cand_n in candidates_n], dtype=np.float32)
        max_deltas = np.max(np.abs(q_cands - q_cur[None, :]), axis=1)
        nsegs = np.maximum(1, np.ceil(max_deltas / max_delta_step).astype(np.int32))
        edges = []
        edge_ids = []
        for edge_idx, (q_cand, nseg) in enumerate(zip(q_cands, nsegs)):
            alpha = (np.arange(int(nseg) + 1, dtype=np.float32) / float(nseg))[:, None]
            pts = q_cur[None, :] + alpha * (q_cand[None, :] - q_cur[None, :])
            edges.append(pts.astype(np.float32))
            edge_ids.extend([edge_idx] * len(pts))
        if not edges:
            return np.zeros((0,), dtype=np.float32)
        all_pts = np.vstack(edges).astype(np.float32)
        all_clearances = checker.clearance_batch(all_pts)
        out = np.full((len(q_cands),), np.inf, dtype=np.float32)
        for edge_idx, clearance in zip(edge_ids, all_clearances):
            out[int(edge_idx)] = min(float(out[int(edge_idx)]), float(clearance))
        return out.astype(np.float32)

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
    ) -> tuple[int | None, str]:
        nearest = self._nearest_idx(nodes, q_target)
        q_near = nodes[nearest]
        q_new = self._steer(q_near, q_target, step_size_q)
        if np.max(np.abs(q_new - q_near)) < 1.0e-6:
            return None, "trapped"
        if not self._edge_is_valid(checker, q_near, q_new, clearance_margin_m, max_delta_step=0.5 * step_size_q):
            return None, "trapped"
        nodes.append(q_new.copy())
        parents.append(nearest)
        if np.linalg.norm(q_new - q_target) <= step_size_q and self._edge_is_valid(
            checker, q_new, q_target, clearance_margin_m, max_delta_step=0.5 * step_size_q
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
    ) -> np.ndarray:
        q_start = self.kinematics.clamp(np.asarray(q_start, dtype=np.float64))
        q_goal = self.kinematics.clamp(np.asarray(q_goal, dtype=np.float64))
        if self._edge_is_valid(checker, q_start, q_goal, clearance_margin_m, max_delta_step=0.5 * step_size_q):
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
                checker, tree_a, parents_a, q_rand, step_size_q=step_size_q, clearance_margin_m=clearance_margin_m
            )
            if idx_new is None:
                continue
            q_new = tree_a[idx_new]
            while True:
                idx_other, status_other = self._extend_tree(
                    checker, tree_b, parents_b, q_new, step_size_q=step_size_q, clearance_margin_m=clearance_margin_m
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
                    return np.asarray(merged, dtype=np.float32)
                if status_other != "advanced":
                    break
        return np.zeros((0, 6), dtype=np.float32)
