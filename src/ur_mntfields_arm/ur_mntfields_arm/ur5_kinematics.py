from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import yaml


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _transform(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def look_at_rotation(origin: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    n = np.linalg.norm(forward)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    forward /= n
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    # ROS optical frame convention: +Z looks forward, +X right, +Y down.
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.column_stack((right, down, forward))


def pose_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    pos_err = desired[:3, 3] - current[:3, 3]
    r_err = desired[:3, :3] @ current[:3, :3].T
    skew = 0.5 * np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )
    return np.concatenate((pos_err, skew), axis=0)


@dataclass
class ViewGoal:
    frontier_id: int
    centroid: np.ndarray
    camera_pose: np.ndarray
    tool_pose: np.ndarray
    q_goal: np.ndarray
    score: float
    pose_kind: str = "local"
    visibility_score: float = 0.0
    local_coverage: float = 0.0
    gain_score: float = 0.0
    move_cost: float = 0.0
    clearance: float = 0.0


class UR5Kinematics:
    def __init__(self, ur_type: str = "ur5", description_root: str = "/home/mayank/ur_ws/src/Universal_Robots_ROS2_Description"):
        self.ur_type = ur_type
        self.description_root = Path(description_root)
        self.joint_min, self.joint_max = self._load_joint_limits(ur_type)
        # The standard UR DH chain is rooted in the UR "base" frame, while the
        # ROS stack publishes sensor/world geometry in "base_link". In the URDF,
        # base_link -> base is a fixed pi rotation about Z, so expose FK/IK in
        # base_link coordinates to match the rest of the exploration stack.
        self.base_link_from_dh = _transform(_rot_z(math.pi), np.zeros(3, dtype=np.float64))

        if ur_type.endswith("e"):
            self.d = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=np.float64)
            self.a = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.d = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823], dtype=np.float64)
            self.a = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0], dtype=np.float64)
        self.alpha = np.array([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0], dtype=np.float64)

    def _load_joint_limits(self, ur_type: str) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.description_root / "config" / ur_type / "joint_limits.yaml"
        text = cfg.read_text(encoding="utf-8")
        text = text.replace("!degrees", "")
        data = yaml.safe_load(text)
        mins = []
        maxs = []
        for name in JOINT_NAMES:
            entry = data["joint_limits"][name]
            mins.append(math.radians(float(entry["min_position"])))
            maxs.append(math.radians(float(entry["max_position"])))
        return np.asarray(mins, dtype=np.float64), np.asarray(maxs, dtype=np.float64)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_min, self.joint_max)

    def normalize(self, q: np.ndarray) -> np.ndarray:
        return ((q - self.joint_min) / (self.joint_max - self.joint_min) - 0.5).astype(np.float32)

    def denormalize(self, qn: np.ndarray) -> np.ndarray:
        return (self.joint_min + (np.asarray(qn, dtype=np.float64) + 0.5) * (self.joint_max - self.joint_min)).astype(np.float64)

    def fk_all(self, q: np.ndarray) -> list[np.ndarray]:
        q = self.clamp(np.asarray(q, dtype=np.float64))
        cur = self.base_link_from_dh.copy()
        out = [cur.copy()]
        for idx in range(6):
            cur = cur @ _dh(self.a[idx], self.d[idx], self.alpha[idx], q[idx])
            out.append(cur.copy())
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.fk_all(q)[-1]

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        base = self.fk(q)
        jac = np.zeros((6, 6), dtype=np.float64)
        for idx in range(6):
            pert = np.asarray(q, dtype=np.float64).copy()
            pert[idx] += eps
            diff = pose_error(base, self.fk(pert))
            jac[:, idx] = -diff / eps
        return jac

    def _solve_ik_local(self, desired_tool_pose: np.ndarray, seed: np.ndarray, max_iters: int = 120) -> np.ndarray | None:
        desired_tool_pose = np.asarray(desired_tool_pose, dtype=np.float64)

        def _accept(q_candidate: np.ndarray) -> np.ndarray | None:
            cur = self.fk(q_candidate)
            err = pose_error(cur, desired_tool_pose)
            if np.linalg.norm(err[:3]) < 0.03 and np.linalg.norm(err[3:]) < 0.20:
                return self.clamp(q_candidate)
            return None

        q = self.clamp(np.asarray(seed, dtype=np.float64).copy())
        lam = 1e-3
        for _ in range(max_iters):
            cur = self.fk(q)
            err = pose_error(cur, desired_tool_pose)
            accepted = _accept(q)
            if accepted is not None:
                return accepted
            jac = self.numerical_jacobian(q)
            lhs = jac.T @ jac + lam * np.eye(6, dtype=np.float64)
            step = np.linalg.solve(lhs, jac.T @ err)
            q = self.clamp(q + step)
        return None

    def _ik_restart_seeds_fast(self, seed: np.ndarray) -> list[np.ndarray]:
        mid = 0.5 * (self.joint_min + self.joint_max)
        seeds = [
            self.clamp(np.asarray(seed, dtype=np.float64)),
            mid.copy(),
            np.zeros(6, dtype=np.float64),
        ]
        for pan in (-1.2, -0.6, 0.0, 0.6, 1.2):
            guess = mid.copy()
            guess[0] = pan
            guess[1] = -1.4
            guess[2] = 1.7
            guess[3] = -1.8
            guess[4] = -1.57
            guess[5] = 0.0
            seeds.append(self.clamp(guess))
        return seeds

    def _ik_restart_seeds_full(self, seed: np.ndarray) -> list[np.ndarray]:
        seeds = self._ik_restart_seeds_fast(seed)
        # Cover both elbow-up and elbow-down families, plus wrist flips.
        shoulder_pans = (-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8)
        shoulder_lifts = (-1.9, -1.4, -0.9)
        elbow_configs = (1.7, -1.7)
        wrist1_configs = (-2.2, -1.8, 1.8, 2.2)
        wrist2_configs = (-1.57, 1.57)
        wrist3_configs = (0.0, np.pi, -np.pi)
        for pan in shoulder_pans:
            for lift in shoulder_lifts:
                for elbow in elbow_configs:
                    for wrist1 in wrist1_configs:
                        for wrist2 in wrist2_configs:
                            for wrist3 in wrist3_configs:
                                guess = np.array(
                                    [pan, lift, elbow, wrist1, wrist2, wrist3],
                                    dtype=np.float64,
                                )
                                seeds.append(self.clamp(guess))
        return seeds

    def _solve_ik_restarts(self, desired_tool_pose: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray | None:
        desired_tool_pose = np.asarray(desired_tool_pose, dtype=np.float64)

        def residual(qvec: np.ndarray) -> np.ndarray:
            err = pose_error(self.fk(qvec), desired_tool_pose)
            return np.concatenate((2.5 * err[:3], 0.35 * err[3:]), axis=0)

        def _accept(q_candidate: np.ndarray) -> np.ndarray | None:
            cur = self.fk(q_candidate)
            err = pose_error(cur, desired_tool_pose)
            if np.linalg.norm(err[:3]) < 0.03 and np.linalg.norm(err[3:]) < 0.20:
                return self.clamp(q_candidate)
            return None

        seen = set()
        for guess in seeds:
            q_key = tuple(np.round(np.asarray(guess, dtype=np.float64), 4).tolist())
            if q_key in seen:
                continue
            seen.add(q_key)
            result = least_squares(
                residual,
                guess,
                bounds=(self.joint_min, self.joint_max),
                max_nfev=250,
                xtol=1e-4,
                ftol=1e-4,
                gtol=1e-4,
            )
            if not result.success:
                continue
            accepted = _accept(result.x)
            if accepted is not None:
                return accepted
        return None

    def solve_ik_fast(self, desired_tool_pose: np.ndarray, seed: np.ndarray, max_iters: int = 120) -> np.ndarray | None:
        accepted = self._solve_ik_local(desired_tool_pose, seed, max_iters=max_iters)
        if accepted is not None:
            return accepted
        return self._solve_ik_restarts(desired_tool_pose, self._ik_restart_seeds_fast(seed))

    def solve_ik_full(self, desired_tool_pose: np.ndarray, seed: np.ndarray, max_iters: int = 120) -> np.ndarray | None:
        accepted = self._solve_ik_local(desired_tool_pose, seed, max_iters=max_iters)
        if accepted is not None:
            return accepted
        return self._solve_ik_restarts(desired_tool_pose, self._ik_restart_seeds_full(seed))

    def solve_ik(self, desired_tool_pose: np.ndarray, seed: np.ndarray, max_iters: int = 120) -> np.ndarray | None:
        return self.solve_ik_fast(desired_tool_pose, seed, max_iters=max_iters)

    def camera_to_tool_pose(self, camera_pose: np.ndarray, camera_in_tool: np.ndarray) -> np.ndarray:
        return camera_pose @ np.linalg.inv(camera_in_tool)

    def tool_to_camera_pose(self, tool_pose: np.ndarray, camera_in_tool: np.ndarray) -> np.ndarray:
        return tool_pose @ camera_in_tool
