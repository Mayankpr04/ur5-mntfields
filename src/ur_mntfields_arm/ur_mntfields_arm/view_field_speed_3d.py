from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.planner import ArmFieldPlanner
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


DEFAULT_START_Q = np.asarray([0.0, -2.74850, 1.50004, -1.71994, -1.57000, 0.03334], dtype=np.float64)
DEFAULT_GOALS_Q = np.asarray(
    [
        [0.0, -1.20804, 1.09161, -2.81853, -1.57000, 0.03334],
        [0.0, -0.83967, 1.61948, -4.01894, -1.57000, 0.03334],
        [-0.77152, -0.68967, 1.43704, -3.81446, -1.57000, 0.03334],
    ],
    dtype=np.float64,
)


def _latest_sample(samples_dir: Path) -> Path:
    files = sorted(samples_dir.glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files found in {samples_dir}")
    return files[-1]


def _sample_path(root: Path, step: str) -> Path:
    samples_dir = root / "samples"
    return _latest_sample(samples_dir) if step == "latest" else samples_dir / f"step_{step}.npz"


def _auto_checkpoint(root: Path) -> Path:
    final_path = root / "model" / "weights_final.pt"
    if final_path.exists():
        return final_path
    candidates = sorted((root / "model").glob("weights_epoch_*.pt"))
    if candidates:
        return candidates[-1]
    partial_path = root / "model" / "weights_partial.pt"
    if partial_path.exists():
        return partial_path
    raise FileNotFoundError(f"No checkpoint found under {root / 'model'}")


def _make_point_cloud(points: np.ndarray, colors: np.ndarray | tuple[float, float, float]):
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if isinstance(colors, tuple):
        cols = np.tile(np.asarray(colors, dtype=np.float64), (len(pts), 1))
    else:
        cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def _speed_colors(speed: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(speed, dtype=np.float32).reshape(-1), 0.0, 1.0)
    # Red/yellow means low-speed/danger; blue means high-speed/free.
    red = 1.0 - 0.15 * s
    green = 0.10 + 0.70 * np.minimum(s, 0.5) * 2.0
    blue = s
    return np.stack((red, green, blue), axis=1).astype(np.float64)


def _sample_joint_path(plan: np.ndarray, interp_step_rad: float) -> np.ndarray:
    pts = np.asarray(plan, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    if len(pts) == 1:
        return pts.copy()
    out: list[np.ndarray] = []
    for seg_idx, (qa, qb) in enumerate(zip(pts[:-1], pts[1:])):
        max_delta = float(np.max(np.abs(qb - qa)))
        nseg = max(1, int(np.ceil(max_delta / max(1.0e-3, interp_step_rad))))
        start_i = 0 if seg_idx == 0 else 1
        for i in range(start_i, nseg + 1):
            out.append(qa + (float(i) / float(nseg)) * (qb - qa))
    return np.asarray(out, dtype=np.float64)


def _tool_points(kin: UR5Kinematics, qs: np.ndarray) -> np.ndarray:
    return np.asarray([kin.fk(q)[:3, 3] for q in np.asarray(qs, dtype=np.float64)], dtype=np.float64)


def _line_set(points: np.ndarray, color: tuple[float, float, float]):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(pts)
    if len(pts) >= 2:
        lines = np.asarray([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
    else:
        lines = np.zeros((0, 2), dtype=np.int32)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return line


def _sphere(center: np.ndarray, radius: float, color: tuple[float, float, float]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=16)
    mesh.translate(np.asarray(center, dtype=np.float64))
    mesh.paint_uniform_color(color)
    return mesh


def _load_occupied(root: Path, sample_file: Path, data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "occupied_points" in data:
        return np.asarray(data["occupied_points"], dtype=np.float32).reshape(-1, 3)
    pcd_path = root / "PCD" / f"{sample_file.stem}_occupied_world.pcd"
    if pcd_path.exists():
        return np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def _replay_states(data: np.lib.npyio.NpzFile, max_states: int, seed: int) -> np.ndarray:
    rows = np.asarray(data["frame_data"], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 12 or len(rows) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    states = np.concatenate((rows[:, :6], rows[:, 6:12]), axis=0)
    if len(states) > max_states:
        rng = np.random.default_rng(seed)
        states = states[rng.choice(len(states), size=max_states, replace=False)]
    return np.clip(states, -0.5, 0.5).astype(np.float32)


def _predict_speeds_to_goal(
    model: ArmFieldModel,
    kin: UR5Kinematics,
    q_states: np.ndarray,
    q_goal: np.ndarray,
) -> np.ndarray:
    q_states = np.asarray(q_states, dtype=np.float32).reshape(-1, 6)
    q_goal_n = kin.normalize(q_goal).astype(np.float32)
    q_goals = np.repeat(q_goal_n[None, :], len(q_states), axis=0)
    pred, _ = model.predict_normalized_pair_speeds(q_states, q_goals, batch_size=4096)
    return np.clip(pred, 0.0, 1.5)


def _trajectory_states(
    planner: ArmFieldPlanner,
    start_q: np.ndarray,
    goals_q: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    sequence = [np.asarray(start_q, dtype=np.float64)]
    sequence.extend([np.asarray(g, dtype=np.float64) for g in goals_q])
    if args.return_to_first and len(goals_q) > 0:
        sequence.append(np.asarray(goals_q[0], dtype=np.float64))

    legs = []
    for leg_idx, (qa, qb) in enumerate(zip(sequence[:-1], sequence[1:]), start=1):
        if args.leg != 0 and args.leg != leg_idx:
            continue
        plan = planner.plan_learned_speed_search(
            qa,
            qb,
            args.step_size_q,
            args.rollout_max_steps,
            min_predicted_speed=args.min_predicted_speed,
            max_local_candidates=args.local_candidates,
            allow_direct_edge=not args.no_direct_edge,
            mode="bidirectional",
        )
        dense = _sample_joint_path(plan, args.tool_sample_step_rad)
        legs.append((dense, qb, dict(planner.last_debug)))
    return legs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View learned UR5 field predicted speeds in 3D. Red=low predicted speed, blue=high."
    )
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"), help="Training output root.")
    parser.add_argument("--step", default="latest", help="Step id like 000128 or 'latest'.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to weights_final/latest epoch.")
    parser.add_argument("--mode", choices=["replay", "trajectory", "both"], default="both")
    parser.add_argument("--target-goal", type=int, default=1, help="Goal index for replay-speed evaluation: 1, 2, or 3.")
    parser.add_argument("--max-states", type=int, default=8000, help="Max replay states to evaluate in 3D.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leg", type=int, default=0, help="Trajectory leg to show; 0 means all.")
    parser.add_argument("--return-to-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-size-q", type=float, default=0.03)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--min-predicted-speed", type=float, default=0.10)
    parser.add_argument("--local-candidates", type=int, default=32)
    parser.add_argument("--no-direct-edge", action="store_true", help="Force field rollout instead of direct valid edge.")
    parser.add_argument("--tool-sample-step-rad", type=float, default=0.035)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    if o3d is None:
        raise RuntimeError("open3d is required for 3D viewing.")

    root = args.root.expanduser().resolve()
    sample_file = _sample_path(root, args.step)
    data = np.load(sample_file)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else _auto_checkpoint(root)

    kin = UR5Kinematics()
    model = ArmFieldModel(str(root / "model"))
    model.load_checkpoint(checkpoint)
    occupied = _load_occupied(root, sample_file, data)
    evaluation_checker = UR5PointCloudCollisionChecker(kin, occupied)

    geoms = []
    if len(occupied):
        geoms.append(_make_point_cloud(occupied, (0.15, 0.85, 0.25)))

    if args.mode in ("replay", "both"):
        goal_idx = int(np.clip(args.target_goal, 1, len(DEFAULT_GOALS_Q))) - 1
        states_n = _replay_states(data, args.max_states, args.seed)
        speeds = _predict_speeds_to_goal(model, kin, states_n, DEFAULT_GOALS_Q[goal_idx])
        qs = np.asarray([kin.denormalize(qn) for qn in states_n], dtype=np.float64)
        pts = _tool_points(kin, qs)
        geoms.append(_make_point_cloud(pts, _speed_colors(speeds)))
        print(
            f"replay target_goal={goal_idx + 1} states={len(qs)} "
            f"pred_speed min={float(np.min(speeds)):.4f} mean={float(np.mean(speeds)):.4f} "
            f"max={float(np.max(speeds)):.4f} low<0.2={float(np.mean(speeds < 0.2)):.3f} high>0.8={float(np.mean(speeds > 0.8)):.3f}"
        )
        geoms.append(_sphere(kin.fk(DEFAULT_GOALS_Q[goal_idx])[:3, 3], 0.035, (0.0, 0.0, 0.0)))

    if args.mode in ("trajectory", "both"):
        planner = ArmFieldPlanner(model, kin)
        for leg_idx, (states_q, q_goal, debug) in enumerate(
            _trajectory_states(planner, DEFAULT_START_Q, DEFAULT_GOALS_Q, args),
            start=1,
        ):
            if len(states_q) == 0:
                print(f"trajectory leg={leg_idx}: planner returned empty path debug={debug}")
                continue
            states_n = np.asarray([kin.normalize(q) for q in states_q], dtype=np.float32)
            speeds = _predict_speeds_to_goal(model, kin, states_n, q_goal)
            geometric_clearance = evaluation_checker.clearance_batch(states_q.astype(np.float32))
            geometric_min = float(np.min(geometric_clearance)) if len(geometric_clearance) else float("nan")
            pts = _tool_points(kin, states_q)
            geoms.append(_make_point_cloud(pts, _speed_colors(speeds)))
            geoms.append(_line_set(pts, (0.05, 0.05, 0.05)))
            geoms.append(_sphere(pts[0], 0.025, (1.0, 0.8, 0.0)))
            geoms.append(_sphere(pts[-1], 0.025, (0.0, 0.0, 0.0)))
            print(
                f"trajectory leg={leg_idx} points={len(states_q)} planner_status={debug.get('status')} "
                f"pred_speed min={float(np.min(speeds)):.4f} mean={float(np.mean(speeds)):.4f} "
                f"max={float(np.max(speeds)):.4f} low<0.2={float(np.mean(speeds < 0.2)):.3f}"
                f" geometric_eval_min_clearance_m={geometric_min:.4f}"
            )

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    print(f"sample_file={sample_file}")
    print(f"checkpoint={checkpoint}")
    print("colors: occupied=green, predicted speed red/yellow=low -> blue=high, black spheres=goals")

    if not args.no_view:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="UR5 Field Predicted Speed")
        for geom in geoms:
            vis.add_geometry(geom)
        render = vis.get_render_option()
        render.point_size = float(args.point_size)
        render.background_color = np.asarray([1.0, 1.0, 1.0])
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
