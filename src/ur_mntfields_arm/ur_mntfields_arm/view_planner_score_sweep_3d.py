from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker, make_ur5_collision_checker
from ur_mntfields_arm.planner import ArmFieldPlanner
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics
from ur_mntfields_arm.view_field_trajectories_3d import (
    DEFAULT_GOALS_Q,
    DEFAULT_START_Q,
    _auto_checkpoint,
    _edge_min_clearance,
    _load_depth_points,
    _load_scene_boxes,
    _make_point_cloud,
    _parse_q_list,
    _plan_length,
    _sample_joint_path,
    _sample_path,
    _sphere_geometries,
    _tool_path_length,
    _trajectory_curve_geometries,
)


COLOR_TABLE: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.10, 0.05),
    "blue": (0.10, 0.35, 1.0),
    "green": (0.10, 0.80, 0.25),
    "yellow": (1.0, 0.75, 0.05),
    "cyan": (0.0, 0.85, 0.95),
    "magenta": (0.90, 0.10, 0.90),
    "orange": (1.0, 0.45, 0.05),
    "purple": (0.45, 0.20, 0.85),
    "black": (0.02, 0.02, 0.02),
}


DEFAULT_SWEEPS = (
    "red:0.55,3.0,0.04,0.0;"
    "blue:0.10,3.0,0.04,0.0;"
    "green:2.00,3.0,0.04,0.0;"
    "orange:0.55,0.35,0.04,0.0;"
    "purple:0.55,6.0,0.04,0.0;"
    "cyan:0.55,3.0,0.04,0.20"
)


def _parse_sweeps(spec: str) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float, float]]]:
    out = []
    auto_colors = list(COLOR_TABLE.items())
    for idx, chunk in enumerate(str(spec).split(";")):
        item = chunk.strip()
        if not item:
            continue
        if ":" in item:
            label, values_text = item.split(":", 1)
            label = label.strip()
        else:
            label = auto_colors[idx % len(auto_colors)][0]
            values_text = item
        values = [float(v.strip()) for v in values_text.split(",") if v.strip()]
        if len(values) != 4:
            raise ValueError(
                f"Invalid sweep '{item}'. Expected label:tau,goal_dist,depth,clearance, "
                "for example red:0.55,3.0,0.04,0.0"
            )
        color = COLOR_TABLE.get(label.lower(), auto_colors[idx % len(auto_colors)][1])
        out.append((label, color, (values[0], values[1], values[2], values[3])))
    if not out:
        raise ValueError("No score sweeps configured.")
    return out


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


def _goal_marker(xyz: np.ndarray, color: tuple[float, float, float], radius: float):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=16)
    mesh.translate(np.asarray(xyz, dtype=np.float64))
    mesh.paint_uniform_color(color)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ArmFieldPlanner score weights and overlay generated tool trajectories. "
            "Only the start-pose robot model is displayed."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"), help="Training output root.")
    parser.add_argument("--step", default="latest", help="Step id like 000129 or 'latest'.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to latest checkpoint under root/model.")
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=Path("src/ur_mntfields_arm_sim/config/sim_scene.yaml"),
        help="Config containing test_trained_field.scene_boxes.",
    )
    parser.add_argument("--scene-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collision-cloud",
        choices=["scene_boxes", "occupied", "depth", "occupied_plus_boxes", "depth_plus_boxes"],
        default="scene_boxes",
    )
    parser.add_argument("--self-filter-collision-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-q", type=float, nargs=6, default=DEFAULT_START_Q.tolist())
    parser.add_argument("--goals-q", type=float, nargs="+", default=DEFAULT_GOALS_Q.reshape(-1).tolist())
    parser.add_argument("--return-to-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leg", type=int, default=0, help="Show only one leg; 0 means all legs.")
    parser.add_argument("--sweeps", default=DEFAULT_SWEEPS, help="Semicolon list: color:tau,goal_dist,depth,clearance.")
    parser.add_argument("--step-size-q", type=float, default=0.03)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--field-local-rollout-candidates", type=int, default=32)
    parser.add_argument("--planner-direct-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planner-shortcut", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clearance-backend", choices=["original", "sdf"], default="sdf")
    parser.add_argument("--sdf-voxel-size-m", type=float, default=0.04)
    parser.add_argument("--sdf-padding-m", type=float, default=0.75)
    parser.add_argument("--sdf-max-cells", type=int, default=4000000)
    parser.add_argument("--clearance-margin-m", type=float, default=0.01)
    parser.add_argument("--edge-check-step-rad", type=float, default=0.015)
    parser.add_argument("--tool-sample-step-rad", type=float, default=0.03)
    parser.add_argument("--max-spheres", type=int, default=80)
    parser.add_argument("--radius-scale", type=float, default=0.9)
    parser.add_argument("--trajectory-radius", type=float, default=0.012)
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    if o3d is None:
        raise RuntimeError("open3d is required. Install/open the ROS env that has open3d available.")

    root = args.root.expanduser().resolve()
    sample_file = _sample_path(root, args.step)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else _auto_checkpoint(root)
    data = np.load(sample_file)
    occupied = np.asarray(data["occupied_points"], dtype=np.float32)
    depth_points = _load_depth_points(root, sample_file)
    scene_boxes = _load_scene_boxes(args.scene_config) if args.scene_boxes else np.zeros((0, 6), dtype=np.float64)
    sweeps = _parse_sweeps(args.sweeps)

    kin = UR5Kinematics()
    display_checker = UR5PointCloudCollisionChecker(kin, occupied, box_obstacles=scene_boxes)
    model = ArmFieldModel(model_dir=str(root / "model"), device="cuda:0")
    model.load_checkpoint(checkpoint)
    planner = ArmFieldPlanner(model, kin)

    def _collision_points_for_leg(q_start: np.ndarray) -> np.ndarray:
        if args.collision_cloud == "scene_boxes":
            return np.zeros((0, 3), dtype=np.float32)
        points = depth_points.copy() if args.collision_cloud in ("depth", "depth_plus_boxes") else occupied.copy()
        if args.self_filter_collision_cloud and len(points):
            filter_checker = UR5PointCloudCollisionChecker(
                kin,
                np.zeros((0, 3), dtype=np.float32),
                box_obstacles=np.zeros((0, 6), dtype=np.float64),
            )
            points, removed = filter_checker.filter_robot_self_points(points, q_start, padding_m=0.04)
            if removed:
                print(f"self_filter_collision_cloud removed={removed} start_q={np.round(q_start, 3).tolist()}")
        return np.asarray(points, dtype=np.float32)

    def _checker_for_leg(q_start: np.ndarray):
        boxes = (
            scene_boxes
            if args.collision_cloud in ("scene_boxes", "occupied_plus_boxes", "depth_plus_boxes")
            else np.zeros((0, 6), dtype=np.float64)
        )
        return make_ur5_collision_checker(
            kin,
            _collision_points_for_leg(q_start),
            box_obstacles=boxes,
            clearance_backend=args.clearance_backend,
            sdf_voxel_size_m=args.sdf_voxel_size_m,
            sdf_padding_m=args.sdf_padding_m,
            sdf_max_cells=args.sdf_max_cells,
        )

    goals_unique = _parse_q_list(args.goals_q, "--goals-q")
    sequence = list(goals_unique)
    if args.return_to_first and len(goals_unique) > 0:
        sequence.append(goals_unique[0].copy())

    geoms = [_make_point_cloud(occupied, (0.15, 0.85, 0.25))]
    if args.show_depth:
        depth_path = root / "PCD" / f"{sample_file.stem}_depth_world.pcd"
        if depth_path.exists():
            depth = o3d.io.read_point_cloud(str(depth_path))
            depth.paint_uniform_color((0.20, 0.45, 1.0))
            geoms.append(depth)

    start_q = np.asarray(args.start_q, dtype=np.float64)
    geoms.extend(_sphere_geometries(display_checker, start_q, (1.0, 0.8, 0.1), args.radius_scale, args.max_spheres))
    for goal_idx, q_goal in enumerate(goals_unique, start=1):
        geoms.append(_goal_marker(kin.fk(q_goal)[:3, 3], (0.02, 0.02, 0.02), 0.04))
        print(f"goal_marker={goal_idx} tool_xyz={np.round(kin.fk(q_goal)[:3, 3], 3).tolist()}")

    print(f"sample_file={sample_file}")
    print(f"checkpoint={checkpoint}")
    print(f"collision_cloud={args.collision_cloud} clearance_backend={args.clearance_backend} scene_boxes={len(scene_boxes)}")
    print("score = tau_w * tau + goal_w * goal_dist + depth_w * depth - clearance_w * clearance_bonus")
    print("colors:")
    for label, color, weights in sweeps:
        print(f"  {label}: rgb={tuple(round(c, 3) for c in color)} weights={weights}")

    for label, color, weights in sweeps:
        planner.set_score_weights(tau=weights[0], goal_dist=weights[1], depth=weights[2], clearance=weights[3])
        current = start_q.copy()
        for leg_idx, q_goal in enumerate(sequence, start=1):
            if args.leg != 0 and leg_idx != args.leg:
                current = q_goal.copy()
                continue
            checker = _checker_for_leg(current)
            plan = planner.plan_collision_aware(
                checker,
                current,
                q_goal,
                args.step_size_q,
                args.rollout_max_steps,
                clearance_margin_m=args.clearance_margin_m,
                max_local_candidates=args.field_local_rollout_candidates,
                allow_direct_edge=args.planner_direct_edge,
                shortcut_path=args.planner_shortcut,
            )
            _, min_clearance = _edge_min_clearance(checker, plan, args.edge_check_step_rad)
            sampled = _sample_joint_path(plan, args.tool_sample_step_rad)
            tool_points = np.asarray([kin.fk(q)[:3, 3] for q in sampled], dtype=np.float64) if len(sampled) else np.zeros((0, 3))
            if len(tool_points):
                geoms.extend(_trajectory_curve_geometries(tool_points, color, args.trajectory_radius))
            debug = dict(planner.last_debug)
            start_tool = kin.fk(current)[:3, 3]
            goal_tool = kin.fk(q_goal)[:3, 3]
            direct_tool = float(np.linalg.norm(goal_tool - start_tool))
            tool_len = _tool_path_length(tool_points)
            print(
                f"sweep={label} leg={leg_idx}/{len(sequence)} status={debug.get('status')} "
                f"waypoints={len(plan)} sampled={len(sampled)} joint_len={_plan_length(plan):.3f} "
                f"tool_len_m={tool_len:.3f} tool_direct_m={direct_tool:.3f} "
                f"tool_ratio={tool_len / max(1.0e-9, direct_tool):.2f} "
                f"min_clearance_m={min_clearance:.4f} steps={debug.get('steps')} "
                f"valid_edges={debug.get('valid_edge_count')}/{debug.get('candidate_count')}"
            )
            current = q_goal.copy()

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    if not args.no_view:
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
