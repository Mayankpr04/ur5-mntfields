from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker, make_ur5_collision_checker
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
LEG_COLORS = (
    (1.0, 0.12, 0.08),
    (0.10, 0.45, 1.0),
    (0.95, 0.70, 0.08),
    (0.15, 0.80, 0.30),
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
    if not candidates:
        partial = root / "model" / "weights_partial.pt"
        if partial.exists():
            return partial
        raise FileNotFoundError(f"No checkpoint found under {root / 'model'}")
    return candidates[-1]


def _load_scene_boxes(config_path: Path | None) -> np.ndarray:
    if config_path is None:
        return np.zeros((0, 6), dtype=np.float64)
    path = config_path.expanduser()
    if not path.exists():
        return np.zeros((0, 6), dtype=np.float64)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data.get("test_trained_field", {}).get("ros__parameters", {}) if isinstance(data, dict) else {}
    boxes = []
    for entry in params.get("scene_boxes", []):
        if isinstance(entry, str):
            values = [float(x) for x in entry.split(",") if x.strip()]
        else:
            values = [float(x) for x in entry]
        if len(values) == 6:
            boxes.append(values)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 6) if boxes else np.zeros((0, 6), dtype=np.float64)


def _load_depth_points(root: Path, sample_file: Path) -> np.ndarray:
    depth_path = root / "PCD" / f"{sample_file.stem}_depth_world.pcd"
    if not depth_path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    depth = o3d.io.read_point_cloud(str(depth_path))
    return np.asarray(depth.points, dtype=np.float32)


def _parse_q_list(values: list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or arr.size % 6 != 0:
        raise ValueError(f"{name} must contain a non-empty multiple of 6 values")
    return arr.reshape(-1, 6)


def _make_point_cloud(points: np.ndarray, color: tuple[float, float, float]):
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(pts), 1)))
    return pcd


def _sphere_geometries(
    checker: UR5PointCloudCollisionChecker,
    q: np.ndarray,
    color: tuple[float, float, float],
    radius_scale: float,
    max_spheres: int,
    alpha: float = 1.0,
) -> list:
    centers, radii = checker.robot_spheres(np.asarray(q, dtype=np.float64))
    if max_spheres > 0 and len(centers) > max_spheres:
        idx = np.linspace(0, len(centers) - 1, max_spheres).astype(int)
        centers = centers[idx]
        radii = radii[idx]
    geoms = []
    paint = tuple(float(c) for c in color)
    for center, radius in zip(centers, radii):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius) * radius_scale, resolution=8)
        mesh.translate(np.asarray(center, dtype=np.float64))
        mesh.paint_uniform_color(paint)
        geoms.append(mesh)
    return geoms


def _goal_marker(xyz: np.ndarray, color: tuple[float, float, float], radius: float):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=16)
    mesh.translate(np.asarray(xyz, dtype=np.float64))
    mesh.paint_uniform_color(color)
    return mesh


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


def _segment_cylinder(
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[float, float, float],
    radius: float,
):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1.0e-8:
        return None
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=length, resolution=16)
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    direction = delta / length
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if axis_norm > 1.0e-8:
        axis = axis / axis_norm
        angle = float(np.arccos(dot))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle), center=np.zeros(3))
    elif dot < 0.0:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([np.pi, 0.0, 0.0])), center=np.zeros(3))
    mesh.translate(0.5 * (a + b))
    mesh.paint_uniform_color(color)
    return mesh


def _trajectory_curve_geometries(points: np.ndarray, color: tuple[float, float, float], radius: float) -> list:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return []
    geoms = [_line_set(pts, color)]
    if radius <= 0.0:
        return geoms
    for a, b in zip(pts[:-1], pts[1:]):
        cyl = _segment_cylinder(a, b, color, radius)
        if cyl is not None:
            geoms.append(cyl)
    return geoms


def _edge_min_clearance(
    checker: UR5PointCloudCollisionChecker,
    plan: np.ndarray,
    interp_step_rad: float,
) -> tuple[bool, float]:
    pts = np.asarray(plan, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return False, -1.0
    if len(pts) == 1:
        return True, float(checker.clearance(pts[0]))
    mins = []
    for qa, qb in zip(pts[:-1], pts[1:]):
        max_delta = float(np.max(np.abs(qb - qa)))
        nseg = max(1, int(np.ceil(max_delta / max(1.0e-3, interp_step_rad))))
        edge = np.asarray([qa + (i / nseg) * (qb - qa) for i in range(nseg + 1)], dtype=np.float32)
        clearances = checker.clearance_batch(edge)
        mins.append(float(np.min(clearances)) if clearances.size else -1.0)
    return True, float(np.min(mins)) if mins else -1.0


def _sample_joint_path(plan: np.ndarray, interp_step_rad: float) -> np.ndarray:
    pts = np.asarray(plan, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    if len(pts) == 1:
        return pts.copy()
    out = []
    for seg_idx, (qa, qb) in enumerate(zip(pts[:-1], pts[1:])):
        max_delta = float(np.max(np.abs(qb - qa)))
        nseg = max(1, int(np.ceil(max_delta / max(1.0e-3, interp_step_rad))))
        start_i = 0 if seg_idx == 0 else 1
        for i in range(start_i, nseg + 1):
            out.append(qa + (float(i) / float(nseg)) * (qb - qa))
    return np.asarray(out, dtype=np.float64)


def _plan_length(plan: np.ndarray) -> float:
    pts = np.asarray(plan, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def _tool_path_length(tool_points: np.ndarray) -> float:
    pts = np.asarray(tool_points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def _trajectory_waypoint_indices(count: int, max_waypoints: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_waypoints <= 0 or count <= max_waypoints:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, max_waypoints).astype(np.int64))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visualization/test of learned UR5 field trajectories.")
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_urdf_fk"), help="Training output root.")
    parser.add_argument("--step", default="latest", help="Step id like 000129 or 'latest'.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to latest checkpoint under root/model.")
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=Path("src/ur_mntfields_arm_sim/config/sim_scene.yaml"),
        help="Config containing test_trained_field.scene_boxes. Use 'none' by passing --no-scene-boxes.",
    )
    parser.add_argument("--scene-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collision-cloud",
        choices=["scene_boxes", "occupied", "depth", "occupied_plus_boxes", "depth_plus_boxes"],
        default="scene_boxes",
        help="Geometry used for collision checking. Default avoids stale accumulated training-map points.",
    )
    parser.add_argument(
        "--self-filter-collision-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter robot-body points from occupied/depth collision clouds at each leg start.",
    )
    parser.add_argument("--start-q", type=float, nargs=6, default=DEFAULT_START_Q.tolist(), help="Initial joint state.")
    parser.add_argument(
        "--goals-q",
        type=float,
        nargs="+",
        default=DEFAULT_GOALS_Q.reshape(-1).tolist(),
        help="Flat list of unique goal joint states. Must be N*6 values.",
    )
    parser.add_argument("--return-to-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-size-q", type=float, default=0.03)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--field-local-rollout-candidates", type=int, default=32)
    parser.add_argument(
        "--planner-direct-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow planner to return [start, goal] when the direct joint edge is collision-free.",
    )
    parser.add_argument(
        "--planner-shortcut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shortcut sparse planner waypoints after field search.",
    )
    parser.add_argument("--clearance-backend", choices=["original", "sdf"], default="original")
    parser.add_argument("--sdf-voxel-size-m", type=float, default=0.04)
    parser.add_argument("--sdf-padding-m", type=float, default=0.75)
    parser.add_argument("--sdf-max-cells", type=int, default=4000000)
    parser.add_argument("--clearance-margin-m", type=float, default=0.01)
    parser.add_argument("--edge-check-step-rad", type=float, default=0.015)
    parser.add_argument(
        "--tool-sample-step-rad",
        type=float,
        default=0.03,
        help="Joint interpolation step used to render/report the actual FK tool curve between planner waypoints.",
    )
    parser.add_argument("--max-spheres", type=int, default=80, help="Max spheres per displayed robot pose. 0 means all.")
    parser.add_argument(
        "--show-arm-waypoints",
        action="store_true",
        help="Overlay UR5 sphere model at sampled trajectory waypoints. Disabled by default so trajectories appear as curves.",
    )
    parser.add_argument("--max-waypoints-per-leg", type=int, default=0, help="Max visualized robot poses per leg when --show-arm-waypoints is set. 0 means all.")
    parser.add_argument("--radius-scale", type=float, default=0.9)
    parser.add_argument("--trajectory-radius", type=float, default=0.012, help="Rendered trajectory curve radius in meters. Use 0 for thin Open3D lines.")
    parser.add_argument(
        "--show-direct-tool-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay grey straight tool-position lines between leg endpoints for comparison.",
    )
    parser.add_argument("--direct-line-radius", type=float, default=0.004, help="Radius for direct endpoint tool-line overlays.")
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--save-ply", type=Path, default=None, help="Optional path to export the combined visualization point cloud.")
    args = parser.parse_args()

    if o3d is None:
        raise RuntimeError("open3d is required. Install/open the ROS env that has open3d available.")

    root = args.root.expanduser().resolve()
    sample_file = _sample_path(root, args.step)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else _auto_checkpoint(root)
    data = np.load(sample_file)
    if "occupied_points" not in data:
        raise KeyError(f"{sample_file} does not contain occupied_points")
    occupied = np.asarray(data["occupied_points"], dtype=np.float32)
    depth_points = _load_depth_points(root, sample_file)
    scene_boxes = _load_scene_boxes(args.scene_config) if args.scene_boxes else np.zeros((0, 6), dtype=np.float64)

    kin = UR5Kinematics()
    display_checker = UR5PointCloudCollisionChecker(kin, occupied, box_obstacles=scene_boxes)
    model = ArmFieldModel(model_dir=str(root / "model"), device="cuda:0")
    model.load_checkpoint(checkpoint)
    planner = ArmFieldPlanner(model, kin)

    def _collision_points_for_leg(q_start: np.ndarray) -> np.ndarray:
        if args.collision_cloud in ("scene_boxes",):
            return np.zeros((0, 3), dtype=np.float32)
        if args.collision_cloud in ("depth", "depth_plus_boxes"):
            points = depth_points.copy()
        else:
            points = occupied.copy()
        if args.self_filter_collision_cloud and len(points):
            filter_checker = UR5PointCloudCollisionChecker(kin, np.zeros((0, 3), dtype=np.float32), box_obstacles=np.zeros((0, 6), dtype=np.float64))
            points, removed = filter_checker.filter_robot_self_points(points, q_start, padding_m=0.04)
            if removed:
                print(f"self_filter_collision_cloud removed={removed} start_q={np.round(q_start, 3).tolist()}")
        return np.asarray(points, dtype=np.float32)

    def _checker_for_leg(q_start: np.ndarray) -> UR5PointCloudCollisionChecker:
        boxes = scene_boxes if args.collision_cloud in ("scene_boxes", "occupied_plus_boxes", "depth_plus_boxes") else np.zeros((0, 6), dtype=np.float64)
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

    for idx, q in enumerate(goals_unique):
        color = LEG_COLORS[idx % len(LEG_COLORS)]
        geoms.append(_goal_marker(kin.fk(q)[:3, 3], color, 0.055))
        geoms.extend(_sphere_geometries(display_checker, q, color, args.radius_scale, args.max_spheres))

    current = np.asarray(args.start_q, dtype=np.float64)
    geoms.extend(_sphere_geometries(display_checker, current, (1.0, 0.8, 0.1), args.radius_scale, args.max_spheres))
    all_tool_points = []
    all_visual_points = [occupied]
    print(f"sample_file={sample_file}")
    print(f"checkpoint={checkpoint}")
    print(f"clearance_backend={args.clearance_backend}")
    print(f"collision_cloud={args.collision_cloud} scene_boxes={len(scene_boxes)} occupied_points={len(occupied)} depth_points={len(depth_points)}")
    print(f"unique_goals={len(goals_unique)} trajectory_legs={len(sequence)} return_to_first={args.return_to_first}")
    for leg_idx, q_goal in enumerate(sequence, start=1):
        color = LEG_COLORS[(leg_idx - 1) % len(LEG_COLORS)]
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
        direct_plan = np.asarray([current, q_goal], dtype=np.float64)
        _, direct_joint_min_clearance = _edge_min_clearance(checker, direct_plan, args.edge_check_step_rad)
        start_tool = kin.fk(current)[:3, 3]
        goal_tool = kin.fk(q_goal)[:3, 3]
        direct_tool_distance = float(np.linalg.norm(goal_tool - start_tool))
        debug = dict(planner.last_debug)
        if len(plan):
            sparse_tool_points = np.asarray([kin.fk(q)[:3, 3] for q in plan], dtype=np.float64)
            sampled_plan = _sample_joint_path(plan, args.tool_sample_step_rad)
            tool_points_for_stats = np.asarray([kin.fk(q)[:3, 3] for q in sampled_plan], dtype=np.float64)
            sparse_tool_len = _tool_path_length(sparse_tool_points)
            tool_len = _tool_path_length(tool_points_for_stats)
        else:
            sparse_tool_points = np.zeros((0, 3), dtype=np.float64)
            sampled_plan = np.zeros((0, 6), dtype=np.float64)
            tool_points_for_stats = np.zeros((0, 3), dtype=np.float64)
            sparse_tool_len = 0.0
            tool_len = 0.0
        tool_ratio = tool_len / max(1.0e-9, direct_tool_distance)
        sparse_tool_ratio = sparse_tool_len / max(1.0e-9, direct_tool_distance)
        print(
            f"leg={leg_idx}/{len(sequence)} start_q={np.round(current, 3).tolist()} "
            f"goal_q={np.round(q_goal, 3).tolist()} waypoints={len(plan)} "
            f"sampled_waypoints={len(sampled_plan)} joint_len={_plan_length(plan):.3f} "
            f"sparse_tool_len_m={sparse_tool_len:.3f} sparse_tool_len_ratio={sparse_tool_ratio:.2f} "
            f"tool_len_m={tool_len:.3f} tool_direct_m={direct_tool_distance:.3f} tool_len_ratio={tool_ratio:.2f} "
            f"path_min_clearance_m={min_clearance:.4f} direct_joint_min_clearance_m={direct_joint_min_clearance:.4f} "
            f"planner_status={debug.get('status')} "
            f"search_direction={debug.get('search_direction')} debug={debug}"
        )
        if len(plan) == 0:
            break
        tool_points = tool_points_for_stats
        all_tool_points.append(tool_points)
        if args.show_direct_tool_lines:
            direct_color = (0.68, 0.68, 0.68) if direct_joint_min_clearance >= args.clearance_margin_m else (0.15, 0.15, 0.15)
            geoms.extend(
                _trajectory_curve_geometries(
                    np.asarray([start_tool, goal_tool], dtype=np.float64),
                    direct_color,
                    args.direct_line_radius,
                )
            )
        geoms.extend(_trajectory_curve_geometries(tool_points, color, args.trajectory_radius))
        if args.show_arm_waypoints:
            for wp_idx in _trajectory_waypoint_indices(len(plan), args.max_waypoints_per_leg):
                geoms.extend(_sphere_geometries(display_checker, plan[int(wp_idx)], color, args.radius_scale, args.max_spheres))
        current = np.asarray(q_goal, dtype=np.float64)

    if all_tool_points:
        all_visual_points.extend(all_tool_points)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    print(
        "colors: occupied=green start=yellow goal_arms/trajectory_lines=red,blue,yellow-green,green "
        "direct_tool_lines=grey(if direct joint edge clears)/dark(if direct joint edge collides)"
    )

    if args.save_ply is not None:
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        pts = np.vstack([np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in all_visual_points])
        o3d.io.write_point_cloud(str(args.save_ply), _make_point_cloud(pts, (0.8, 0.8, 0.8)))
        print(f"wrote {args.save_ply}")

    if not args.no_view:
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
