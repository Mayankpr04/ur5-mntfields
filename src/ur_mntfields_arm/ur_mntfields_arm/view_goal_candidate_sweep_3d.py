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
    _auto_checkpoint,
    _load_depth_points,
    _load_scene_boxes,
    _make_point_cloud,
    _sample_joint_path,
    _sample_path,
    _trajectory_curve_geometries,
)


POSE_COLORS = (
    (1.0, 0.12, 0.08),
    (0.10, 0.45, 1.0),
    (0.15, 0.80, 0.30),
    (0.95, 0.70, 0.08),
    (0.85, 0.25, 0.95),
)


def _sphere(center: np.ndarray, radius: float, color: tuple[float, float, float]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=24)
    mesh.translate(np.asarray(center, dtype=np.float64).reshape(3))
    mesh.paint_uniform_color(color)
    return mesh


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


def _tool_points(kin: UR5Kinematics, qs: np.ndarray) -> np.ndarray:
    return np.asarray([kin.fk(q)[:3, 3] for q in np.asarray(qs, dtype=np.float64)], dtype=np.float64)


def _frame_geometry(pose: np.ndarray, size: float):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size), origin=np.zeros(3))
    frame.transform(np.asarray(pose, dtype=np.float64).reshape(4, 4))
    return frame


def _box_geometry(
    pose: np.ndarray,
    center_local: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
):
    sx, sy, sz = (float(v) for v in size)
    mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    mesh.translate(np.asarray([-0.5 * sx, -0.5 * sy, -0.5 * sz], dtype=np.float64))
    mesh.translate(np.asarray(center_local, dtype=np.float64))
    mesh.transform(np.asarray(pose, dtype=np.float64).reshape(4, 4))
    mesh.paint_uniform_color(color)
    return mesh


def _end_effector_glyph(
    pose: np.ndarray,
    color: tuple[float, float, float],
    scale: float,
    finger_gap: float,
    finger_length: float,
):
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    s = float(scale)
    gap = float(finger_gap)
    length = float(finger_length)
    dark = tuple(max(0.0, 0.55 * c) for c in color)
    geoms = [
        # A compact palm block just behind the tool point.
        _box_geometry(pose, (-0.035 * s, 0.0, 0.0), (0.045 * s, 0.105 * s, 0.045 * s), dark),
        # Two parallel fingers. The target/tool point sits between the fingers near their tips.
        _box_geometry(pose, (0.025 * s, 0.5 * gap, 0.0), (length, 0.018 * s, 0.024 * s), color),
        _box_geometry(pose, (0.025 * s, -0.5 * gap, 0.0), (length, 0.018 * s, 0.024 * s), color),
        _box_geometry(pose, (0.085 * s, 0.5 * gap, 0.0), (0.020 * s, 0.026 * s, 0.032 * s), color),
        _box_geometry(pose, (0.085 * s, -0.5 * gap, 0.0), (0.020 * s, 0.026 * s, 0.032 * s), color),
    ]
    wrist = _segment_cylinder(
        pose[:3, 3] - 0.115 * s * pose[:3, 0],
        pose[:3, 3] - 0.055 * s * pose[:3, 0],
        dark,
        0.018 * s,
    )
    if wrist is not None:
        geoms.append(wrist)
    return geoms


def _tool_approach_geometry(
    pose: np.ndarray,
    color: tuple[float, float, float],
    length: float,
    radius: float,
):
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    origin = pose[:3, 3]
    x_axis = pose[:3, 0]
    shaft = _segment_cylinder(origin, origin + length * x_axis, color, radius)
    tip = _sphere(origin + length * x_axis, radius * 2.4, color)
    return [g for g in (shaft, tip) if g is not None]


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.asarray(a, dtype=np.float64).reshape(4, 4)[:3, :3]
    rb = np.asarray(b, dtype=np.float64).reshape(4, 4)[:3, :3]
    r = ra.T @ rb
    cos_angle = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _select_candidate_indices(data: np.lib.npyio.NpzFile, count: int, mode: str) -> list[int]:
    path_ok = np.asarray(data["path_ok"], dtype=bool)
    valid = np.flatnonzero(path_ok)
    if len(valid) == 0:
        valid = np.arange(len(data["q_goal"]))
    if len(valid) == 0:
        return []

    if mode == "first":
        return valid[:count].tolist()

    if mode == "shortest":
        order = valid[np.argsort(np.asarray(data["plan_len"], dtype=np.float64)[valid])]
        return order[:count].tolist()

    if mode == "clearance":
        order = valid[np.argsort(-np.asarray(data["path_clearance_m"], dtype=np.float64)[valid])]
        return order[:count].tolist()

    if mode == "indices":
        return valid[:count].tolist()

    poses = np.asarray(data["tool_pose"], dtype=np.float64)
    shortest = valid[int(np.argmin(np.asarray(data["plan_len"], dtype=np.float64)[valid]))]
    clearest = valid[int(np.argmax(np.asarray(data["path_clearance_m"], dtype=np.float64)[valid]))]
    selected = []
    for idx in (shortest, clearest):
        if int(idx) not in selected:
            selected.append(int(idx))
    while len(selected) < min(count, len(valid)):
        best = None
        best_score = -1.0
        for idx in valid:
            idx = int(idx)
            if idx in selected:
                continue
            min_angle = min(_rotation_angle(poses[idx], poses[j]) for j in selected) if selected else np.pi
            # Prefer orientation diversity, then clearance.
            score = min_angle + 0.25 * float(data["path_clearance_m"][idx])
            if score > best_score:
                best_score = score
                best = idx
        if best is None:
            break
        selected.append(best)
    return selected[:count]


def _parse_indices(text: str, n: int) -> list[int]:
    if not text.strip():
        return []
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 1 or idx > n:
            raise ValueError(f"Candidate index {idx} is outside 1..{n}")
        out.append(idx - 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a saved same-point end-effector candidate sweep as a few tool poses."
    )
    parser.add_argument("--sweep", type=Path, default=Path("/tmp/tool_point_grasp_sweep.npz"))
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"), help="Training output root used for scene/context.")
    parser.add_argument("--step", default="latest", help="Sample step id for scene/context, or 'latest'.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint used to recompute selected candidate trajectories.")
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=Path("src/ur_mntfields_arm_sim/config/sim_scene.yaml"),
        help="Config containing scene_boxes.",
    )
    parser.add_argument("--scene-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collision-cloud",
        choices=["scene_boxes", "occupied", "depth", "occupied_plus_boxes", "depth_plus_boxes"],
        default="depth_plus_boxes",
    )
    parser.add_argument("--clearance-backend", choices=["original", "sdf"], default="sdf")
    parser.add_argument("--sdf-voxel-size-m", type=float, default=0.04)
    parser.add_argument("--sdf-padding-m", type=float, default=0.75)
    parser.add_argument("--sdf-max-cells", type=int, default=4000000)
    parser.add_argument("--step-size-q", type=float, default=0.03)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--field-local-rollout-candidates", type=int, default=32)
    parser.add_argument("--collision-margin-m", type=float, default=0.01)
    parser.add_argument("--tool-sample-step-rad", type=float, default=0.03)
    parser.add_argument("--trajectory-radius", type=float, default=0.012)
    parser.add_argument("--show-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--show-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--count", type=int, default=3, help="Number of end-effector poses to show.")
    parser.add_argument(
        "--select",
        choices=["diverse", "shortest", "clearance", "first", "indices"],
        default="diverse",
        help="How to pick candidates from the sweep file.",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated 1-based candidate indices to show when --select indices is used.",
    )
    parser.add_argument("--frame-size", type=float, default=0.10)
    parser.add_argument("--goal-radius", type=float, default=0.025)
    parser.add_argument("--show-pose-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-pose-rays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-ee-glyph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ee-scale", type=float, default=1.0)
    parser.add_argument("--ee-finger-gap", type=float, default=0.065)
    parser.add_argument("--ee-finger-length", type=float, default=0.115)
    parser.add_argument("--approach-length", type=float, default=0.16, help="Length of the tool +X approach ray.")
    parser.add_argument("--approach-radius", type=float, default=0.006)
    parser.add_argument("--show-start", action="store_true", help="Also show the start tool frame if q_start is available.")
    parser.add_argument("--print-all", action="store_true", help="Print all candidate summary rows.")
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    if o3d is None:
        raise RuntimeError("open3d is required for 3D viewing.")

    data = np.load(args.sweep.expanduser(), allow_pickle=True)
    poses = np.asarray(data["tool_pose"], dtype=np.float64)
    n = len(poses)
    if n == 0:
        raise RuntimeError(f"No candidates found in {args.sweep}")

    if args.select == "indices":
        selected = _parse_indices(args.indices, n)
        if not selected:
            raise ValueError("--select indices requires --indices like '1,5,12'")
        selected = selected[: args.count]
    else:
        selected = _select_candidate_indices(data, max(1, int(args.count)), args.select)

    target = np.asarray(data["target"], dtype=np.float64).reshape(3)
    geoms = []

    kin = UR5Kinematics()
    root = args.root.expanduser().resolve()
    scene_boxes = np.zeros((0, 6), dtype=np.float64)
    occupied = np.zeros((0, 3), dtype=np.float32)
    depth_points = np.zeros((0, 3), dtype=np.float32)
    sample_file = None
    if args.show_scene or args.show_trajectories:
        sample_file = _sample_path(root, args.step)
        sample_data = np.load(sample_file)
        if "occupied_points" in sample_data:
            occupied = np.asarray(sample_data["occupied_points"], dtype=np.float32).reshape(-1, 3)
        depth_points = _load_depth_points(root, sample_file)
        scene_boxes = _load_scene_boxes(args.scene_config) if args.scene_boxes else np.zeros((0, 6), dtype=np.float64)

    if args.show_scene and len(occupied):
        geoms.append(_make_point_cloud(occupied, (0.15, 0.85, 0.25)))
    if args.show_depth and len(depth_points):
        geoms.append(_make_point_cloud(depth_points, (0.20, 0.45, 1.0)))

    geoms.extend(
        [
            _sphere(target, args.goal_radius, (0.05, 1.0, 0.10)),
            _frame_geometry(np.eye(4, dtype=np.float64), args.frame_size * 1.5),
        ]
    )

    planner = None
    if args.show_trajectories:
        checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else _auto_checkpoint(root)
        model = ArmFieldModel(model_dir=str(root / "model"), device="cuda:0")
        model.load_checkpoint(checkpoint)
        planner = ArmFieldPlanner(model, kin)

    def _collision_points() -> np.ndarray:
        if args.collision_cloud == "scene_boxes":
            return np.zeros((0, 3), dtype=np.float32)
        if args.collision_cloud in ("depth", "depth_plus_boxes"):
            return np.asarray(depth_points, dtype=np.float32)
        return np.asarray(occupied, dtype=np.float32)

    def _checker() -> UR5PointCloudCollisionChecker:
        boxes = (
            scene_boxes
            if args.collision_cloud in ("scene_boxes", "occupied_plus_boxes", "depth_plus_boxes")
            else np.zeros((0, 6), dtype=np.float64)
        )
        return make_ur5_collision_checker(
            kin,
            _collision_points(),
            box_obstacles=boxes,
            clearance_backend=args.clearance_backend,
            sdf_voxel_size_m=args.sdf_voxel_size_m,
            sdf_padding_m=args.sdf_padding_m,
            sdf_max_cells=args.sdf_max_cells,
        )

    if args.show_start and "q_start" in data.files:
        start_pose = kin.fk(np.asarray(data["q_start"], dtype=np.float64).reshape(6))
        if args.show_pose_frames:
            geoms.append(_frame_geometry(start_pose, args.frame_size))
        geoms.append(_sphere(start_pose[:3, 3], args.goal_radius * 0.65, (0.0, 0.0, 0.0)))

    checker = _checker() if args.show_trajectories else None

    for display_i, idx in enumerate(selected, start=1):
        color = POSE_COLORS[(display_i - 1) % len(POSE_COLORS)]
        pose = poses[idx]
        if args.show_trajectories and planner is not None and checker is not None:
            q_start = np.asarray(data["q_start"], dtype=np.float64).reshape(6)
            q_goal = np.asarray(data["q_goal"][idx], dtype=np.float64).reshape(6)
            plan = planner.plan_collision_aware(
                checker,
                q_start,
                q_goal,
                args.step_size_q,
                args.rollout_max_steps,
                clearance_margin_m=args.collision_margin_m,
                max_local_candidates=args.field_local_rollout_candidates,
                allow_direct_edge=True,
                shortcut_path=True,
            )
            dense = _sample_joint_path(plan, args.tool_sample_step_rad)
            tool_pts = _tool_points(kin, dense)
            geoms.extend(_trajectory_curve_geometries(tool_pts, color, args.trajectory_radius))
            print(
                f"trajectory[{display_i}] waypoints={len(plan)} sampled={len(tool_pts)} "
                f"planner_status={getattr(planner, 'last_debug', {}).get('status', '')}"
            )
        if args.show_pose_frames:
            geoms.append(_frame_geometry(pose, args.frame_size))
        if args.show_pose_rays:
            geoms.extend(_tool_approach_geometry(pose, color, args.approach_length, args.approach_radius))
        if args.show_ee_glyph:
            geoms.extend(
                _end_effector_glyph(
                    pose,
                    color,
                    args.ee_scale,
                    args.ee_finger_gap,
                    args.ee_finger_length,
                )
            )
        geoms.append(_sphere(pose[:3, 3], args.goal_radius * 0.45, color))
        print(
            f"shown[{display_i}] candidate={int(data['candidate_index'][idx])} "
            f"path_ok={bool(data['path_ok'][idx])} "
            f"q_goal={np.round(data['q_goal'][idx], 3).tolist()} "
            f"tool_xyz={np.round(pose[:3, 3], 4).tolist()} "
            f"plan_len={float(data['plan_len'][idx]):.3f} "
            f"path_clearance_m={float(data['path_clearance_m'][idx]):.4f} "
            f"field_speed_goal={float(data['field_speed_goal'][idx]):.4f} "
            f"planner_status={data['planner_status'][idx]}"
        )

    if args.print_all:
        print("all candidates:")
        for idx in range(n):
            print(
                f"  {int(data['candidate_index'][idx]):02d}: ok={bool(data['path_ok'][idx])} "
                f"plan_len={float(data['plan_len'][idx]):.3f} "
                f"clearance={float(data['path_clearance_m'][idx]):.4f} "
                f"field_goal={float(data['field_speed_goal'][idx]):.4f} "
                f"q={np.round(data['q_goal'][idx], 3).tolist()}"
            )

    if sample_file is not None:
        print(
            f"sample_file={sample_file} occupied_points={len(occupied)} depth_points={len(depth_points)} "
            f"scene_boxes={len(scene_boxes)} collision_cloud={args.collision_cloud}"
        )
    print(f"sweep={args.sweep} target_base_xyz={np.round(target, 4).tolist()} shown={len(selected)}/{n}")
    if not args.no_view:
        o3d.visualization.draw_geometries(geoms, window_name="UR5 same-point end-effector candidate sweep")


if __name__ == "__main__":
    main()
