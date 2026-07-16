from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


PLANNER_COLORS = {
    "field": (0.05, 0.65, 1.0),
    "field_collision": (1.0, 0.55, 0.05),
    "rrt_connect": (0.75, 0.15, 0.95),
}
BOX_COLOR = (0.55, 0.42, 0.25)


def _load_boxes(config: Path) -> np.ndarray:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    params = data.get("test_trained_field", {}).get("ros__parameters", {})
    rows = []
    for key in ("scene_boxes", "support_boxes"):
        for value in params.get(key, []):
            fields = value.split(",") if isinstance(value, str) else value
            row = [float(item) for item in fields]
            if len(row) == 6:
                rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


def _box_lines(box: np.ndarray):
    center, half = box[:3], 0.5 * box[3:]
    corners = np.asarray(
        [center + half * np.asarray([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.count_nonzero(np.abs(corners[i] - corners[j]) > 1.0e-9)) == 1:
                edges.append([i, j])
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32)),
    )
    lines.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(BOX_COLOR), (len(edges), 1)))
    return lines


def _path_lines(points: np.ndarray, color: tuple[float, float, float]):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    edges = np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points)))) if len(points) > 1 else np.zeros((0, 2), dtype=np.int32)
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(edges.astype(np.int32)),
    )
    lines.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(edges), 1)))
    return lines


def _sphere(position: np.ndarray, radius: float, color: tuple[float, float, float]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    mesh.translate(np.asarray(position, dtype=np.float64))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _robot_skeleton(kin: UR5Kinematics, q: np.ndarray, base_world: np.ndarray, color: tuple[float, float, float]):
    points = np.asarray([frame[:3, 3] + base_world for frame in kin.fk_all(q)], dtype=np.float64)
    return _path_lines(points, color)


def main() -> None:
    parser = argparse.ArgumentParser(description="View offline field/RRT benchmark paths without ROS or Gazebo.")
    parser.add_argument("results", type=Path, help="JSON emitted by the offline three-planner benchmark.")
    parser.add_argument("--case", default="lower_to_middle", help="Benchmark case to display, or 'all'.")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--planners", nargs="+", default=list(PLANNER_COLORS), choices=list(PLANNER_COLORS))
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=Path("/home/mayank/ur_ws/src/ur_mntfields_arm_sim/config/sim_scene.yaml"),
    )
    parser.add_argument("--base-world", type=float, nargs=3, default=(0.15, 0.35, 0.50))
    parser.add_argument("--show-robot-poses", action="store_true", help="Show tool-frame markers at every path waypoint.")
    args = parser.parse_args()
    if o3d is None:
        raise RuntimeError("open3d is required for this viewer")

    payload = json.loads(args.results.expanduser().read_text(encoding="utf-8"))
    selected = [
        row for row in payload["raw"]
        if int(row["repeat"]) == args.repeat
        and row["planner"] in args.planners
        and (args.case == "all" or row["case"] == args.case)
    ]
    if not selected:
        available = sorted({row["case"] for row in payload["raw"]})
        raise ValueError(f"No matching paths. Available cases: {available}")

    base_world = np.asarray(args.base_world, dtype=np.float64)
    kin = UR5Kinematics()
    geometries = [_box_lines(box) for box in _load_boxes(args.scene_config.expanduser())]
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20))

    endpoints: dict[str, np.ndarray] = {}
    for row in selected:
        q_path = np.asarray(row.get("path_q", []), dtype=np.float64).reshape(-1, 6)
        if len(q_path) == 0:
            continue
        tool_path = np.asarray([kin.fk(q)[:3, 3] + base_world for q in q_path], dtype=np.float64)
        color = (0.95, 0.05, 0.05) if not row["geometrically_safe"] else PLANNER_COLORS[row["planner"]]
        geometries.append(_path_lines(tool_path, color))
        endpoints.setdefault("start", tool_path[0])
        endpoints.setdefault("goal", tool_path[-1])
        if args.show_robot_poses:
            for q in q_path:
                geometries.append(_robot_skeleton(kin, q, base_world, color))
        print(
            f"{row['case']} {row['planner']}: generation={row['generation_ms']:.1f} ms "
            f"safe={row['geometrically_safe']} clearance={1000.0 * row['min_clearance_m']:.1f} mm "
            f"waypoints={row['waypoints']}"
        )

    if "start" in endpoints:
        geometries.append(_sphere(endpoints["start"], 0.035, (0.10, 0.85, 0.15)))
    if "goal" in endpoints:
        geometries.append(_sphere(endpoints["goal"], 0.035, (0.05, 0.05, 0.05)))
    anchor = payload.get("goals", {}).get("anchor", {}).get("tool_xyz_world")
    if anchor is not None:
        geometries.append(_sphere(np.asarray(anchor), 0.04, (1.0, 0.85, 0.05)))

    print("colors: field=blue, field_collision=orange, rrt_connect=purple, unsafe=red, start=green, goal=black, anchor=yellow")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Offline planner comparison: {args.case}",
        width=1400,
        height=900,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
