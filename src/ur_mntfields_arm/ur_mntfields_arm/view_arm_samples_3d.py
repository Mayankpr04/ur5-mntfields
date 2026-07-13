from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


def _latest_sample(samples_dir: Path) -> Path:
    files = sorted(samples_dir.glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files found in {samples_dir}")
    return files[-1]


def _sample_path(root: Path, step: str) -> Path:
    samples_dir = root / "samples"
    return _latest_sample(samples_dir) if step == "latest" else samples_dir / f"step_{step}.npz"


def _pcd_path(root: Path, step_token: str, kind: str) -> Path:
    return root / "PCD" / f"{step_token}_{kind}_world.pcd"


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
) -> list:
    centers, radii = checker.robot_spheres(np.asarray(q, dtype=np.float64))
    if max_spheres > 0 and len(centers) > max_spheres:
        idx = np.linspace(0, len(centers) - 1, max_spheres).astype(int)
        centers = centers[idx]
        radii = radii[idx]
    geoms = []
    for center, radius in zip(centers, radii):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius) * radius_scale, resolution=8)
        mesh.translate(np.asarray(center, dtype=np.float64))
        mesh.paint_uniform_color(color)
        geoms.append(mesh)
    return geoms


def _speed_color(speed: float) -> tuple[float, float, float]:
    s = float(np.clip(speed, 0.0, 1.0))
    return (1.0 - s, 0.15, s)


def _selected_pairs(
    frame_data: np.ndarray,
    count: int,
    seed: int,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    rows = np.asarray(frame_data, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0 or count <= 0:
        return []
    if len(rows) > count:
        if mode == "low":
            idx = np.argsort(rows[:, 12])[:count]
        elif mode == "high":
            idx = np.argsort(rows[:, 12])[::-1][:count]
        elif mode == "mixed":
            low_count = max(1, count // 2)
            low_idx = np.argsort(rows[:, 12])[:low_count]
            high_idx = np.argsort(rows[:, 12])[::-1][: count - low_count]
            idx = np.unique(np.concatenate((low_idx, high_idx), axis=0))[:count]
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(rows), size=count, replace=False)
        rows = rows[idx]
    pairs = []
    kin = UR5Kinematics()
    for row in rows:
        q0 = kin.denormalize(row[:6])
        q1 = kin.denormalize(row[6:12])
        pairs.append((q0, q1, float(row[12]), float(row[13])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="View UR5 MNTFields samples with occupied PCD and sphere model overlays.")
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"), help="Training output root.")
    parser.add_argument("--step", default="latest", help="Step id like 000128 or 'latest'.")
    parser.add_argument("--show-depth", action="store_true", help="Also show step_*_depth_world.pcd when present.")
    parser.add_argument("--sample-pairs", type=int, default=3, help="Number of q0/q1 sample pairs to overlay.")
    parser.add_argument(
        "--select",
        choices=["random", "low", "high", "mixed"],
        default="random",
        help="Which sample pairs to overlay. Low/high sort by q0 speed0.",
    )
    parser.add_argument(
        "--color-by-speed",
        action="store_true",
        help="Color q0/q1 spheres by speed label instead of endpoint identity.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for choosing sample pairs.")
    parser.add_argument("--max-spheres", type=int, default=120, help="Max spheres per robot pose. 0 means all.")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Scale displayed UR5 sphere radii.")
    parser.add_argument("--no-view", action="store_true", help="Do not open Open3D viewer.")
    parser.add_argument("--save-ply", type=Path, default=None, help="Optional path to export occupied points as PLY.")
    args = parser.parse_args()

    if o3d is None:
        raise RuntimeError("open3d is required. Install/open the ROS env that has open3d available.")

    root = args.root.expanduser().resolve()
    sample_file = _sample_path(root, args.step)
    if not sample_file.exists():
        raise FileNotFoundError(sample_file)
    data = np.load(sample_file)
    step_token = sample_file.stem

    geoms = []
    if "occupied_points" in data:
        occupied = np.asarray(data["occupied_points"], dtype=np.float32)
        geoms.append(_make_point_cloud(occupied, (0.15, 0.85, 0.25)))
    else:
        occ_path = _pcd_path(root, step_token, "occupied")
        if occ_path.exists():
            occ = o3d.io.read_point_cloud(str(occ_path))
            occ.paint_uniform_color((0.15, 0.85, 0.25))
            geoms.append(occ)
            occupied = np.asarray(occ.points, dtype=np.float32)
        else:
            occupied = np.zeros((0, 3), dtype=np.float32)

    if args.show_depth:
        depth_path = _pcd_path(root, step_token, "depth")
        if depth_path.exists():
            depth = o3d.io.read_point_cloud(str(depth_path))
            depth.paint_uniform_color((0.20, 0.45, 1.0))
            geoms.append(depth)

    kin = UR5Kinematics()
    checker = UR5PointCloudCollisionChecker(kin, occupied)
    if "current_q" in data:
        geoms.extend(_sphere_geometries(checker, data["current_q"], (1.0, 0.8, 0.1), args.radius_scale, args.max_spheres))

    for q0, q1, speed0, speed1 in _selected_pairs(data["frame_data"], args.sample_pairs, args.seed, args.select):
        print(f"sample_pair speed0={speed0:.4f} speed1={speed1:.4f} q0={np.round(q0, 3).tolist()} q1={np.round(q1, 3).tolist()}")
        color0 = _speed_color(speed0) if args.color_by_speed else (1.0, 0.1, 0.1)
        color1 = _speed_color(speed1) if args.color_by_speed else (0.1, 0.4, 1.0)
        geoms.extend(_sphere_geometries(checker, q0, color0, args.radius_scale, args.max_spheres))
        geoms.extend(_sphere_geometries(checker, q1, color1, args.radius_scale, args.max_spheres))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    print(f"sample_file={sample_file}")
    print(f"occupied_points={len(occupied)}")
    if args.color_by_speed:
        print("colors: occupied=green current_q=yellow sample_spheres=red(low speed)->blue(high speed)")
    else:
        print("colors: occupied=green current_q=yellow q0=red q1=blue")

    if args.save_ply is not None and len(occupied):
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.save_ply), _make_point_cloud(occupied, (0.15, 0.85, 0.25)))
        print(f"wrote {args.save_ply}")

    if not args.no_view:
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
