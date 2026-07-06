import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and view saved MNTFields RGB-D samples in 3D.")
    parser.add_argument("--samples-dir", type=Path, required=True, help="Directory containing samples/*.npz.")
    parser.add_argument(
        "--mode",
        choices=["raw", "pc", "surf"],
        default="raw",
        help="Which points to visualize: raw sample endpoints, sampled point cloud, or surface points.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of sample files loaded. 0 means all.")
    parser.add_argument("--max-points", type=int, default=200000, help="Randomly downsample to at most this many points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for downsampling.")
    parser.add_argument("--save-ply", type=Path, default=None, help="Optional path to save the aggregated cloud as PLY.")
    parser.add_argument("--no-view", action="store_true", help="Do not open the interactive Open3D viewer.")
    return parser.parse_args()


def load_points(samples_dir: Path, mode: str, max_files: int) -> tuple[np.ndarray, np.ndarray]:
    sample_files = sorted((samples_dir / "samples").glob("*.npz"))
    if max_files > 0:
        sample_files = sample_files[:max_files]
    if not sample_files:
        raise FileNotFoundError(f"No sample files found under {samples_dir / 'samples'}")

    all_points = []
    all_colors = []
    for path in sample_files:
        data = np.load(path)

        if mode == "raw":
            raw = data["raw_frame_data"].astype(np.float32)
            x0 = raw[:, 0:3]
            x1 = raw[:, 3:6]
            cp0 = raw[:, 6:9]
            cp1 = raw[:, 9:12]
            points = np.concatenate((x0, x1, cp0, cp1), axis=0)
            colors = np.concatenate(
                (
                    np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float32), (len(x0), 1)),
                    np.tile(np.array([[1.0, 0.6, 0.2]], dtype=np.float32), (len(x1), 1)),
                    np.tile(np.array([[0.2, 0.8, 1.0]], dtype=np.float32), (len(cp0), 1)),
                    np.tile(np.array([[0.1, 0.5, 1.0]], dtype=np.float32), (len(cp1), 1)),
                ),
                axis=0,
            )
        elif mode == "pc":
            points = data["pc_world"].astype(np.float32)
            colors = np.tile(np.array([[0.85, 0.85, 0.85]], dtype=np.float32), (len(points), 1))
        else:
            points = data["surf_pc_world"].astype(np.float32)
            colors = np.tile(np.array([[0.2, 0.9, 0.2]], dtype=np.float32), (len(points), 1))

        all_points.append(points)
        all_colors.append(colors)

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)


def downsample(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def main() -> None:
    args = parse_args()
    if o3d is None and not args.no_view:
        raise RuntimeError("open3d is required for interactive viewing.")

    samples_dir = args.samples_dir.expanduser().resolve()
    points, colors = load_points(samples_dir, args.mode, args.max_files)
    points, colors = downsample(points, colors, args.max_points, args.seed)

    print(f"[view-samples] mode={args.mode} points={len(points)}")
    print(f"[view-samples] bounds min={points.min(axis=0)} max={points.max(axis=0)}")

    if args.save_ply is not None or not args.no_view:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if args.save_ply is not None:
            save_ply = args.save_ply.expanduser().resolve()
            save_ply.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(save_ply), pcd)
            print(f"[view-samples] wrote {save_ply}")

        if not args.no_view:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            o3d.visualization.draw_geometries([pcd, frame])


if __name__ == "__main__":
    main()
