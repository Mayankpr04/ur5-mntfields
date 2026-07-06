import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .sampling import calculate_speed_and_normal, compute_normalization_bound
except ImportError:
    from sampling import calculate_speed_and_normal, compute_normalization_bound

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from skimage import measure
except ImportError:
    measure = None


DEFAULT_MNTFIELDS_SRC = Path("/home/mayank/mntfields/src/mntfields")
MODEL_SUBDIR = "trained_field"
FIELD_CMAP_NAME = "gist_heat"


def ensure_mntfields_importable(mntfields_src: Path) -> None:
    if str(mntfields_src) not in sys.path:
        sys.path.insert(0, str(mntfields_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline MNTFields model from saved RGB-D samples.")
    parser.add_argument("--samples-dir", type=Path, required=True, help="Directory containing samples/*.npz from ur_mntfields_rgbd.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save the trained model and visualizations.")
    parser.add_argument("--mntfields-src", type=Path, default=DEFAULT_MNTFIELDS_SRC, help="Path to the local mntfields Python package root.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for training.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of train_core iterations.")
    parser.add_argument("--max-train-rows", type=int, default=30000, help="Max rows to use per training epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the model.")
    parser.add_argument("--bound-padding-xy", type=float, default=0.05, help="Padding added to x/y bounds.")
    parser.add_argument("--bound-padding-z", type=float, default=0.05, help="Padding added to z bounds.")
    parser.add_argument("--sample-min", type=float, default=0.07, help="Minimum clearance used to rebuild Nx14.")
    parser.add_argument("--sample-max", type=float, default=0.30, help="Maximum clearance used to rebuild Nx14.")
    parser.add_argument("--grid-size", type=int, default=40, help="Grid size for slice visualization.")
    parser.add_argument("--query-points", type=int, default=4000, help="3D query points for point-cloud visualization.")
    parser.add_argument("--volume-grid-size", type=int, default=36, help="3D grid size for iso-surface contour extraction.")
    parser.add_argument("--n-contours", type=int, default=5, help="Number of iso-surface contour levels to export.")
    parser.add_argument("--n-contour-slices", type=int, default=6, help="Number of z slices to render as contour-line plots.")
    parser.add_argument("--source", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Fixed source point for field visualization.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def load_raw_samples(samples_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_files = sorted((samples_dir / "samples").glob("*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No sample files found under {samples_dir / 'samples'}")

    raw_rows = []
    camera_poses = []
    for path in sample_files:
        data = np.load(path)
        if "raw_frame_data" not in data:
            raise KeyError(f"{path} missing raw_frame_data")
        raw_rows.append(data["raw_frame_data"].astype(np.float32))
        if "camera_pose_world" in data:
            camera_poses.append(data["camera_pose_world"].astype(np.float32))

    raw = np.concatenate(raw_rows, axis=0)
    poses = np.stack(camera_poses, axis=0) if camera_poses else np.zeros((0, 4, 4), dtype=np.float32)
    return raw, poses


def choose_source(raw_frame_data: np.ndarray, camera_poses: np.ndarray, user_source: list[float] | None) -> np.ndarray:
    if user_source is not None:
        return np.asarray(user_source, dtype=np.float32)
    if len(camera_poses) > 0:
        return camera_poses[:, :3, 3].mean(axis=0).astype(np.float32)
    all_points = raw_frame_data[:, :6].reshape(-1, 3)
    return all_points.mean(axis=0).astype(np.float32)


def train_model(args: argparse.Namespace, frame_data: np.ndarray, model_dir: Path):
    ensure_mntfields_importable(args.mntfields_src)
    from mntfields.mntfields_core.core.model_base import Model

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = Model(
        folder=str(model_dir),
        dim=3,
        B_scale=10,
        device=device,
        init_network=True,
        eval=False,
        lr=args.lr,
    )

    rng = np.random.default_rng(args.seed)
    frame_data_t = torch.from_numpy(frame_data).float()
    losses = []
    for epoch in range(args.epochs):
        if len(frame_data_t) > args.max_train_rows:
            idx = rng.choice(len(frame_data_t), size=args.max_train_rows, replace=False)
            batch = frame_data_t[idx]
        else:
            batch = frame_data_t
        loss, _ = model.train_core(epoch=1, frame_data=batch.to(model.device), is_one_frame=True)
        loss_val = None if loss is None else float(loss)
        losses.append(loss_val)
        print(f"[offline-train] epoch={epoch+1:03d} loss={loss_val}")

    return model, losses


def evaluate_travel_time_grid(model, bound: np.ndarray, source_xyz: np.ndarray, grid_size: int) -> tuple[list[tuple[float, np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    xs = np.linspace(bound[0, 0], bound[1, 0], grid_size, dtype=np.float32)
    ys = np.linspace(bound[0, 1], bound[1, 1], grid_size, dtype=np.float32)
    zs = np.linspace(bound[0, 2], bound[1, 2], 3, dtype=np.float32)

    slice_results = []
    for z in zs:
        xg, yg = np.meshgrid(xs, ys)
        target = np.stack((xg.reshape(-1), yg.reshape(-1), np.full(xg.size, z, dtype=np.float32)), axis=1)
        source = np.repeat(source_xyz[None, :], len(target), axis=0)
        pair_world = np.concatenate((source, target), axis=1)
        pair_norm = normalize_pairs(pair_world, bound)
        pair_t = torch.from_numpy(pair_norm).float().to(model.device)
        with torch.no_grad():
            tt = model.function.TravelTimes(pair_t).detach().cpu().numpy().reshape(xg.shape)
        slice_results.append((float(z), xg, yg, tt))
    return slice_results, xs, ys


def normalize_pairs(pair_world: np.ndarray, bound: np.ndarray) -> np.ndarray:
    bound_len = np.tile(bound[1] - bound[0], 2)
    bound_start = np.tile(bound[0], 2)
    return ((pair_world - bound_start) / bound_len - 0.5).astype(np.float32)


def evaluate_query_cloud(model, bound: np.ndarray, source_xyz: np.ndarray, query_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = rng.uniform(bound[0], bound[1], size=(query_points, 3)).astype(np.float32)
    source = np.repeat(source_xyz[None, :], len(target), axis=0)
    pair_world = np.concatenate((source, target), axis=1)
    pair_norm = normalize_pairs(pair_world, bound)
    pair_t = torch.from_numpy(pair_norm).float().to(model.device)
    with torch.no_grad():
        tt = model.function.TravelTimes(pair_t).detach().cpu().numpy()
    return target, tt


def evaluate_travel_time_volume(model, bound: np.ndarray, source_xyz: np.ndarray, volume_grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(bound[0, 0], bound[1, 0], volume_grid_size, dtype=np.float32)
    ys = np.linspace(bound[0, 1], bound[1, 1], volume_grid_size, dtype=np.float32)
    zs = np.linspace(bound[0, 2], bound[1, 2], volume_grid_size, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    target = np.stack((xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)), axis=1)
    source = np.repeat(source_xyz[None, :], len(target), axis=0)
    pair_world = np.concatenate((source, target), axis=1)
    pair_norm = normalize_pairs(pair_world, bound)
    pair_t = torch.from_numpy(pair_norm).float().to(model.device)
    with torch.no_grad():
        tt = model.function.TravelTimes(pair_t).detach().cpu().numpy().reshape(xg.shape)
    return xs, ys, zs, tt.astype(np.float32)


def evaluate_speed_volume(model, bound: np.ndarray, source_xyz: np.ndarray, volume_grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(bound[0, 0], bound[1, 0], volume_grid_size, dtype=np.float32)
    ys = np.linspace(bound[0, 1], bound[1, 1], volume_grid_size, dtype=np.float32)
    zs = np.linspace(bound[0, 2], bound[1, 2], volume_grid_size, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    target = np.stack((xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)), axis=1)
    source = np.repeat(source_xyz[None, :], len(target), axis=0)
    pair_world = np.concatenate((source, target), axis=1)
    pair_norm = normalize_pairs(pair_world, bound)
    pair_t = torch.from_numpy(pair_norm).float().to(model.device).requires_grad_(True)
    with torch.enable_grad():
        sp = model.function.Speed(pair_t).detach().cpu().numpy().reshape(xg.shape)
    sp = np.clip(sp.astype(np.float32), 0.0, 1.0)
    return xs, ys, zs, sp


def save_visualizations(
    output_dir: Path,
    raw_frame_data: np.ndarray,
    bound: np.ndarray,
    source_xyz: np.ndarray,
    slice_results,
    query_xyz: np.ndarray,
    query_tt: np.ndarray,
    volume_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    volume_tt: np.ndarray | None,
    n_contours: int,
    n_contour_slices: int,
    losses: list[float | None],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    field_cmap = plt.get_cmap(FIELD_CMAP_NAME)

    fig, axes = plt.subplots(1, len(slice_results), figsize=(5 * len(slice_results), 5), constrained_layout=True)
    if len(slice_results) == 1:
        axes = [axes]
    obs = raw_frame_data[:, 9:12]
    for ax, (z, xg, yg, tt) in zip(axes, slice_results):
        im = ax.contourf(xg, yg, tt, levels=20, cmap=FIELD_CMAP_NAME)
        mask = np.abs(obs[:, 2] - z) < max(0.02, 0.1 * max(1e-3, bound[1, 2] - bound[0, 2]))
        if mask.any():
            ax.scatter(obs[mask, 0], obs[mask, 1], s=2, c="white", alpha=0.4)
        ax.scatter([source_xyz[0]], [source_xyz[1]], c="red", s=40)
        ax.set_title(f"TT slice z={z:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.savefig(output_dir / "field_slices.png", dpi=180)
    plt.close(fig)

    try:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(query_xyz[:, 0], query_xyz[:, 1], query_xyz[:, 2], c=query_tt, s=6, cmap=FIELD_CMAP_NAME)
        ax.scatter([source_xyz[0]], [source_xyz[1]], [source_xyz[2]], c="red", s=60)
        ax.set_title("3D Travel-Time Queries")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(p, ax=ax, label="travel time")
        fig.savefig(output_dir / "field_query_cloud.png", dpi=180)
        plt.close(fig)
    except ValueError:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        p0 = axes[0].scatter(query_xyz[:, 0], query_xyz[:, 1], c=query_tt, s=6, cmap=FIELD_CMAP_NAME)
        axes[0].scatter([source_xyz[0]], [source_xyz[1]], c="red", s=40)
        axes[0].set_title("Field Queries (XY)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(p0, ax=axes[0], label="travel time")

        p1 = axes[1].scatter(query_xyz[:, 0], query_xyz[:, 2], c=query_tt, s=6, cmap=FIELD_CMAP_NAME)
        axes[1].scatter([source_xyz[0]], [source_xyz[2]], c="red", s=40)
        axes[1].set_title("Field Queries (XZ)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        fig.colorbar(p1, ax=axes[1], label="travel time")
        fig.savefig(output_dir / "field_query_cloud.png", dpi=180)
        plt.close(fig)

    finite_losses = [v for v in losses if v is not None]
    if finite_losses:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(losses) + 1), [np.nan if v is None else v for v in losses], marker="o")
        ax.set_title("Offline Training Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "training_loss.png", dpi=180)
        plt.close(fig)

    np.savez_compressed(
        output_dir / "field_query_cloud.npz",
        xyz=query_xyz.astype(np.float32),
        travel_time=query_tt.astype(np.float32),
        source_xyz=source_xyz.astype(np.float32),
    )

    if o3d is not None and len(query_xyz) > 0:
        tt = query_tt.astype(np.float32)
        tt_min = float(np.min(tt))
        tt_max = float(np.max(tt))
        denom = max(tt_max - tt_min, 1e-6)
        tt_norm = (tt - tt_min) / denom
        colors = field_cmap(tt_norm)[:, :3].astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(query_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(output_dir / "field_query_cloud.ply"), pcd)

        src = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        src.paint_uniform_color([1.0, 0.0, 0.0])
        src.translate(source_xyz.astype(np.float64))
        o3d.io.write_triangle_mesh(str(output_dir / "field_source_marker.ply"), src)

    if measure is not None and o3d is not None and volume_axes is not None and volume_tt is not None:
        xs, ys, zs = volume_axes
        spacing = (
            float(xs[1] - xs[0]) if len(xs) > 1 else 1.0,
            float(ys[1] - ys[0]) if len(ys) > 1 else 1.0,
            float(zs[1] - zs[0]) if len(zs) > 1 else 1.0,
        )
        origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float32)
        finite = np.isfinite(volume_tt)
        if finite.any():
            vmin = float(np.min(volume_tt[finite]))
            vmax = float(np.max(volume_tt[finite]))
            if vmax - vmin > 1e-6:
                levels = np.linspace(vmin, vmax, n_contours + 2, dtype=np.float32)[1:-1]
                contour_dir = output_dir / "contours"
                contour_dir.mkdir(exist_ok=True)
                contour_manifest = []
                for idx, level in enumerate(levels):
                    verts, faces, normals, _ = measure.marching_cubes(volume_tt, level=float(level), spacing=spacing)
                    verts = verts + origin[None, :]
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
                    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
                    level_norm = (float(level) - vmin) / (vmax - vmin)
                    color = field_cmap(level_norm)[:3]
                    mesh.paint_uniform_color(color)
                    safe_level = f"{level:.4f}".replace(".", "p")
                    fname = f"contour_{idx:02d}_{safe_level}.ply"
                    o3d.io.write_triangle_mesh(str(contour_dir / fname), mesh)
                    contour_manifest.append(
                        {
                            "file": fname,
                            "level": float(level),
                            "level_norm": float(level_norm),
                            "color_rgb": [float(c) for c in color],
                        }
                    )
                with open(contour_dir / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "cmap": FIELD_CMAP_NAME,
                            "vmin": vmin,
                            "vmax": vmax,
                            "contours": contour_manifest,
                        },
                        f,
                        indent=2,
                    )

    if volume_axes is not None and volume_tt is not None:
        xs, ys, zs = volume_axes
        np.savez_compressed(
            output_dir / "field_volume.npz",
            xs=xs.astype(np.float32),
            ys=ys.astype(np.float32),
            zs=zs.astype(np.float32),
            travel_time=volume_tt.astype(np.float32),
            source_xyz=source_xyz.astype(np.float32),
        )

        obs = raw_frame_data[:, 9:12].astype(np.float32)
        finite = np.isfinite(volume_tt)
        if finite.any():
            tt_finite = volume_tt[finite]
            t_min = float(np.min(tt_finite))
            t_max = float(np.max(tt_finite))
            if t_max - t_min > 1e-6:
                contour_levels_major = np.linspace(t_min, t_max, 9, dtype=np.float32)
                contour_levels_minor = np.linspace(t_min, t_max, 25, dtype=np.float32)
                slice_indices = np.linspace(0, len(zs) - 1, max(1, n_contour_slices)).astype(int)
                cols = min(3, len(slice_indices))
                rows = int(np.ceil(len(slice_indices) / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), constrained_layout=True)
                axes = np.atleast_1d(axes).reshape(rows, cols)

                for ax in axes.flat:
                    ax.set_facecolor((8 / 255, 10 / 255, 120 / 255))

                for plot_i, slice_idx in enumerate(slice_indices):
                    r = plot_i // cols
                    c = plot_i % cols
                    ax = axes[r, c]
                    z = float(zs[slice_idx])
                    tt_slice = volume_tt[:, :, slice_idx].T
                    xg, yg = np.meshgrid(xs, ys)

                    major = contour_levels_major[
                        (contour_levels_major >= float(np.nanmin(tt_slice))) &
                        (contour_levels_major <= float(np.nanmax(tt_slice)))
                    ]
                    minor = contour_levels_minor[
                        (contour_levels_minor >= float(np.nanmin(tt_slice))) &
                        (contour_levels_minor <= float(np.nanmax(tt_slice)))
                    ]
                    if len(minor) > 0:
                        ax.contour(
                            xg, yg, tt_slice,
                            levels=minor,
                            cmap=FIELD_CMAP_NAME,
                            linewidths=0.5,
                            alpha=0.45,
                        )
                    if len(major) > 0:
                        ax.contour(
                            xg, yg, tt_slice,
                            levels=major,
                            cmap=FIELD_CMAP_NAME,
                            linewidths=1.1,
                            alpha=0.95,
                        )

                    z_band = max(0.02, 0.5 * (float(zs[1] - zs[0]) if len(zs) > 1 else 0.02))
                    mask = np.abs(obs[:, 2] - z) <= z_band
                    if np.any(mask):
                        ax.scatter(obs[mask, 0], obs[mask, 1], s=2, c="#ffe600", alpha=0.5)
                    if abs(float(source_xyz[2]) - z) <= z_band:
                        ax.scatter([source_xyz[0]], [source_xyz[1]], s=35, c="red")

                    ax.set_title(f"z = {z:.3f}", color="white")
                    ax.set_xlabel("x", color="white")
                    ax.set_ylabel("y", color="white")
                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_color("white")
                    ax.set_aspect("equal")

                for plot_i in range(len(slice_indices), rows * cols):
                    r = plot_i // cols
                    c = plot_i % cols
                    axes[r, c].axis("off")

                fig.savefig(output_dir / "field_contour_stack.png", dpi=180, facecolor=(8 / 255, 10 / 255, 120 / 255))
                plt.close(fig)


def save_speed_visualizations(
    output_dir: Path,
    raw_frame_data: np.ndarray,
    source_xyz: np.ndarray,
    volume_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_tt: np.ndarray,
    volume_speed: np.ndarray,
    n_contour_slices: int,
) -> None:
    xs, ys, zs = volume_axes
    obs = raw_frame_data[:, 9:12].astype(np.float32)
    slice_indices = np.linspace(0, len(zs) - 1, max(1, n_contour_slices)).astype(int)
    cols = min(3, len(slice_indices))
    rows = int(np.ceil(len(slice_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.set_facecolor((8 / 255, 10 / 255, 120 / 255))

    xg, yg = np.meshgrid(xs, ys)
    z_band = max(0.02, 0.5 * (float(zs[1] - zs[0]) if len(zs) > 1 else 0.02))

    for plot_i, slice_idx in enumerate(slice_indices):
        r = plot_i // cols
        c = plot_i % cols
        ax = axes[r, c]
        z = float(zs[slice_idx])
        tt_slice = volume_tt[:, :, slice_idx].T
        sp_slice = np.clip(volume_speed[:, :, slice_idx].T, 0.0, 1.0)

        im = ax.imshow(
            sp_slice,
            origin="lower",
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            zorder=1,
        )
        tt_finite = tt_slice[np.isfinite(tt_slice)]
        if tt_finite.size > 0 and float(np.max(tt_finite) - np.min(tt_finite)) > 1e-6:
            levels_minor = np.linspace(float(np.min(tt_finite)), float(np.max(tt_finite)), 25, dtype=np.float32)
            levels_major = np.linspace(float(np.min(tt_finite)), float(np.max(tt_finite)), 10, dtype=np.float32)
            ax.contour(xg, yg, tt_slice, levels=levels_minor, colors="black", linewidths=0.35, alpha=0.35, zorder=2)
            ax.contour(xg, yg, tt_slice, levels=levels_major, colors="black", linewidths=0.8, alpha=0.7, zorder=3)

        mask = np.abs(obs[:, 2] - z) <= z_band
        if np.any(mask):
            ax.scatter(obs[mask, 0], obs[mask, 1], s=2, c="#ffe600", alpha=0.4, zorder=4)
        if abs(float(source_xyz[2]) - z) <= z_band:
            ax.scatter([source_xyz[0]], [source_xyz[1]], s=35, c="red", zorder=5)

        ax.set_title(f"z = {z:.3f}", color="white")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    for plot_i in range(len(slice_indices), rows * cols):
        r = plot_i // cols
        c = plot_i % cols
        axes[r, c].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Speed")
    fig.savefig(output_dir / "field_speed_contour_stack.png", dpi=180, facecolor=(8 / 255, 10 / 255, 120 / 255))
    plt.close(fig)

    np.savez_compressed(
        output_dir / "field_speed_volume.npz",
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        zs=zs.astype(np.float32),
        speed=volume_speed.astype(np.float32),
        source_xyz=source_xyz.astype(np.float32),
    )


def save_checkpoint(model, output_dir: Path, bound: np.ndarray, source_xyz: np.ndarray, losses: list[float | None], n_rows: int) -> None:
    torch.save(model.network.state_dict(), output_dir / "network.pt")
    torch.save(model.B, output_dir / "B.pt")
    np.savez_compressed(
        output_dir / "field_metadata.npz",
        normalization_bound=bound.astype(np.float32),
        source_xyz=source_xyz.astype(np.float32),
        losses=np.array([np.nan if v is None else v for v in losses], dtype=np.float32),
        n_rows=np.array([n_rows], dtype=np.int32),
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows_used": int(n_rows),
                "source_xyz": [float(v) for v in source_xyz],
                "bound_min": [float(v) for v in bound[0]],
                "bound_max": [float(v) for v in bound[1]],
                "final_loss": None if not losses or losses[-1] is None else float(losses[-1]),
            },
            f,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    samples_dir = args.samples_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else samples_dir / MODEL_SUBDIR)

    raw_frame_data, camera_poses = load_raw_samples(samples_dir)
    bound = compute_normalization_bound(raw_frame_data, args.bound_padding_xy, args.bound_padding_z)
    frame_data = calculate_speed_and_normal(raw_frame_data, bound, args.sample_min, args.sample_max)
    if len(frame_data) == 0:
        raise RuntimeError("Merged raw samples converted to an empty Nx14 dataset.")

    source_xyz = choose_source(raw_frame_data, camera_poses, args.source)
    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model, losses = train_model(args, frame_data, model_dir)
    slices, _, _ = evaluate_travel_time_grid(model, bound, source_xyz, args.grid_size)
    query_xyz, query_tt = evaluate_query_cloud(model, bound, source_xyz, args.query_points, args.seed)
    xs_v, ys_v, zs_v, volume_tt = evaluate_travel_time_volume(model, bound, source_xyz, args.volume_grid_size)
    _, _, _, volume_speed = evaluate_speed_volume(model, bound, source_xyz, args.volume_grid_size)

    save_checkpoint(model, model_dir, bound, source_xyz, losses, len(frame_data))
    save_visualizations(
        model_dir,
        raw_frame_data,
        bound,
        source_xyz,
        slices,
        query_xyz,
        query_tt,
        (xs_v, ys_v, zs_v),
        volume_tt,
        args.n_contours,
        args.n_contour_slices,
        losses,
    )
    save_speed_visualizations(
        model_dir,
        raw_frame_data,
        source_xyz,
        (xs_v, ys_v, zs_v),
        volume_tt,
        volume_speed,
        args.n_contour_slices,
    )

    print(f"[offline-train] wrote model and plots to {model_dir}")


if __name__ == "__main__":
    main()
