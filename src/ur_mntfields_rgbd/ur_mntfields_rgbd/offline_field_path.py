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
    import open3d as o3d
except ImportError:
    o3d = None


DEFAULT_MNTFIELDS_SRC = Path("/home/mayank/mntfields/src/mntfields")
FIELD_CMAP_NAME = "gist_heat"


def ensure_mntfields_importable(mntfields_src: Path) -> None:
    if str(mntfields_src) not in sys.path:
        sys.path.insert(0, str(mntfields_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out and visualize an offline trajectory through a trained MNTFields model.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory produced by offline_field_train.py.")
    parser.add_argument("--samples-dir", type=Path, default=None, help="Optional samples directory for choosing a default goal.")
    parser.add_argument("--mntfields-src", type=Path, default=DEFAULT_MNTFIELDS_SRC, help="Path to the local mntfields package root.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inference.")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Start point. Defaults to saved source_xyz.")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Goal point. If omitted, chooses a heuristic goal from samples.")
    parser.add_argument("--goal-min-dist", type=float, default=0.10, help="Minimum preferred distance from start for auto goal selection.")
    parser.add_argument("--goal-max-dist", type=float, default=0.35, help="Maximum preferred distance from start for auto goal selection.")
    parser.add_argument("--step-size", type=float, default=0.05, help="Step size for rollout in meters.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Blend between field direction and geometric steering.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum rollout steps.")
    parser.add_argument("--goal-tolerance", type=float, default=0.05, help="Distance threshold for success.")
    parser.add_argument("--out-name", type=str, default="trajectory", help="Prefix for saved outputs.")
    return parser.parse_args()


def load_metadata(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = np.load(model_dir / "field_metadata.npz")
    bound = meta["normalization_bound"].astype(np.float32)
    source = meta["source_xyz"].astype(np.float32)
    return bound, source


def load_model(model_dir: Path, mntfields_src: Path, device: str):
    ensure_mntfields_importable(mntfields_src)
    from mntfields.mntfields_core.ntrl import model_function_metric as model_function
    from mntfields.mntfields_core.ntrl import model_network_metric as model_network

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    B = torch.load(model_dir / "B.pt", map_location=device, weights_only=False)
    network = model_network.NN(device, 3, B)
    state = torch.load(model_dir / "network.pt", map_location=device, weights_only=False)
    network.load_state_dict(state)
    network.float()
    network.to(device)
    network.eval()
    function = model_function.Function(str(model_dir), device, network, 3)
    return network, function, device


def normalize_pairs(pair_world: np.ndarray, bound: np.ndarray) -> np.ndarray:
    bound_len = np.tile(bound[1] - bound[0], 2)
    bound_start = np.tile(bound[0], 2)
    return ((pair_world - bound_start) / bound_len - 0.5).astype(np.float32)


def choose_default_goal(
    samples_dir: Path,
    start_xyz: np.ndarray,
    goal_min_dist: float,
    goal_max_dist: float,
) -> np.ndarray:
    sample_files = sorted((samples_dir / "samples").glob("*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No sample files found under {samples_dir / 'samples'}")

    rows = []
    for path in sample_files:
        data = np.load(path)
        rows.append(data["raw_frame_data"].astype(np.float32))
    raw = np.concatenate(rows, axis=0)

    x0 = raw[:, :3]
    x1 = raw[:, 3:6]
    cp0 = raw[:, 6:9]
    cp1 = raw[:, 9:12]
    pts = np.concatenate((x0, x1), axis=0)
    cps = np.concatenate((cp0, cp1), axis=0)
    clearance = np.linalg.norm(pts - cps, axis=1)
    dist = np.linalg.norm(pts - start_xyz[None, :], axis=1)

    valid = clearance > 0.08
    band = (dist >= goal_min_dist) & (dist <= goal_max_dist)

    if np.any(valid & band):
        candidates = valid & band
        score = clearance.copy()
        score[~candidates] = -np.inf
        goal = pts[np.argmax(score)].astype(np.float32)
        return goal

    if np.any(valid):
        score = -np.abs(dist - min(max(goal_min_dist, 0.2), goal_max_dist))
        score += 0.25 * clearance
        score[~valid] = -np.inf
        goal = pts[np.argmax(score)].astype(np.float32)
        return goal

    goal = pts[np.argmin(dist)].astype(np.float32)
    return goal


def rollout_path(
    function,
    device: str,
    bound: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    step_size: float,
    alpha: float,
    max_steps: int,
    goal_tolerance: float,
) -> tuple[np.ndarray, bool]:
    cur = start_xyz.astype(np.float32).copy()
    goal = goal_xyz.astype(np.float32).copy()
    path = [cur.copy()]

    for _ in range(max_steps):
        delta = goal - cur
        dist = float(np.linalg.norm(delta))
        if dist <= goal_tolerance:
            path.append(goal.copy())
            return np.stack(path, axis=0), True

        geom_dir = delta / max(dist, 1e-8)
        pair = np.concatenate((cur, goal), axis=0)[None, :]
        pair_n = normalize_pairs(pair, bound)
        pair_t = torch.from_numpy(pair_n).float().to(device).requires_grad_(True)
        tau, _, coords = function.network.out(pair_t)
        grad = torch.autograd.grad(
            outputs=tau.sum(),
            inputs=coords,
            retain_graph=False,
            create_graph=False,
        )[0]

        grad_xyz_norm = grad[0, :3].detach().cpu().numpy().astype(np.float32)
        bound_len = (bound[1] - bound[0]).astype(np.float32)
        grad_xyz = grad_xyz_norm / (bound_len + 1e-8)
        grad_norm = float(np.linalg.norm(grad_xyz))
        if grad_norm < 1e-6:
            direction = geom_dir
        else:
            field_dir = -grad_xyz / grad_norm
            direction = alpha * field_dir + (1.0 - alpha) * geom_dir
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        step = min(step_size, dist)
        nxt = cur + direction.astype(np.float32) * step
        nxt = np.minimum(np.maximum(nxt, bound[0]), bound[1])
        if float(np.linalg.norm(nxt - cur)) < 1e-6:
            break
        cur = nxt.astype(np.float32)
        path.append(cur.copy())

    return np.stack(path, axis=0), False


def save_outputs(
    model_dir: Path,
    out_name: str,
    path_xyz: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    success: bool,
) -> None:
    np.savez_compressed(
        model_dir / f"{out_name}.npz",
        path_xyz=path_xyz.astype(np.float32),
        start_xyz=start_xyz.astype(np.float32),
        goal_xyz=goal_xyz.astype(np.float32),
        success=np.array([1 if success else 0], dtype=np.int32),
    )

    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], "-o", color="tab:red", markersize=3)
        ax.scatter([start_xyz[0]], [start_xyz[1]], [start_xyz[2]], c="green", s=60, label="start")
        ax.scatter([goal_xyz[0]], [goal_xyz[1]], [goal_xyz[2]], c="blue", s=60, label="goal")
        ax.set_title(f"Offline Field Trajectory ({'success' if success else 'partial'})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        fig.savefig(model_dir / f"{out_name}.png", dpi=180)
        plt.close(fig)
    except ValueError:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(path_xyz[:, 0], path_xyz[:, 1], "-o", color="tab:red", markersize=3)
        axes[0].scatter([start_xyz[0]], [start_xyz[1]], c="green", s=50)
        axes[0].scatter([goal_xyz[0]], [goal_xyz[1]], c="blue", s=50)
        axes[0].set_title("Trajectory XY")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        axes[1].plot(path_xyz[:, 0], path_xyz[:, 2], "-o", color="tab:red", markersize=3)
        axes[1].scatter([start_xyz[0]], [start_xyz[2]], c="green", s=50)
        axes[1].scatter([goal_xyz[0]], [goal_xyz[2]], c="blue", s=50)
        axes[1].set_title("Trajectory XZ")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        fig.savefig(model_dir / f"{out_name}.png", dpi=180)
        plt.close(fig)

    if o3d is not None:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(path_xyz.astype(np.float64))
        lines = np.array([[i, i + 1] for i in range(len(path_xyz) - 1)], dtype=np.int32)
        if len(lines) > 0:
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1)))
            o3d.io.write_line_set(str(model_dir / f"{out_name}.ply"), line_set)

        start_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        start_mesh.paint_uniform_color([0.0, 1.0, 0.0])
        start_mesh.translate(start_xyz.astype(np.float64))
        o3d.io.write_triangle_mesh(str(model_dir / f"{out_name}_start.ply"), start_mesh)

        goal_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        goal_mesh.paint_uniform_color([0.0, 0.0, 1.0])
        goal_mesh.translate(goal_xyz.astype(np.float64))
        o3d.io.write_triangle_mesh(str(model_dir / f"{out_name}_goal.ply"), goal_mesh)

    with open(model_dir / f"{out_name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "success": bool(success),
                "num_waypoints": int(len(path_xyz)),
                "start_xyz": [float(v) for v in start_xyz],
                "goal_xyz": [float(v) for v in goal_xyz],
            },
            f,
            indent=2,
        )


def save_contour_overlay(
    model_dir: Path,
    out_name: str,
    path_xyz: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
) -> None:
    volume_path = model_dir / "field_volume.npz"
    if not volume_path.exists():
        return

    data = np.load(volume_path)
    xs = data["xs"].astype(np.float32)
    ys = data["ys"].astype(np.float32)
    zs = data["zs"].astype(np.float32)
    volume_tt = data["travel_time"].astype(np.float32)

    finite = np.isfinite(volume_tt)
    if not finite.any():
        return

    tt_finite = volume_tt[finite]
    t_min = float(np.min(tt_finite))
    t_max = float(np.max(tt_finite))
    if t_max - t_min <= 1e-6:
        return

    contour_levels_major = np.linspace(t_min, t_max, 9, dtype=np.float32)
    contour_levels_minor = np.linspace(t_min, t_max, 25, dtype=np.float32)

    slice_indices = np.unique(np.clip(
        [int(np.argmin(np.abs(zs - z))) for z in np.linspace(float(path_xyz[:, 2].min()), float(path_xyz[:, 2].max()), 6)],
        0,
        len(zs) - 1,
    ))
    if len(slice_indices) == 0:
        slice_indices = np.linspace(0, len(zs) - 1, min(6, len(zs))).astype(int)

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
                colors="#39d2ff",
                linewidths=0.5,
                alpha=0.45,
            )
        if len(major) > 0:
            ax.contour(
                xg, yg, tt_slice,
                levels=major,
                colors="#9ffcff",
                linewidths=1.1,
                alpha=0.95,
            )

        mask = np.abs(path_xyz[:, 2] - z) <= z_band
        if np.any(mask):
            pts = path_xyz[mask]
            ax.plot(pts[:, 0], pts[:, 1], "-o", color="#ff5a36", linewidth=1.5, markersize=3, alpha=0.95)

        if abs(float(start_xyz[2]) - z) <= z_band:
            ax.scatter([start_xyz[0]], [start_xyz[1]], s=40, c="#00ff66", edgecolors="black", linewidths=0.4)
        if abs(float(goal_xyz[2]) - z) <= z_band:
            ax.scatter([goal_xyz[0]], [goal_xyz[1]], s=40, c="#3b82f6", edgecolors="black", linewidths=0.4)

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

    fig.savefig(model_dir / f"{out_name}_contour_stack.png", dpi=180, facecolor=(8 / 255, 10 / 255, 120 / 255))
    plt.close(fig)


def save_plane_field_overlay(
    model_dir: Path,
    out_name: str,
    path_xyz: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
) -> None:
    volume_path = model_dir / "field_volume.npz"
    speed_path = model_dir / "field_speed_volume.npz"
    if not volume_path.exists() or not speed_path.exists():
        return

    data = np.load(volume_path)
    speed_data = np.load(speed_path)
    xs = data["xs"].astype(np.float32)
    ys = data["ys"].astype(np.float32)
    zs = data["zs"].astype(np.float32)
    volume_tt = data["travel_time"].astype(np.float32)
    volume_speed = np.clip(speed_data["speed"].astype(np.float32), 0.0, 1.0)

    finite = np.isfinite(volume_tt)
    if not finite.any():
        return

    tt_finite = volume_tt[finite]
    t_min = float(np.min(tt_finite))
    t_max = float(np.max(tt_finite))
    if t_max - t_min <= 1e-6:
        return

    z_ref = float(np.median(path_xyz[:, 2]))
    y_ref = float(np.median(path_xyz[:, 1]))
    z_idx = int(np.argmin(np.abs(zs - z_ref)))
    y_idx = int(np.argmin(np.abs(ys - y_ref)))

    xg_xy, yg_xy = np.meshgrid(xs, ys)
    tt_xy = volume_tt[:, :, z_idx].T

    xg_xz, zg_xz = np.meshgrid(xs, zs)
    tt_xz = volume_tt[:, y_idx, :].T

    levels_minor_xy = np.linspace(float(np.nanmin(tt_xy)), float(np.nanmax(tt_xy)), 25, dtype=np.float32)
    levels_major_xy = np.linspace(float(np.nanmin(tt_xy)), float(np.nanmax(tt_xy)), 10, dtype=np.float32)
    levels_minor_xz = np.linspace(float(np.nanmin(tt_xz)), float(np.nanmax(tt_xz)), 25, dtype=np.float32)
    levels_major_xz = np.linspace(float(np.nanmin(tt_xz)), float(np.nanmax(tt_xz)), 10, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sp_xy = volume_speed[:, :, z_idx].T
    im0 = axes[0].imshow(
        sp_xy,
        origin="lower",
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[0].contour(xg_xy, yg_xy, tt_xy, levels=levels_minor_xy, colors="black", linewidths=0.35, alpha=0.35)
    axes[0].contour(xg_xy, yg_xy, tt_xy, levels=levels_major_xy, colors="black", linewidths=0.8, alpha=0.7)
    axes[0].plot(path_xyz[:, 0], path_xyz[:, 1], "-o", color="#ff2d20", markersize=3, linewidth=2.0)
    axes[0].scatter([start_xyz[0]], [start_xyz[1]], c="#00ff66", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    axes[0].scatter([goal_xyz[0]], [goal_xyz[1]], c="#2563eb", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    axes[0].set_title(f"Trajectory XY on Speed Field (z={zs[z_idx]:.3f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="Speed")

    sp_xz = volume_speed[:, y_idx, :].T
    im1 = axes[1].imshow(
        sp_xz,
        origin="lower",
        extent=[float(xs[0]), float(xs[-1]), float(zs[0]), float(zs[-1])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[1].contour(xg_xz, zg_xz, tt_xz, levels=levels_minor_xz, colors="black", linewidths=0.35, alpha=0.35)
    axes[1].contour(xg_xz, zg_xz, tt_xz, levels=levels_major_xz, colors="black", linewidths=0.8, alpha=0.7)
    axes[1].plot(path_xyz[:, 0], path_xyz[:, 2], "-o", color="#ff2d20", markersize=3, linewidth=2.0)
    axes[1].scatter([start_xyz[0]], [start_xyz[2]], c="#00ff66", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    axes[1].scatter([goal_xyz[0]], [goal_xyz[2]], c="#2563eb", s=70, edgecolors="black", linewidths=0.5, zorder=5)
    axes[1].set_title(f"Trajectory XZ on Speed Field (y={ys[y_idx]:.3f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    fig.colorbar(im1, ax=axes[1], label="Speed")

    fig.savefig(model_dir / f"{out_name}_field_planes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    bound, source_xyz = load_metadata(model_dir)
    _, function, device = load_model(model_dir, args.mntfields_src, args.device)

    start_xyz = np.asarray(args.start, dtype=np.float32) if args.start is not None else source_xyz.astype(np.float32)
    if args.goal is not None:
        goal_xyz = np.asarray(args.goal, dtype=np.float32)
    else:
        samples_dir = args.samples_dir.expanduser().resolve() if args.samples_dir else model_dir.parent
        goal_xyz = choose_default_goal(
            samples_dir,
            start_xyz,
            args.goal_min_dist,
            args.goal_max_dist,
        )

    path_xyz, success = rollout_path(
        function=function,
        device=device,
        bound=bound,
        start_xyz=start_xyz,
        goal_xyz=goal_xyz,
        step_size=args.step_size,
        alpha=args.alpha,
        max_steps=args.max_steps,
        goal_tolerance=args.goal_tolerance,
    )
    save_outputs(model_dir, args.out_name, path_xyz, start_xyz, goal_xyz, success)
    save_contour_overlay(model_dir, args.out_name, path_xyz, start_xyz, goal_xyz)
    save_plane_field_overlay(model_dir, args.out_name, path_xyz, start_xyz, goal_xyz)
    print(f"[offline-path] wrote {args.out_name} outputs to {model_dir}")


if __name__ == "__main__":
    main()
