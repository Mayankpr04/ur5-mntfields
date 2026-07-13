from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
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
    details = _edge_clearance_details(checker, plan, interp_step_rad)
    return bool(details["ok"]), float(details["min_clearance"])


def _edge_clearance_details(
    checker: UR5PointCloudCollisionChecker,
    plan: np.ndarray,
    interp_step_rad: float,
) -> dict[str, object]:
    pts = np.asarray(plan, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return {"ok": False, "min_clearance": -1.0, "min_q": None, "states": np.zeros((0, 6)), "clearances": np.zeros((0,))}
    states = _sample_joint_path(pts, interp_step_rad).astype(np.float32)
    if len(states) == 0:
        return {"ok": False, "min_clearance": -1.0, "min_q": None, "states": states, "clearances": np.zeros((0,))}
    clearances = checker.clearance_batch(states)
    if clearances.size == 0:
        return {"ok": True, "min_clearance": -1.0, "min_q": states[0], "states": states, "clearances": clearances}
    min_idx = int(np.argmin(clearances))
    return {
        "ok": True,
        "min_clearance": float(clearances[min_idx]),
        "min_q": states[min_idx].astype(np.float64),
        "min_idx": min_idx,
        "states": states.astype(np.float64),
        "clearances": clearances.astype(np.float32),
    }


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


def _nearest_box_point(center: np.ndarray, box: np.ndarray) -> tuple[float, np.ndarray]:
    box = np.asarray(box, dtype=np.float64).reshape(6)
    half = 0.5 * box[3:]
    bmin = box[:3] - half
    bmax = box[:3] + half
    clipped = np.minimum(np.maximum(center, bmin), bmax)
    delta = center - clipped
    outside_dist = float(np.linalg.norm(delta))
    if outside_dist > 1.0e-10:
        return outside_dist, clipped
    dist_to_min = center - bmin
    dist_to_max = bmax - center
    axis_min = int(np.argmin(np.minimum(dist_to_min, dist_to_max)))
    if dist_to_min[axis_min] < dist_to_max[axis_min]:
        nearest = center.copy()
        nearest[axis_min] = bmin[axis_min]
        signed = -float(dist_to_min[axis_min])
    else:
        nearest = center.copy()
        nearest[axis_min] = bmax[axis_min]
        signed = -float(dist_to_max[axis_min])
    return signed, nearest


def _sphere_contact_diagnostic(checker: UR5PointCloudCollisionChecker, q: np.ndarray) -> dict[str, object] | None:
    centers, radii = checker.robot_spheres(np.asarray(q, dtype=np.float64))
    best: dict[str, object] | None = None
    for sphere_idx, (center, radius) in enumerate(zip(centers, radii)):
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
        if checker.kdtree is not None and len(checker.occupied_points):
            dist, obs_idx = checker.kdtree.query(center)
            clearance = float(dist) - radius
            cand = {
                "clearance": clearance,
                "sphere_idx": int(sphere_idx),
                "center": center,
                "radius": radius,
                "nearest": np.asarray(checker.occupied_points[int(obs_idx)], dtype=np.float64),
                "obstacle": "point",
                "obstacle_idx": int(obs_idx),
            }
            if best is None or clearance < float(best["clearance"]):
                best = cand
        for box_idx, box in enumerate(np.asarray(checker.box_obstacles, dtype=np.float64).reshape(-1, 6)):
            signed_dist, nearest = _nearest_box_point(center, box)
            clearance = float(signed_dist) - radius
            cand = {
                "clearance": clearance,
                "sphere_idx": int(sphere_idx),
                "center": center,
                "radius": radius,
                "nearest": nearest,
                "obstacle": "box",
                "obstacle_idx": int(box_idx),
            }
            if best is None or clearance < float(best["clearance"]):
                best = cand
    return best


def _contact_geometries(
    checker: UR5PointCloudCollisionChecker,
    q: np.ndarray,
    max_spheres: int,
    radius_scale: float,
) -> tuple[list, dict[str, object] | None]:
    diag = _sphere_contact_diagnostic(checker, q)
    if diag is None:
        return [], None
    geoms = _sphere_geometries(checker, q, (0.55, 0.55, 0.55), radius_scale, max_spheres)
    center = np.asarray(diag["center"], dtype=np.float64)
    nearest = np.asarray(diag["nearest"], dtype=np.float64)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.01, float(diag["radius"]) * 1.25), resolution=16)
    sphere.translate(center)
    sphere.paint_uniform_color((1.0, 0.0, 0.0))
    obs = o3d.geometry.TriangleMesh.create_sphere(radius=0.018, resolution=12)
    obs.translate(nearest)
    obs.paint_uniform_color((1.0, 0.0, 1.0))
    geoms.extend([sphere, obs])
    cyl = _segment_cylinder(center, nearest, (1.0, 0.0, 1.0), 0.004)
    if cyl is not None:
        geoms.append(cyl)
    return geoms, diag


def _trajectory_waypoint_indices(count: int, max_waypoints: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_waypoints <= 0 or count <= max_waypoints:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, max_waypoints).astype(np.int64))


def _field_values_to_goal(
    model: ArmFieldModel,
    kin: UR5Kinematics,
    q_states: np.ndarray,
    q_goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(q_states, dtype=np.float64).reshape(-1, 6)
    if len(states) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    qn = np.asarray([kin.normalize(q) for q in states], dtype=np.float32)
    goal_n = kin.normalize(q_goal).astype(np.float32)
    goals = np.repeat(goal_n[None, :], len(qn), axis=0)
    speed, _ = model.predict_normalized_pair_speeds(qn, goals, batch_size=4096)
    tau = model.predict_travel_times(qn, goals, batch_size=4096)
    return np.asarray(speed, dtype=np.float32), np.asarray(tau, dtype=np.float32)


def _slice_lateral_direction(kin: UR5Kinematics, start_q: np.ndarray, goal_q: np.ndarray, plan: np.ndarray) -> np.ndarray:
    start_n = kin.normalize(start_q).astype(np.float64)
    goal_n = kin.normalize(goal_q).astype(np.float64)
    direct = goal_n - start_n
    denom = float(np.dot(direct, direct))
    best = np.zeros(6, dtype=np.float64)
    best_norm = 0.0
    for q in np.asarray(plan, dtype=np.float64).reshape(-1, 6):
        qn = kin.normalize(q).astype(np.float64)
        if denom > 1.0e-12:
            t = float(np.clip(np.dot(qn - start_n, direct) / denom, 0.0, 1.0))
            lateral = qn - (start_n + t * direct)
        else:
            lateral = qn - start_n
        norm = float(np.linalg.norm(lateral))
        if norm > best_norm:
            best = lateral
            best_norm = norm
    if best_norm > 1.0e-6:
        return best / best_norm
    basis = np.eye(6, dtype=np.float64)
    if denom > 1.0e-12:
        direct_unit = direct / np.sqrt(denom)
        scores = np.abs(basis @ direct_unit)
        vec = basis[int(np.argmin(scores))]
        vec = vec - float(np.dot(vec, direct_unit)) * direct_unit
        return vec / max(1.0e-9, float(np.linalg.norm(vec)))
    return basis[0]


def _project_q_to_slice(
    kin: UR5Kinematics,
    q: np.ndarray,
    start_n: np.ndarray,
    direct: np.ndarray,
    lateral: np.ndarray,
) -> np.ndarray:
    qn = kin.normalize(np.asarray(q, dtype=np.float64)).astype(np.float64)
    offset = qn - start_n
    direct_denom = float(np.dot(direct, direct))
    a = float(np.dot(offset, direct) / direct_denom) if direct_denom > 1.0e-12 else 0.0
    b = float(np.dot(offset - a * direct, lateral))
    return np.asarray([a, b], dtype=np.float64)


def _field_tau_speed_gradients(
    model: ArmFieldModel,
    qn: np.ndarray,
    goal_n: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qn = np.asarray(qn, dtype=np.float32).reshape(-1, 6)
    goal_n = np.asarray(goal_n, dtype=np.float32).reshape(6)
    if len(qn) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
    tau_parts: list[np.ndarray] = []
    speed_parts: list[np.ndarray] = []
    grad_parts: list[np.ndarray] = []
    was_training = bool(model.model.network.training)
    model.model.network.train(False)
    for start in range(0, len(qn), max(1, int(batch_size))):
        end = min(len(qn), start + max(1, int(batch_size)))
        q0 = np.clip(qn[start:end], -0.5, 0.5)
        q1 = np.repeat(goal_n[None, :], len(q0), axis=0)
        xp_np = np.concatenate((q0, q1), axis=1).astype(np.float32)
        xp = torch.from_numpy(xp_np).float().to(model.device)
        xp.requires_grad_(True)
        tau_t, _w, xp_grad = model.model.network.out(xp)
        time_t = model.model.function.arrival_time(tau_t, xp_grad)
        dtime = model.model.function.gradient(time_t, xp_grad, create_graph=False)
        grad = dtime[:, :6]
        speed = torch.rsqrt(torch.sum(grad * grad, dim=1) + 1.0e-8)
        tau_parts.append(time_t.detach().cpu().numpy().reshape(-1).astype(np.float32))
        speed_parts.append(speed.detach().cpu().numpy().reshape(-1).astype(np.float32))
        grad_parts.append(grad.detach().cpu().numpy().astype(np.float32))
    model.model.network.train(was_training)
    return np.concatenate(tau_parts), np.concatenate(speed_parts), np.concatenate(grad_parts, axis=0)


def _save_topdown_field_plot(
    path: Path,
    kin: UR5Kinematics,
    model: ArmFieldModel,
    occupied: np.ndarray,
    depth_points: np.ndarray,
    scene_boxes: np.ndarray,
    leg_records: list[dict[str, object]],
    slice_leg: int,
    grid_size: int,
    lateral_span: float,
    value_kind: str,
) -> None:
    import matplotlib.pyplot as plt

    if not leg_records:
        raise RuntimeError("No leg records available for top-down plot")
    leg_idx = min(max(1, int(slice_leg)), len(leg_records))
    rec = leg_records[leg_idx - 1]
    start_q = np.asarray(rec["start_q"], dtype=np.float64)
    goal_q = np.asarray(rec["goal_q"], dtype=np.float64)
    plan = np.asarray(rec["plan"], dtype=np.float64).reshape(-1, 6)
    start_n = kin.normalize(start_q).astype(np.float64)
    goal_n = kin.normalize(goal_q).astype(np.float64)
    direct = goal_n - start_n
    direct_norm = max(1.0e-9, float(np.linalg.norm(direct)))
    lateral = _slice_lateral_direction(kin, start_q, goal_q, plan)
    a_vals = np.linspace(0.0, 1.0, max(3, int(grid_size)))
    b_vals = np.linspace(-float(lateral_span), float(lateral_span), max(3, int(grid_size)))
    aa, bb = np.meshgrid(a_vals, b_vals)
    qn = start_n[None, :] + aa.reshape(-1, 1) * direct[None, :] + bb.reshape(-1, 1) * lateral[None, :]
    qn = np.clip(qn, -0.5, 0.5).astype(np.float32)
    tau, speed, grad = _field_tau_speed_gradients(model, qn, goal_n.astype(np.float32), batch_size=4096)
    values = np.asarray(speed if value_kind == "speed" else tau, dtype=np.float32).reshape(aa.shape)
    tau_grid = tau.reshape(aa.shape)
    speed_grid = speed.reshape(aa.shape)
    grad_direct = grad @ (direct / direct_norm)
    grad_lateral = grad @ lateral
    descent_a = (-grad_direct / direct_norm).reshape(aa.shape)
    descent_b = (-grad_lateral).reshape(aa.shape)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("Field slice has no finite values")
    contour = ax.contourf(aa, bb, values, levels=32, cmap="viridis", alpha=0.85)
    line_values = tau_grid if value_kind == "tau" else speed_grid
    ax.contour(aa, bb, line_values, levels=14, colors="black", linewidths=0.45, alpha=0.55)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(f"field predicted {value_kind}")

    stride = max(1, int(grid_size) // 18)
    ax.quiver(
        aa[::stride, ::stride],
        bb[::stride, ::stride],
        descent_a[::stride, ::stride],
        descent_b[::stride, ::stride],
        color="white",
        alpha=0.75,
        angles="xy",
        scale_units="xy",
        scale=45.0,
        width=0.0025,
        label="-grad T projected",
    )

    for idx, item in enumerate(leg_records, start=1):
        if idx != leg_idx:
            continue
        color = LEG_COLORS[(idx - 1) % len(LEG_COLORS)]
        direct_states = np.asarray(item["direct_states"], dtype=np.float64).reshape(-1, 6)
        sampled_plan = np.asarray(item["sampled_plan"], dtype=np.float64).reshape(-1, 6)
        if len(direct_states) >= 2:
            direct_ab = np.asarray([_project_q_to_slice(kin, q, start_n, direct, lateral) for q in direct_states])
            ax.plot(direct_ab[:, 0], direct_ab[:, 1], "--", color="black", alpha=0.65, linewidth=1.8, label="direct joint edge")
        if len(sampled_plan) >= 2:
            plan_ab = np.asarray([_project_q_to_slice(kin, q, start_n, direct, lateral) for q in sampled_plan])
            ax.plot(plan_ab[:, 0], plan_ab[:, 1], "-", color=color, linewidth=3.2, label=f"leg {idx} planned")
        min_q = item.get("direct_min_q")
        if min_q is not None:
            ab = _project_q_to_slice(kin, np.asarray(min_q, dtype=np.float64), start_n, direct, lateral)
            ax.scatter([ab[0]], [ab[1]], c="red", s=90, marker="x", linewidths=2.2)

    ax.scatter([0.0], [0.0], c="white", edgecolors="black", s=55, marker="o", label="leg start")
    ax.scatter([1.0], [0.0], c="black", edgecolors="white", s=55, marker="*", label="leg goal")
    ax.set_xlabel("a: normalized joint-space progress start -> goal")
    ax.set_ylabel("b: normalized lateral joint-space direction")
    ax.set_title(
        f"Field slice for leg {leg_idx}: q = start + a*(goal-start) + b*lateral\n"
        f"color={value_kind}, black lines=contours, white arrows=-grad T"
    )
    ax.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visualization/test of learned UR5 field trajectories.")
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"), help="Training output root.")
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
    parser.add_argument("--field-path-joint-edge-weight", type=float, default=0.0)
    parser.add_argument("--field-path-tool-edge-weight", type=float, default=0.0)
    parser.add_argument("--field-path-clearance-penalty-weight", type=float, default=0.0)
    parser.add_argument("--field-path-clearance-soft-margin-m", type=float, default=0.04)
    parser.add_argument("--field-path-return-first-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--field-path-forward-probe-fraction", type=float, default=0.15)
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
        "--show-direct-collision",
        action="store_true",
        help="Draw the robot sphere pose and nearest obstacle at the minimum-clearance point of a direct start-goal edge.",
    )
    parser.add_argument(
        "--direct-collision-leg",
        type=int,
        default=0,
        help="Leg to diagnose with --show-direct-collision. 0 means every direct edge below clearance margin.",
    )
    parser.add_argument(
        "--show-direct-tool-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay grey straight tool-position lines between leg endpoints for comparison.",
    )
    parser.add_argument("--direct-line-radius", type=float, default=0.004, help="Radius for direct endpoint tool-line overlays.")
    parser.add_argument("--save-topdown-field", type=Path, default=None, help="Save a top-down XY field-slice diagnostic PNG.")
    parser.add_argument("--field-slice-leg", type=int, default=2, help="Leg used for the top-down field slice.")
    parser.add_argument("--field-slice-grid", type=int, default=61, help="Grid resolution for the top-down field slice.")
    parser.add_argument("--field-slice-lateral-span", type=float, default=0.18, help="Normalized joint-space lateral half-width for field slice.")
    parser.add_argument("--field-slice-value", choices=["tau", "speed"], default="tau")
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
    planner.set_score_weights(
        joint_edge=args.field_path_joint_edge_weight,
        tool_edge=args.field_path_tool_edge_weight,
        clearance_penalty=args.field_path_clearance_penalty_weight,
        clearance_soft_margin_m=args.field_path_clearance_soft_margin_m,
        return_first_goal=args.field_path_return_first_goal,
        forward_probe_fraction=args.field_path_forward_probe_fraction,
    )

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
    leg_records: list[dict[str, object]] = []
    print(f"sample_file={sample_file}")
    print(f"checkpoint={checkpoint}")
    print(f"clearance_backend={args.clearance_backend}")
    print(f"collision_cloud={args.collision_cloud} scene_boxes={len(scene_boxes)} occupied_points={len(occupied)} depth_points={len(depth_points)}")
    print(
        "planner_path_weights="
        f"joint_edge={args.field_path_joint_edge_weight:.3f} "
        f"tool_edge={args.field_path_tool_edge_weight:.3f} "
        f"clearance_penalty={args.field_path_clearance_penalty_weight:.3f} "
        f"clearance_soft_margin_m={args.field_path_clearance_soft_margin_m:.3f} "
        f"return_first_goal={args.field_path_return_first_goal} "
        f"forward_probe_fraction={args.field_path_forward_probe_fraction:.2f}"
    )
    print(f"unique_goals={len(goals_unique)} trajectory_legs={len(sequence)} return_to_first={args.return_to_first}")
    for leg_idx, q_goal in enumerate(sequence, start=1):
        color = LEG_COLORS[(leg_idx - 1) % len(LEG_COLORS)]
        checker = _checker_for_leg(current)
        leg_start = current.copy()
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
        path_details = _edge_clearance_details(checker, plan, args.edge_check_step_rad)
        min_clearance = float(path_details["min_clearance"])
        direct_plan = np.asarray([current, q_goal], dtype=np.float64)
        direct_details = _edge_clearance_details(checker, direct_plan, args.edge_check_step_rad)
        direct_joint_min_clearance = float(direct_details["min_clearance"])
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
        direct_min_q = direct_details.get("min_q")
        if direct_min_q is not None and (
            args.show_direct_collision
            and (args.direct_collision_leg == 0 or args.direct_collision_leg == leg_idx)
            and direct_joint_min_clearance < args.clearance_margin_m
        ):
            contact_geoms, contact = _contact_geometries(checker, np.asarray(direct_min_q, dtype=np.float64), args.max_spheres, args.radius_scale)
            geoms.extend(contact_geoms)
            if contact is not None:
                print(
                    f"direct_collision leg={leg_idx} min_clearance_m={direct_joint_min_clearance:.4f} "
                    f"min_idx={direct_details.get('min_idx')} min_q={np.round(np.asarray(direct_min_q), 3).tolist()} "
                    f"sphere_idx={contact['sphere_idx']} sphere_center={np.round(np.asarray(contact['center']), 4).tolist()} "
                    f"sphere_radius={float(contact['radius']):.4f} obstacle={contact['obstacle']} "
                    f"obstacle_idx={contact['obstacle_idx']} nearest={np.round(np.asarray(contact['nearest']), 4).tolist()} "
                    "colors: direct_collision robot=grey, min sphere=red, nearest obstacle=magenta"
                )
        if len(plan) == 0:
            break
        tool_points = tool_points_for_stats
        path_speed, path_tau = _field_values_to_goal(model, kin, sampled_plan, q_goal)
        direct_states = np.asarray(direct_details["states"], dtype=np.float64).reshape(-1, 6)
        direct_tool_points = np.asarray([kin.fk(q)[:3, 3] for q in direct_states], dtype=np.float64) if len(direct_states) else np.asarray([start_tool, goal_tool], dtype=np.float64)
        direct_speed, direct_tau = _field_values_to_goal(model, kin, direct_states, q_goal)
        leg_records.append(
            {
                "leg_idx": leg_idx,
                "start_q": leg_start,
                "goal_q": np.asarray(q_goal, dtype=np.float64).copy(),
                "plan": np.asarray(plan, dtype=np.float64).copy(),
                "sampled_plan": sampled_plan.copy(),
                "tool_points": tool_points.copy(),
                "path_speed": path_speed,
                "path_tau": path_tau,
                "direct_states": direct_states.copy(),
                "direct_tool_points": direct_tool_points,
                "direct_speed": direct_speed,
                "direct_tau": direct_tau,
                "direct_min_q": None if direct_min_q is None else np.asarray(direct_min_q, dtype=np.float64).copy(),
                "direct_min_clearance": direct_joint_min_clearance,
            }
        )
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

    if args.save_topdown_field is not None:
        _save_topdown_field_plot(
            args.save_topdown_field.expanduser().resolve(),
            kin,
            model,
            occupied,
            depth_points,
            scene_boxes,
            leg_records,
            args.field_slice_leg,
            args.field_slice_grid,
            args.field_slice_lateral_span,
            args.field_slice_value,
        )
        print(
            f"wrote_topdown_field={args.save_topdown_field.expanduser().resolve()} "
            "note=plot axes are a 2D normalized joint-space slice, not world XY"
        )

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
