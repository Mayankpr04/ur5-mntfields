from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import JOINT_NAMES, UR5Kinematics


LINK_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]


def _parse_q(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.replace(",", " ").split() if v.strip()]
    if len(vals) != 6:
        raise ValueError(f"Expected six joint values, got {len(vals)}: {text}")
    return np.asarray(vals, dtype=np.float64)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(origin) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if origin is None:
        return out
    xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
    out[:3, :3] = _rpy_matrix(*rpy)
    out[:3, 3] = xyz
    return out


def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("file://"):
        return Path(filename[len("file://") :])
    candidate = (urdf_path.parent / filename).resolve()
    if candidate.exists():
        return candidate
    fallback_names = {
        "upperarm.stl": "upper_arm.stl",
        "wrist1.stl": "wrist_1.stl",
        "wrist2.stl": "wrist_2.stl",
        "wrist3.stl": "wrist_3.stl",
    }
    fallback = Path("/home/mayank/ur_ws/ntrl-demo/datasets/arm/UR5/meshes/collision") / fallback_names.get(
        Path(filename).name, Path(filename).name
    )
    if fallback.exists():
        return fallback
    return candidate


def _collision_mesh_specs(urdf_path: Path) -> dict[str, tuple[Path, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    specs: dict[str, tuple[Path, np.ndarray]] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if name not in LINK_NAMES:
            continue
        collision = link.find("collision")
        if collision is None:
            continue
        geometry = collision.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or "filename" not in mesh.attrib:
            continue
        specs[name] = (
            _resolve_mesh_path(mesh.attrib["filename"], urdf_path),
            _origin_matrix(collision.find("origin")),
        )
    return specs


def _add_mesh(ax, mesh: trimesh.Trimesh, color: tuple[float, float, float, float]):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > 2000:
        step = max(1, len(faces) // 2000)
        faces = faces[::step]
    polys = vertices[faces]
    coll = Poly3DCollection(polys, facecolors=[color], edgecolors=(0.15, 0.15, 0.15, 0.08), linewidths=0.1)
    ax.add_collection3d(coll)


def _set_equal_axes(ax, points: np.ndarray):
    pts = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.25, 0.55 * float(np.max(maxs - mins)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_overlay(
    q: np.ndarray,
    out_path: Path,
    ur_type: str = "ur5",
    urdf_path: Path = Path("/home/mayank/ur_ws/curobo_configs/ur5_wrist_camera.urdf"),
) -> Path:
    kin = UR5Kinematics(ur_type=ur_type)
    checker = UR5PointCloudCollisionChecker(kin, np.zeros((0, 3), dtype=np.float32))
    frames = checker._urdf_link_frames_np(q)
    sphere_centers, sphere_radii = checker.robot_spheres(q)
    mesh_specs = _collision_mesh_specs(urdf_path)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    all_points = [sphere_centers]
    colors = plt.get_cmap("tab10")

    for idx, link_name in enumerate(LINK_NAMES):
        mesh_path, collision_origin = mesh_specs[link_name]
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        mesh.apply_transform(frames[idx] @ collision_origin)
        all_points.append(np.asarray(mesh.vertices))
        _add_mesh(ax, mesh, colors(idx % 10, alpha=0.22))

    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=float(radius))
        sphere.apply_translation(center)
        _add_mesh(ax, sphere, (0.95, 0.25, 0.05, 0.28))

    ax.scatter(
        sphere_centers[:, 0],
        sphere_centers[:, 1],
        sphere_centers[:, 2],
        c="red",
        s=np.maximum(8.0, 1600.0 * sphere_radii),
        depthshade=False,
        label="sphere centers",
    )
    ee = kin.fk(q)
    ax.scatter([ee[0, 3]], [ee[1, 3]], [ee[2, 3]], c="black", s=60, label="tool0")
    ax.set_title("UR5 collision meshes with NTFields sphere model")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=25, azim=-55)
    ax.legend(loc="upper right")
    _set_equal_axes(ax, np.vstack(all_points))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Save one UR5 URDF mesh + collision sphere overlay image.")
    parser.add_argument(
        "--q",
        default="0.35 -1.10 1.10 -1.55 -1.57 0.0",
        help="Six joint values in radians, separated by spaces or commas.",
    )
    parser.add_argument("--ur-type", default="ur5", help="UR description type, default ur5.")
    parser.add_argument(
        "--urdf",
        default="/home/mayank/ur_ws/curobo_configs/ur5_wrist_camera.urdf",
        help="Flattened URDF used for collision mesh origins.",
    )
    parser.add_argument(
        "--out",
        default="/home/mayank/ur_ws/ur5_sphere_urdf_overlay.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    q = _parse_q(args.q)
    out = render_overlay(q, Path(args.out).expanduser(), ur_type=args.ur_type, urdf_path=Path(args.urdf).expanduser())
    print(f"wrote_overlay={out}")
    print("joint_order=" + ",".join(JOINT_NAMES))
    print("q_rad=" + " ".join(f"{v:.6f}" for v in q.tolist()))


if __name__ == "__main__":
    main()
