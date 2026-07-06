import argparse
import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View 3D arrival-time contours with an overlaid trajectory.")
    parser.add_argument("--model-dir", type=Path, required=True, help="trained_field directory.")
    parser.add_argument("--path-name", type=str, default="trajectory", help="Trajectory prefix, e.g. trajectory.")
    parser.add_argument("--line-width", type=float, default=1.0, help="Unused placeholder for future viewer backends.")
    return parser.parse_args()


def _load_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        return None
    mesh.compute_vertex_normals()
    return mesh


def _load_line_set(path: Path):
    line_set = o3d.io.read_line_set(str(path))
    if line_set.is_empty():
        return None
    return line_set


def main() -> None:
    if o3d is None:
        raise RuntimeError("open3d is required for 3D field viewing.")

    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    contour_dir = model_dir / "contours"
    geoms = []

    manifest_path = contour_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest.get("contours", []):
            mesh = _load_mesh(contour_dir / entry["file"])
            if mesh is not None:
                geoms.append(mesh)
    else:
        for p in sorted(contour_dir.glob("contour_*.ply")):
            mesh = _load_mesh(p)
            if mesh is not None:
                geoms.append(mesh)

    line_set = _load_line_set(model_dir / f"{args.path_name}.ply")
    if line_set is not None:
        geoms.append(line_set)

    for suffix in ("_start.ply", "_goal.ply"):
        mesh = _load_mesh(model_dir / f"{args.path_name}{suffix}")
        if mesh is not None:
            geoms.append(mesh)

    source_mesh = _load_mesh(model_dir / "field_source_marker.ply")
    if source_mesh is not None:
        geoms.append(source_mesh)

    if not geoms:
        raise FileNotFoundError(f"No contours/path geometry found under {model_dir}")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geoms.append(frame)
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
