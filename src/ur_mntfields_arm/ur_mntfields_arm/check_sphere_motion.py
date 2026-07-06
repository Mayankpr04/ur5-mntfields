from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


def _parse_q(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.replace(",", " ").split() if v.strip()]
    if len(vals) != 6:
        raise ValueError(f"Expected six joint values, got {len(vals)}: {text}")
    return np.asarray(vals, dtype=np.float32)


def _default_qs() -> np.ndarray:
    return np.asarray(
        [
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            [0.35, -1.10, 1.10, -1.55, -1.57, 0.0],
            [1.0, -0.8, 1.8, -0.8, 0.3, 1.2],
        ],
        dtype=np.float32,
    )


def run_check(qs: np.ndarray, ur_type: str, occupied_npz: Path | None = None) -> dict[str, float]:
    kin = UR5Kinematics(ur_type=ur_type)
    occupied = np.zeros((0, 3), dtype=np.float32)
    if occupied_npz is not None:
        payload = np.load(occupied_npz)
        if "occupied_points" not in payload:
            raise KeyError(f"{occupied_npz} does not contain occupied_points")
        occupied = np.asarray(payload["occupied_points"], dtype=np.float32)
    checker = UR5PointCloudCollisionChecker(kin, occupied)

    cpu_centers = []
    for idx, q in enumerate(qs):
        centers, radii = checker.robot_spheres(q)
        cpu_centers.append(centers)
        print(
            f"pose={idx} sphere_count={len(centers)} "
            f"bbox_min={np.array2string(centers.min(axis=0), precision=4)} "
            f"bbox_max={np.array2string(centers.max(axis=0), precision=4)} "
            f"radius_range=({float(radii.min()):.4f},{float(radii.max()):.4f})"
        )

    moving_fracs = []
    mean_deltas = []
    for idx in range(len(cpu_centers) - 1):
        delta = np.linalg.norm(cpu_centers[idx + 1] - cpu_centers[idx], axis=1)
        moving_frac = float(np.mean(delta > 0.01))
        moving_fracs.append(moving_frac)
        mean_deltas.append(float(np.mean(delta)))
        print(
            f"pose_delta={idx}->{idx + 1} "
            f"min={float(delta.min()):.6f} mean={float(delta.mean()):.6f} "
            f"max={float(delta.max()):.6f} moving_frac_gt_1cm={moving_frac:.6f}"
        )

    q_t = torch.as_tensor(qs, dtype=torch.float32, device=checker.device)
    torch_centers, _radii, _link_ids, _joint_frames = checker._sphere_samples_batch_torch(q_t)
    torch_centers_np = torch_centers.detach().cpu().numpy()
    max_cpu_torch_err = 0.0
    for idx, centers in enumerate(cpu_centers):
        err = float(np.max(np.linalg.norm(centers - torch_centers_np[idx], axis=1)))
        max_cpu_torch_err = max(max_cpu_torch_err, err)
        print(f"cpu_torch_center_max_error pose={idx} error_m={err:.9e}")

    if len(occupied):
        clearances, normals = checker.clearance_and_normal_batch(qs)
        print("clearance_m=" + " ".join(f"{float(v):.6f}" for v in clearances.tolist()))
        print("normal_norm=" + " ".join(f"{float(v):.6f}" for v in np.linalg.norm(normals, axis=1).tolist()))

    return {
        "min_moving_frac_gt_1cm": min(moving_fracs) if moving_fracs else 0.0,
        "mean_delta_m": float(np.mean(mean_deltas)) if mean_deltas else 0.0,
        "max_cpu_torch_err_m": max_cpu_torch_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify UR5 collision spheres move with sampled joint states.")
    parser.add_argument("--ur-type", default="ur5", help="UR type passed to UR5Kinematics.")
    parser.add_argument(
        "--q",
        action="append",
        default=[],
        help="Six joint radians. Pass multiple --q values; defaults to three built-in poses.",
    )
    parser.add_argument(
        "--occupied-npz",
        default="",
        help="Optional sample npz containing occupied_points for clearance labels.",
    )
    args = parser.parse_args()
    qs = np.asarray([_parse_q(q) for q in args.q], dtype=np.float32) if args.q else _default_qs()
    occupied_npz = Path(args.occupied_npz).expanduser() if args.occupied_npz else None
    summary = run_check(qs, args.ur_type, occupied_npz=occupied_npz)
    print(
        "summary "
        + " ".join(f"{key}={value:.9g}" for key, value in sorted(summary.items()))
    )


if __name__ == "__main__":
    main()
