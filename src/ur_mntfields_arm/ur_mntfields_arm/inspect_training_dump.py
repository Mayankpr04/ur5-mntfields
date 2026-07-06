from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


def _latest_step(samples_dir: Path) -> Path:
    files = sorted(samples_dir.glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files found in {samples_dir}")
    return files[-1]


def _print_stats(name: str, arr: np.ndarray):
    print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
    if arr.size == 0:
        return
    flat = arr.reshape(-1)
    print(
        f"  min={float(flat.min()):.6f} max={float(flat.max()):.6f} "
        f"mean={float(flat.mean()):.6f}"
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _frame_pair_rows(frame_data: np.ndarray) -> np.ndarray:
    rows = np.asarray(frame_data, dtype=np.float32)
    if rows.ndim != 2 or len(rows) == 0:
        return np.zeros((0, 26), dtype=np.float32)
    if rows.shape[1] == 26:
        return rows
    if rows.shape[1] != 13 or len(rows) < 2:
        return np.zeros((0, 26), dtype=np.float32)
    even_count = (len(rows) // 2) * 2
    rows = rows[:even_count]
    return np.concatenate(
        (
            rows[0::2, :6],
            rows[1::2, :6],
            rows[0::2, 6:7],
            rows[1::2, 6:7],
            rows[0::2, 7:],
            rows[1::2, 7:],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _label_diagnostics(pair_rows: np.ndarray) -> dict[str, float]:
    rows = np.asarray(pair_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
        return {"rows": 0.0}
    speeds = np.clip(rows[:, 12:14], 0.0, 1.0)
    n0 = np.linalg.norm(rows[:, 14:20], axis=1)
    n1 = np.linalg.norm(rows[:, 20:26], axis=1)
    return {
        "rows": float(len(rows)),
        "speed_min": float(np.min(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_mean": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "speed_le_0p1_frac": float(np.mean(speeds <= 0.10)),
        "speed_ge_0p999_frac": float(np.mean(speeds >= 0.999)),
        "normal0_norm_mean": float(np.mean(n0)),
        "normal1_norm_mean": float(np.mean(n1)),
        "normal0_bad_frac": float(np.mean(~np.isfinite(n0) | (n0 < 0.95) | (n0 > 1.05))),
        "normal1_bad_frac": float(np.mean(~np.isfinite(n1) | (n1 < 0.95) | (n1 > 1.05))),
    }


def _model_diagnostics(root: Path, checkpoint: Path, pair_rows: np.ndarray, max_rows: int) -> dict[str, float]:
    rows = np.asarray(pair_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
        return {"diag_rows": 0.0}
    if len(rows) > max_rows:
        rng = np.random.default_rng(7)
        rows = rows[rng.choice(len(rows), size=max_rows, replace=False)]
    model = ArmFieldModel(model_dir=str(root / "model"), device="cuda:0")
    model.load_checkpoint(checkpoint)
    model.add_rows(rows)
    diag = model.evaluate_replay_diagnostics(max_rows=max_rows)
    target = model._training_speed_target(np.clip(rows[:, 12:14], 0.0, 1.0))
    pred0, pred1 = model.predict_normalized_pair_speeds(rows[:, :6], rows[:, 6:12])
    pred = np.stack((pred0, pred1), axis=1)
    diag["speed_corr_raw_label"] = _safe_corr(np.clip(pred, 0.0, 2.0), rows[:, 12:14])
    diag["speed_corr_training_target"] = _safe_corr(np.clip(pred, 0.0, 2.0), target)
    return diag


def _write_rows_csv(path: Path, frame_data: np.ndarray):
    headers = (
        [f"q0_{i}" for i in range(6)]
        + [f"q1_{i}" for i in range(6)]
        + ["speed0", "speed1"]
        + [f"normal0_{i}" for i in range(6)]
        + [f"normal1_{i}" for i in range(6)]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(_frame_pair_rows(frame_data).tolist())


def _write_speed_csv(path: Path, frame_data: np.ndarray):
    speeds = _frame_pair_rows(frame_data)[:, 12:14]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "speed0", "speed1"])
        for idx, row in enumerate(speeds.tolist()):
            writer.writerow([idx, row[0], row[1]])


def main():
    parser = argparse.ArgumentParser(description="Inspect saved UR MNTFields training dumps.")
    parser.add_argument(
        "--root",
        default="/home/mayank/ur_ws/src/ur5_sim_training",
        help="Training output root directory.",
    )
    parser.add_argument(
        "--step",
        default="latest",
        help='Step id like "000015" or "latest".',
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export frame_data and speed columns to CSV next to the chosen sample file.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="auto",
        help='Checkpoint path, "auto" for <root>/model/weights_final.pt, or "none".',
    )
    parser.add_argument(
        "--max-model-rows",
        type=int,
        default=4096,
        help="Maximum rows used for model prediction diagnostics.",
    )
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write JSON and diagnostic plots under <root>/model/field_diagnostics.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    samples_dir = root / "samples"
    sample_path = _latest_step(samples_dir) if args.step == "latest" else samples_dir / f"step_{args.step}.npz"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    print(f"sample_file={sample_path}")
    data = np.load(sample_path)
    for key in data.files:
        _print_stats(key, data[key])

    frame_data = data["frame_data"]
    pair_rows = _frame_pair_rows(frame_data)
    if frame_data.size:
        if len(pair_rows):
            speeds = pair_rows[:, 12:14]
            print("speed preview:")
            for idx, row in enumerate(speeds[:10]):
                print(f"  row={idx} speed0={row[0]:.6f} speed1={row[1]:.6f}")
            label_diag = _label_diagnostics(pair_rows)
            print("label diagnostics:")
            for key in sorted(label_diag):
                print(f"  {key}={label_diag[key]:.6f}")
        else:
            print(f"label diagnostics: unsupported frame_data width={frame_data.shape[1] if frame_data.ndim == 2 else 'n/a'}")

    checkpoint = None
    if args.model_checkpoint == "auto":
        auto_path = root / "model" / "weights_final.pt"
        checkpoint = auto_path if auto_path.exists() else None
    elif args.model_checkpoint != "none":
        checkpoint = Path(args.model_checkpoint).expanduser()

    model_diag = None
    if checkpoint is not None and len(pair_rows):
        print(f"model_checkpoint={checkpoint}")
        model_diag = _model_diagnostics(root, checkpoint, pair_rows, max(1, int(args.max_model_rows)))
        print("model diagnostics:")
        for key in sorted(model_diag):
            print(f"  {key}={float(model_diag[key]):.6f}")

    pcd_dir = root / "PCD"
    image_dir = root / "Images"
    step_token = sample_path.stem
    print(f"depth_pcd={pcd_dir / (step_token + '_depth_world.pcd')}")
    print(f"occupied_pcd={pcd_dir / (step_token + '_occupied_world.pcd')}")
    print(f"color_overlay={image_dir / (step_token + '_color_frontiers.png')}")
    print(f"depth_overlay={image_dir / (step_token + '_depth_frontiers.png')}")

    if args.export_csv:
        rows_csv = sample_path.with_name(sample_path.stem + "_frame_data.csv")
        speeds_csv = sample_path.with_name(sample_path.stem + "_speeds.csv")
        _write_rows_csv(rows_csv, frame_data)
        _write_speed_csv(speeds_csv, frame_data)
        print(f"wrote_csv={rows_csv}")
        print(f"wrote_csv={speeds_csv}")

    if args.write_diagnostics and len(pair_rows):
        diag_dir = root / "model" / "field_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        summary = {"sample_file": str(sample_path), "labels": _label_diagnostics(pair_rows)}
        if model_diag is not None:
            summary["model"] = model_diag
        out_json = diag_dir / f"inspect_{sample_path.stem}.json"
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote_diagnostics={out_json}")
        if checkpoint is not None:
            model = ArmFieldModel(model_dir=str(root / "model"), device="cuda:0")
            model.load_checkpoint(checkpoint)
            model.add_rows(pair_rows)
            saved = model.save_replay_diagnostic_plots(
                diag_dir,
                UR5Kinematics(),
                step_label=f"inspect_{sample_path.stem}",
                max_rows=max(1, int(args.max_model_rows)),
            )
            print(f"wrote_plots={len(saved)}")


if __name__ == "__main__":
    main()
