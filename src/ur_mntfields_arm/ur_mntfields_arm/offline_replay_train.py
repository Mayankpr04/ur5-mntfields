from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


def _frame_pair_rows(frame_data: np.ndarray) -> np.ndarray:
    rows = np.asarray(frame_data, dtype=np.float32)
    if rows.ndim != 2 or len(rows) == 0:
        return np.zeros((0, 26), dtype=np.float32)
    if rows.shape[1] == 26:
        return rows.astype(np.float32, copy=False)
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


def _sample_files(root: Path, latest_only: bool, max_files: int) -> list[Path]:
    files = sorted((root / "samples").glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sample dumps found under {root / 'samples'}")
    if latest_only:
        return [files[-1]]
    if max_files > 0:
        return files[-max_files:]
    return files


def _latest_checkpoint(model_dir: Path) -> Path | None:
    final_path = model_dir / "weights_final.pt"
    if final_path.exists():
        return final_path
    epoch_files = sorted(model_dir.glob("weights_epoch_*.pt"))
    if epoch_files:
        return epoch_files[-1]
    partial_path = model_dir / "weights_partial.pt"
    if partial_path.exists():
        return partial_path
    return None


def _load_replay_rows(files: list[Path]) -> tuple[np.ndarray, dict[str, int]]:
    rows_list: list[np.ndarray] = []
    counts = {"files": 0, "rows": 0, "skipped_files": 0}
    for path in files:
        try:
            data = np.load(path)
            if "frame_data" not in data:
                counts["skipped_files"] += 1
                continue
            rows = _frame_pair_rows(data["frame_data"])
        except Exception:
            counts["skipped_files"] += 1
            continue
        if len(rows) == 0:
            counts["skipped_files"] += 1
            continue
        rows_list.append(rows)
        counts["files"] += 1
        counts["rows"] += int(len(rows))
    if not rows_list:
        return np.zeros((0, 26), dtype=np.float32), counts
    return np.concatenate(rows_list, axis=0).astype(np.float32, copy=False), counts


def _train_replay_steps(model: ArmFieldModel, steps: int) -> float | None:
    last_loss = None
    for _ in range(max(0, int(steps))):
        batch = model._sample_replay_rows(model.minibatch_size)
        if batch.size == 0:
            continue
        pair_batch = model.build_pair_batch(batch)
        pair_batch = model.reshuffle_pair_endpoints(pair_batch)
        if pair_batch.size == 0:
            continue
        model.last_train_batch_size = int(len(batch))
        model.last_train_pair_count = int(len(pair_batch))
        batch_t = torch.from_numpy(pair_batch).float().to(model.device)
        last_loss = model.model.train_batch(batch_t)
        if last_loss is not None:
            model.loss_history.append(float(last_loss))
        model.total_epochs_trained += 1
    return None if last_loss is None else float(last_loss)


def _print_diag(prefix: str, diag: dict[str, float]) -> None:
    keys = (
        "diag_rows",
        "speed_mae",
        "speed_corr",
        "pred_near_mean",
        "pred_far_mean",
        "near_far_gap",
        "normal_cos_near_mean",
        "grad_goal_cos_mean",
        "grad_goal_neg_frac",
        "low_target_overpred_frac",
    )
    fields = []
    for key in keys:
        if key in diag:
            value = float(diag[key])
            if key == "diag_rows":
                fields.append(f"{key}={int(value)}")
            else:
                fields.append(f"{key}={value:.4f}")
    print(f"{prefix}: " + " ".join(fields), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline fine-tune UR5 MNTFields from saved replay dumps.")
    parser.add_argument("--root", type=Path, default=Path("src/ur5_sim_training_factorized_v2"))
    parser.add_argument("--output-model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="auto", help="'auto', 'none', or a checkpoint path.")
    parser.add_argument("--steps", type=int, default=300, help="Extra optimizer steps to run.")
    parser.add_argument("--chunk-steps", type=int, default=60, help="Train/evaluate chunk size.")
    parser.add_argument("--checkpoint-every", type=int, default=60)
    parser.add_argument("--diagnostics-every", type=int, default=60)
    parser.add_argument("--max-diagnostic-rows", type=int, default=4096)
    parser.add_argument("--latest-only", action="store_true", help="Use only the latest sample dump.")
    parser.add_argument("--max-sample-files", type=int, default=0, help="Use only the latest N sample files. 0 means all.")
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--replay-ratio", type=float, default=1.0, help="Offline replay-only training should usually use 1.0.")
    parser.add_argument("--td-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-loss-weight", type=float, default=1.0e-2)
    parser.add_argument("--log-speed-loss-weight", type=float, default=0.0)
    parser.add_argument("--near-obstacle-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-speed-threshold", type=float, default=0.20)
    parser.add_argument("--low-speed-pred-max", type=float, default=0.30)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=0.0)
    parser.add_argument("--normal-loss-weight", type=float, default=0.0)
    parser.add_argument("--effective-speed-floor", type=float, default=0.10)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_model_dir = (
        args.output_model_dir.expanduser().resolve()
        if args.output_model_dir is not None
        else (root / "model_offline").resolve()
    )
    output_model_dir.mkdir(parents=True, exist_ok=True)

    files = _sample_files(root, bool(args.latest_only), max(0, int(args.max_sample_files)))
    rows, counts = _load_replay_rows(files)
    if len(rows) == 0:
        raise RuntimeError(f"No valid frame_data rows loaded from {root / 'samples'}")

    print(
        f"loaded_replay files={counts['files']} skipped={counts['skipped_files']} "
        f"input_rows={counts['rows']} output_model_dir={output_model_dir}",
        flush=True,
    )

    model = ArmFieldModel(
        model_dir=str(output_model_dir),
        device="cuda:0",
        replay_capacity=max(int(args.replay_capacity), int(len(rows))),
        minibatch_size=max(1, int(args.minibatch_size)),
        replay_ratio=float(np.clip(float(args.replay_ratio), 0.0, 1.0)),
        td_loss_weight=float(args.td_loss_weight),
        speed_loss_weight=float(args.speed_loss_weight),
        log_speed_loss_weight=float(args.log_speed_loss_weight),
        direct_speed_loss_weight=0.0,
        normal_loss_weight=float(args.normal_loss_weight),
        normal_cos_loss_weight=0.0,
        near_obstacle_loss_weight=float(args.near_obstacle_loss_weight),
        low_speed_threshold=float(args.low_speed_threshold),
        low_speed_pred_max=float(args.low_speed_pred_max),
        low_speed_penalty_weight=float(args.low_speed_penalty_weight),
        effective_speed_floor=float(args.effective_speed_floor),
    )

    checkpoint_path = None
    if args.checkpoint == "auto":
        checkpoint_path = _latest_checkpoint(root / "model")
    elif args.checkpoint != "none":
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model.load_checkpoint(checkpoint_path)
        print(f"loaded_checkpoint={checkpoint_path} total_epochs_trained={model.total_epochs_trained}", flush=True)
    else:
        print("loaded_checkpoint=none training_from_scratch", flush=True)

    model.add_rows(rows)
    print(f"replay_size={model.replay_size} replay_capacity={model.replay_capacity}", flush=True)

    initial_diag = model.evaluate_replay_diagnostics(max_rows=max(1, int(args.max_diagnostic_rows)))
    _print_diag("initial_diagnostics", initial_diag)

    total_steps = max(0, int(args.steps))
    chunk_steps = max(1, int(args.chunk_steps))
    checkpoint_every = max(0, int(args.checkpoint_every))
    diagnostics_every = max(0, int(args.diagnostics_every))
    trained = 0
    t0 = time.perf_counter()
    while trained < total_steps:
        this_chunk = min(chunk_steps, total_steps - trained)
        chunk_t0 = time.perf_counter()
        loss = _train_replay_steps(model, this_chunk)
        trained += this_chunk
        chunk_ms = (time.perf_counter() - chunk_t0) * 1e3
        print(
            f"offline_train progress={trained}/{total_steps} total_epochs_trained={model.total_epochs_trained} "
            f"loss={-1.0 if loss is None else loss:.6f} chunk_ms={chunk_ms:.1f}",
            flush=True,
        )

        if diagnostics_every > 0 and (trained % diagnostics_every == 0 or trained >= total_steps):
            diag = model.evaluate_replay_diagnostics(max_rows=max(1, int(args.max_diagnostic_rows)))
            _print_diag(f"diagnostics_after_{trained}", diag)

        if checkpoint_every > 0 and (trained % checkpoint_every == 0 or trained >= total_steps):
            ckpt = output_model_dir / f"weights_offline_epoch_{model.total_epochs_trained:06d}.pt"
            model.save_checkpoint(ckpt)
            print(f"saved_checkpoint={ckpt}", flush=True)

    final_path = output_model_dir / "weights_offline_final.pt"
    model.save_checkpoint(final_path)
    loss_plot = output_model_dir / "offline_loss.png"
    model.save_loss_plot(loss_plot)
    final_diag = model.evaluate_replay_diagnostics(max_rows=max(1, int(args.max_diagnostic_rows)))
    _print_diag("final_diagnostics", final_diag)
    summary = {
        "root": str(root),
        "output_model_dir": str(output_model_dir),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "steps": int(total_steps),
        "total_epochs_trained": int(model.total_epochs_trained),
        "replay_size": int(model.replay_size),
        "final_checkpoint": str(final_path),
        "final_diagnostics": final_diag,
    }
    summary_path = output_model_dir / "offline_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved_final_checkpoint={final_path}", flush=True)
    print(f"saved_loss_plot={loss_plot}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)

    if args.save_plots:
        diag_dir = output_model_dir / "field_diagnostics"
        saved = model.save_replay_diagnostic_plots(
            diag_dir,
            UR5Kinematics(),
            step_label=f"offline_epoch_{model.total_epochs_trained:06d}",
            max_rows=max(1, int(args.max_diagnostic_rows)),
        )
        print(f"saved_diagnostic_plots={len(saved)} dir={diag_dir}", flush=True)

    print(f"total_ms={(time.perf_counter() - t0) * 1e3:.1f}", flush=True)


if __name__ == "__main__":
    main()
