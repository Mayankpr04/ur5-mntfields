from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ur_mntfields_arm.tb_core.model_base import Model


class ArmFieldModel:
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        replay_capacity: int = 300000,
        minibatch_size: int = 512,
        replay_ratio: float = 0.75,
        priority_ratio: float = 0.0,
        gradient_accumulation_steps: int = 1,
        td_loss_weight: float = 1.0e-2,
        speed_loss_weight: float = 1.0e-2,
        log_speed_loss_weight: float = 0.0,
        direct_speed_loss_weight: float = 0.0,
        normal_loss_weight: float = 0.0,
        normal_cos_loss_weight: float = 0.0,
        near_obstacle_loss_weight: float = 0.0,
        low_speed_threshold: float = 0.20,
        low_speed_pred_max: float = 0.35,
        low_speed_penalty_weight: float = 0.0,
        effective_speed_floor: float = 0.05,
        state_clearance_loss_weight: float = 1.0,
        unsafe_loss_weight: float = 0.5,
        shell_normal_loss_weight: float = 0.1,
        false_free_loss_weight: float = 2.0,
        consistency_loss_weight: float = 0.01,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.loss_config = {
            "td_loss_weight": float(td_loss_weight),
            "speed_loss_weight": float(speed_loss_weight),
            "log_speed_loss_weight": float(log_speed_loss_weight),
            "direct_speed_loss_weight": float(direct_speed_loss_weight),
            "normal_loss_weight": float(normal_loss_weight),
            "normal_cos_loss_weight": float(normal_cos_loss_weight),
            "near_obstacle_loss_weight": float(near_obstacle_loss_weight),
            "low_speed_threshold": float(low_speed_threshold),
            "low_speed_pred_max": float(low_speed_pred_max),
            "low_speed_penalty_weight": float(low_speed_penalty_weight),
            "effective_speed_floor": float(effective_speed_floor),
            "state_clearance_loss_weight": float(state_clearance_loss_weight),
            "unsafe_loss_weight": float(unsafe_loss_weight),
            "shell_normal_loss_weight": float(shell_normal_loss_weight),
            "false_free_loss_weight": float(false_free_loss_weight),
            "consistency_loss_weight": float(consistency_loss_weight),
        }
        self.model = Model(
            folder=str(self.model_dir),
            dim=6,
            B_scale=0.2,
            device=self.device,
            lr=5e-4,
            **self.loss_config,
        )
        self.loss_history: list[float] = []
        self.dim = 6
        self.learning_rate = 5e-4
        self.total_epochs_trained = 0
        self.last_checkpoint_epoch = 0
        self.replay_capacity = max(1, int(replay_capacity))
        self.minibatch_size = max(1, int(minibatch_size))
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.effective_minibatch_size = self.minibatch_size * self.gradient_accumulation_steps
        self.replay_ratio = float(np.clip(replay_ratio, 0.0, 1.0))
        self.priority_ratio = float(np.clip(priority_ratio, 0.0, 0.75))
        self.replay_buffer: np.ndarray | None = None
        self.replay_size = 0
        self.replay_insert_idx = 0
        self.last_train_batch_size = 0
        self.last_train_pair_count = 0
        self.last_diagnostics: dict[str, float] = {}
        self.checkpoint_metadata: dict[str, object] = {}
        self.calibration_margins: dict[str, float] = {}
        self.coverage_states = np.zeros((0, 6), dtype=np.float32)
        self.coverage_landmarks = np.zeros((0, 6), dtype=np.float32)
        self.coverage_tree = None
        self.shell_coverage_radius = 0.0
        self.free_coverage_radius = 0.0
        self.certification_passed = False
        self.legacy_checkpoint = False
        from ur_mntfields_arm.online_training import StateReplay
        self.state_replay = StateReplay(capacity=self.replay_capacity)

    def _ensure_replay_buffer(self, row_width: int):
        if self.replay_buffer is None:
            self.replay_buffer = np.zeros((self.replay_capacity, row_width), dtype=np.float32)

    def _valid_replay_rows(self) -> np.ndarray:
        if self.replay_buffer is None or self.replay_size <= 0:
            return np.zeros((0, 26), dtype=np.float32)
        return self.replay_buffer[: self.replay_size].astype(np.float32, copy=False)

    def _cap_replay_rows(self, merged: np.ndarray, max_rows: int) -> np.ndarray:
        rows = np.asarray(merged, dtype=np.float32)
        if rows.ndim != 2 or len(rows) <= max_rows:
            return rows
        # Algorithm 1 retains and samples memory uniformly. Clearance/cell
        # quotas distort the empirical speed distribution toward obstacles.
        idx = np.random.choice(len(rows), int(max_rows), replace=False)
        return rows[idx].astype(np.float32, copy=False)

    def add_rows(self, frame_data: np.ndarray):
        rows = np.asarray(frame_data, dtype=np.float32)
        if rows.ndim != 2 or len(rows) == 0:
            return
        if rows.shape[1] == 17:
            version = int(np.max(np.rint(rows[:, 16])))
            self.state_replay.set_map_version(version)
            self.state_replay.add(rows)
            valid = self.state_replay.valid_rows()
            self.replay_buffer = np.zeros((self.replay_capacity, 17), dtype=np.float32)
            self.replay_buffer[: len(valid)] = valid
            self.replay_size = len(valid)
            self.replay_insert_idx = self.replay_size % self.replay_capacity
            return
        self._ensure_replay_buffer(int(rows.shape[1]))
        assert self.replay_buffer is not None

        merged = rows
        if self.replay_size > 0:
            merged = np.concatenate((self._valid_replay_rows(), rows), axis=0).astype(np.float32, copy=False)
        if len(merged) > self.replay_capacity:
            merged = self._cap_replay_rows(merged, self.replay_capacity)

        keep = len(merged)
        self.replay_buffer[:keep] = merged
        self.replay_size = keep
        self.replay_insert_idx = keep % self.replay_capacity

    def _sample_rows_random(self, rows: np.ndarray, count: int) -> np.ndarray:
        """Uniformly sample labelled states as specified by Algorithm 1."""
        data = np.asarray(rows, dtype=np.float32)
        count = max(0, int(count))
        if data.ndim != 2 or len(data) == 0 or count <= 0:
            width = int(data.shape[1]) if data.ndim == 2 else 26
            return np.zeros((0, width), dtype=np.float32)
        take = min(count, len(data))
        idx = np.random.choice(len(data), size=take, replace=False)
        return data[idx].astype(np.float32, copy=False)

    def _sample_replay_rows(self, count: int) -> np.ndarray:
        if self.replay_buffer is None or self.replay_size <= 0 or count <= 0:
            return np.zeros((0, 26), dtype=np.float32)
        return self._sample_rows_random(self.replay_buffer[: self.replay_size], int(count))

    def sample_training_batch(
        self,
        frame_data: np.ndarray,
        priority_rows: np.ndarray | None = None,
    ) -> np.ndarray:
        frame_rows = np.asarray(frame_data, dtype=np.float32)
        batch_size = self.effective_minibatch_size
        if frame_rows.ndim == 2 and frame_rows.shape[1] == 17:
            # Replay admission enforces source quotas; optimizer sampling also
            # enforces a useful free/unsafe label mixture so the unsafe head
            # cannot minimize its loss by rejecting nearly everything.
            return self.state_replay.sample_learning_balanced(batch_size)
        priority = (
            np.asarray(priority_rows, dtype=np.float32)
            if priority_rows is not None
            else np.zeros((0, frame_rows.shape[1] if frame_rows.ndim == 2 else 26), dtype=np.float32)
        )
        if priority.ndim != 2 or (frame_rows.ndim == 2 and priority.shape[1] != frame_rows.shape[1]):
            priority = np.zeros((0, frame_rows.shape[1] if frame_rows.ndim == 2 else 26), dtype=np.float32)
        priority_count = min(
            len(priority),
            max(0, int(round(batch_size * self.priority_ratio))),
        )
        priority_batch = self._sample_rows_random(priority, priority_count)
        ordinary_batch_size = max(0, batch_size - len(priority_batch))
        if ordinary_batch_size <= 0:
            return priority_batch
        if frame_rows.ndim != 2 or len(frame_rows) == 0:
            ordinary = self._sample_replay_rows(min(ordinary_batch_size, self.replay_size))
            return self._merge_sample_parts(priority_batch, ordinary)

        if self.replay_size <= 0:
            ordinary = self._sample_rows_random(frame_rows, ordinary_batch_size)
            return self._merge_sample_parts(priority_batch, ordinary)

        new_count = min(
            len(frame_rows),
            max(1, int(round(ordinary_batch_size * (1.0 - self.replay_ratio)))),
        )
        replay_count = max(0, ordinary_batch_size - new_count)

        if len(frame_rows) <= new_count:
            frame_batch = frame_rows
        else:
            frame_batch = self._sample_rows_random(frame_rows, new_count)

        replay_batch = self._sample_replay_rows(replay_count)
        ordinary = self._merge_sample_parts(frame_batch, replay_batch)
        return self._merge_sample_parts(priority_batch, ordinary)

    @staticmethod
    def _merge_sample_parts(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if len(first) == 0:
            return second
        if len(second) == 0:
            return first
        batch = np.concatenate((first, second), axis=0).astype(np.float32, copy=False)
        order = np.random.permutation(len(batch))
        return batch[order]

    def build_pair_batch(self, state_batch: np.ndarray) -> np.ndarray:
        rows = np.asarray(state_batch, dtype=np.float32)
        if rows.ndim != 2:
            return np.zeros((0, 26), dtype=np.float32)
        if rows.shape[1] == 26:
            return rows.astype(np.float32, copy=False)
        if len(rows) < 2:
            return np.zeros((0, 26), dtype=np.float32)
        if rows.shape[1] != 13:
            return np.zeros((0, 26), dtype=np.float32)
        order = np.random.permutation(len(rows))
        rows = rows[order]
        even_count = (len(rows) // 2) * 2
        if even_count < 2:
            return np.zeros((0, 26), dtype=np.float32)
        rows = rows[:even_count]
        q0 = rows[0::2, :6]
        y0 = rows[0::2, 6:7]
        n0 = rows[0::2, 7:]
        q1 = rows[1::2, :6]
        y1 = rows[1::2, 6:7]
        n1 = rows[1::2, 7:]
        return np.concatenate((q0, q1, y0, y1, n0, n1), axis=1).astype(np.float32, copy=False)

    @staticmethod
    def reshuffle_pair_endpoints(pair_rows: np.ndarray) -> np.ndarray:
        """Re-form start/goal pairs from independently labelled states.

        Algorithm 1 stores individual C-space samples in memory and shuffles
        them into new start/goal pairs for every optimizer iteration.  Online
        sampling naturally emits 26-column local pairs, so leaving those pairs
        intact over-represents short displacements and does not teach the
        global time field.  This conversion preserves every configuration,
        speed label, and normal while randomizing only the pairing.
        """
        rows = np.asarray(pair_rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) < 2:
            return rows.astype(np.float32, copy=False)
        states = np.concatenate(
            (
                np.concatenate((rows[:, :6], rows[:, 12:13], rows[:, 14:20]), axis=1),
                np.concatenate((rows[:, 6:12], rows[:, 13:14], rows[:, 20:26]), axis=1),
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        states = states[np.random.permutation(len(states))]
        s0 = states[0::2]
        s1 = states[1::2]
        return np.concatenate(
            (s0[:, :6], s1[:, :6], s0[:, 6:7], s1[:, 6:7], s0[:, 7:], s1[:, 7:]),
            axis=1,
        ).astype(np.float32, copy=False)

    def recombine_replay_pairs(self, count: int) -> np.ndarray:
        if self.replay_buffer is None or self.replay_size <= 0 or count <= 0:
            return np.zeros((0, 26), dtype=np.float32)
        rows = self._valid_replay_rows()
        if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
            return np.zeros((0, 26), dtype=np.float32)

        states0 = np.concatenate((rows[:, :6], rows[:, 12:13], rows[:, 14:20]), axis=1)
        states1 = np.concatenate((rows[:, 6:12], rows[:, 13:14], rows[:, 20:26]), axis=1)
        states = np.concatenate((states0, states1), axis=0).astype(np.float32, copy=False)
        finite = np.all(np.isfinite(states), axis=1)
        states = states[finite]
        if len(states) < 2:
            return np.zeros((0, 26), dtype=np.float32)

        count = min(int(count), max(1, len(states)))
        idx0 = np.random.randint(0, len(states), size=count)
        idx1 = np.random.randint(0, len(states), size=count)
        same = idx0 == idx1
        if np.any(same) and len(states) > 1:
            idx1[same] = (idx1[same] + 1) % len(states)

        s0 = states[idx0]
        s1 = states[idx1]
        q_delta = np.max(np.abs(s0[:, :6] - s1[:, :6]), axis=1)
        keep = q_delta > 1.0e-3
        if not np.any(keep):
            return np.zeros((0, 26), dtype=np.float32)
        s0 = s0[keep]
        s1 = s1[keep]
        return np.concatenate((s0[:, :6], s1[:, :6], s0[:, 6:7], s1[:, 6:7], s0[:, 7:], s1[:, 7:]), axis=1).astype(np.float32, copy=False)

    def train_step(
        self,
        frame_data: np.ndarray,
        epochs: int,
        transient_rows: np.ndarray | None = None,
        priority_rows: np.ndarray | None = None,
    ) -> float | None:
        persistent_rows = np.asarray(frame_data, dtype=np.float32)
        transient = (
            np.asarray(transient_rows, dtype=np.float32)
            if transient_rows is not None
            else np.zeros((0, persistent_rows.shape[1] if persistent_rows.ndim == 2 else 26), dtype=np.float32)
        )
        if persistent_rows.ndim != 2:
            persistent_rows = np.zeros((0, 26), dtype=np.float32)
        if transient.ndim != 2 or (len(persistent_rows) and transient.shape[1] != persistent_rows.shape[1]):
            transient = np.zeros((0, persistent_rows.shape[1]), dtype=np.float32)
        if len(persistent_rows) == 0 and len(transient) == 0:
            return None
        # Synthetic random pairs improve conditioning for this optimizer pass,
        # but they are not newly collision-labelled states. Keep them out of
        # the long-lived replay buffer so they cannot recursively dominate it.
        if len(persistent_rows):
            self.add_rows(persistent_rows)
        if (
            priority_rows is not None
            and np.asarray(priority_rows).ndim == 2
            and np.asarray(priority_rows).shape[1] == 17
            and len(priority_rows)
        ):
            self.add_rows(np.asarray(priority_rows, dtype=np.float32))
        training_rows = (
            np.concatenate((persistent_rows, transient), axis=0).astype(np.float32, copy=False)
            if len(transient)
            else persistent_rows
        )
        train_steps = max(1, int(epochs))
        last_loss = None
        state_rows = self.state_replay.valid_rows()
        state_replay_t: torch.Tensor | None = None
        state_indices_t: torch.Tensor | None = None
        if (
            persistent_rows.shape[1] == 17
            and len(state_rows) >= 2
        ):
            # Upload the current replay once per optimizer cycle. Previously
            # every one of ~180 updates copied a NumPy minibatch to CUDA,
            # forcing the GPU to wait on repeated host/device transfers.
            state_replay_t = torch.from_numpy(
                np.ascontiguousarray(state_rows)
            ).to(self.device, dtype=torch.float32, non_blocking=True)
            index_batches = [
                self.state_replay.sample_learning_balanced_indices(
                    self.effective_minibatch_size, rows=state_rows
                )
                for _ in range(train_steps)
            ]
            even = min(((len(index) // 2) * 2 for index in index_batches), default=0)
            if even >= 2:
                indices_np = np.stack(
                    [index[:even] for index in index_batches], axis=0
                ).astype(np.int64, copy=False)
                state_indices_t = torch.from_numpy(indices_np).to(
                    self.device, non_blocking=True
                )
        for train_index in range(train_steps):
            if state_replay_t is not None and state_indices_t is not None:
                index_t = state_indices_t[train_index]
                even = int(len(index_t))
                state_batch_t = state_replay_t.index_select(0, index_t)
                order_t = torch.randperm(even, device=self.device)
                state_batch_t = state_batch_t.index_select(0, order_t)
                pair_points_t = torch.cat(
                    (state_batch_t[0::2, :6], state_batch_t[1::2, :6]), dim=1
                )
                self.last_train_batch_size = int(even)
                self.last_train_pair_count = int(len(pair_points_t))
                last_loss = self.model.train_state_batch(state_batch_t, pair_points_t)
                continue
            state_batch = self.sample_training_batch(training_rows, priority_rows=priority_rows)
            if state_batch.size == 0:
                continue
            self.last_train_batch_size = int(len(state_batch))
            if state_batch.ndim == 2 and state_batch.shape[1] == 17:
                even = (len(state_batch) // 2) * 2
                state_batch = state_batch[:even]
                if even < 2:
                    continue
                order = np.random.permutation(even)
                state_batch = state_batch[order]
                pair_points = np.concatenate(
                    (state_batch[0::2, :6], state_batch[1::2, :6]), axis=1
                ).astype(np.float32)
                last_loss = self.model.train_state_batch(
                    torch.from_numpy(state_batch).float().to(self.device),
                    torch.from_numpy(pair_points).float().to(self.device),
                )
                self.last_train_pair_count = int(len(pair_points))
                if last_loss is not None:
                    continue
            pair_batch = self.build_pair_batch(state_batch)
            pair_batch = self.reshuffle_pair_endpoints(pair_batch)
            self.last_train_pair_count = int(len(pair_batch))
            if pair_batch.size == 0:
                continue
            batch_t = torch.from_numpy(pair_batch).float().to(self.device)
            last_loss = self.model.train_batch(
                batch_t,
                accumulation_steps=self.gradient_accumulation_steps,
            )
            if last_loss is None:
                continue
        self.total_epochs_trained += train_steps
        if last_loss is not None:
            self.loss_history.append(float(last_loss))
        return None if last_loss is None else float(last_loss)

    @staticmethod
    def _training_speed_target(raw_speed: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(raw_speed, dtype=np.float32), 0.0, 1.0)

    def predict_normalized_pair_speeds(
        self,
        q0n_batch: np.ndarray,
        q1n_batch: np.ndarray,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        q0 = np.asarray(q0n_batch, dtype=np.float32)
        q1 = np.asarray(q1n_batch, dtype=np.float32)
        if q0.ndim == 1:
            q0 = q0[None, :]
        if q1.ndim == 1:
            q1 = q1[None, :]
        count = min(len(q0), len(q1))
        if count <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        q0 = np.clip(q0[:count], -0.5, 0.5)
        q1 = np.clip(q1[:count], -0.5, 0.5)
        pred0_parts: list[np.ndarray] = []
        pred1_parts: list[np.ndarray] = []
        was_training = bool(self.model.network.training)
        self.model.network.train(False)
        for start in range(0, count, max(1, int(batch_size))):
            end = min(count, start + max(1, int(batch_size)))
            xp_np = np.concatenate((q0[start:end], q1[start:end]), axis=1).astype(np.float32)
            xp = torch.from_numpy(xp_np).float().to(self.device)
            xp.requires_grad_(True)
            tau, _w, xp_grad = self.model.network.out(xp)
            arrival_time = self.model.function.arrival_time(tau, xp_grad)
            dtime = self.model.function.gradient(arrival_time, xp_grad, create_graph=False)
            dt0 = dtime[:, : self.dim]
            dt1 = dtime[:, self.dim :]
            pred0 = torch.rsqrt(torch.sum(dt0 * dt0, dim=1) + 1.0e-8)
            pred1 = torch.rsqrt(torch.sum(dt1 * dt1, dim=1) + 1.0e-8)
            pred0_parts.append(pred0.detach().cpu().numpy().astype(np.float32))
            pred1_parts.append(pred1.detach().cpu().numpy().astype(np.float32))
        self.model.network.train(was_training)
        return np.concatenate(pred0_parts), np.concatenate(pred1_parts)

    def predict_normalized_state_geometry(
        self, q_batch: np.ndarray, batch_size: int = 4096
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict goal-independent speed, unsafe probability and calibrated speed."""
        q = np.asarray(q_batch, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self.dim or len(q) == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, z
        speeds: list[np.ndarray] = []
        unsafe: list[np.ndarray] = []
        was_training = bool(self.model.network.training)
        self.model.network.train(False)
        with torch.inference_mode():
            for start in range(0, len(q), max(1, int(batch_size))):
                qt = torch.from_numpy(
                    np.ascontiguousarray(np.clip(q[start:start + batch_size], -0.5, 0.5))
                ).to(self.device, non_blocking=True)
                speed_t, unsafe_logit = self.model.network.state_geometry(qt)
                speeds.append(speed_t.cpu().numpy().astype(np.float32))
                unsafe.append(torch.sigmoid(unsafe_logit).cpu().numpy().astype(np.float32))
        self.model.network.train(was_training)
        speed = np.concatenate(speeds)
        unsafe_probability = np.concatenate(unsafe)
        # Calibration is intentionally subtractive and never increases a
        # network prediction.  Per-bin margins are selected on held-out data.
        margins = np.full(len(speed), float(self.calibration_margins.get("global", 0.0)), dtype=np.float32)
        if self.coverage_tree is not None and any(key.startswith("clearance_") for key in self.calibration_margins):
            distance, _ = self.coverage_tree.query(np.clip(q, -0.5, 0.5), k=1)
            predicted_clearance = 0.10 * speed
            clearance_bin = np.select(
                [predicted_clearance < 0.01, predicted_clearance < 0.03, predicted_clearance < 0.07],
                [0, 1, 2], default=3,
            )
            coverage_bin = np.select([distance < 0.02, distance < 0.05], [0, 1], default=2)
            for ci in range(4):
                for di in range(3):
                    mask = (clearance_bin == ci) & (coverage_bin == di)
                    margins[mask] = float(self.calibration_margins.get(
                        f"clearance_{ci}_coverage_{di}", margins[mask][0] if np.any(mask) else 0.0
                    ))
        conservative = np.clip(speed - margins, 0.0, 1.0).astype(np.float32)
        return speed, unsafe_probability, conservative

    def calibrate_state_geometry(
        self,
        q: np.ndarray,
        target_speed: np.ndarray,
        clearance_m: np.ndarray,
        coverage_distance: np.ndarray | None = None,
    ) -> dict[str, float]:
        from ur_mntfields_arm.online_training import calibrate_conservative_speed

        pred, _unsafe, _conservative = self.predict_normalized_state_geometry(q)
        margins = calibrate_conservative_speed(
            pred, target_speed, clearance_m, coverage_distance=coverage_distance
        )
        self.calibration_margins = margins
        return margins

    def set_coverage_support(
        self, states: np.ndarray, shell_radius: float, free_radius: float
    ) -> None:
        from scipy.spatial import cKDTree

        values = np.asarray(states, dtype=np.float32).reshape(-1, self.dim)
        if len(values):
            values = np.unique(np.round(values, decimals=6), axis=0).astype(np.float32)
        self.coverage_states = values
        self.coverage_tree = cKDTree(values) if len(values) else None
        if len(values):
            # One representative per coarse C-space cell provides globally
            # distributed detour candidates without scanning every replay
            # state for every route. Lexicographic unique ordering makes this
            # deterministic across save/load.
            cell = np.floor((values + 0.5) / 0.08).astype(np.int16)
            _cells, first = np.unique(cell, axis=0, return_index=True)
            landmarks = values[np.sort(first)]
            if len(landmarks) > 512:
                choice = np.linspace(0, len(landmarks) - 1, 512, dtype=np.int64)
                landmarks = landmarks[choice]
            self.coverage_landmarks = landmarks.astype(np.float32, copy=False)
        else:
            self.coverage_landmarks = np.zeros((0, self.dim), dtype=np.float32)
        self.shell_coverage_radius = float(np.clip(shell_radius, 0.0, 0.08))
        self.free_coverage_radius = float(np.clip(free_radius, 0.0, 0.08))

    def predict_travel_times(
        self,
        q0n_batch: np.ndarray,
        q1n_batch: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        q0 = np.asarray(q0n_batch, dtype=np.float32)
        q1 = np.asarray(q1n_batch, dtype=np.float32)
        if q0.ndim == 1:
            q0 = q0[None, :]
        if q1.ndim == 1:
            q1 = q1[None, :]
        count = min(len(q0), len(q1))
        if count <= 0:
            return np.zeros((0,), dtype=np.float32)
        q0 = np.clip(q0[:count], -0.5, 0.5)
        q1 = np.clip(q1[:count], -0.5, 0.5)
        parts: list[np.ndarray] = []
        was_training = bool(self.model.network.training)
        self.model.network.train(False)
        for start in range(0, count, max(1, int(batch_size))):
            end = min(count, start + max(1, int(batch_size)))
            xp_np = np.concatenate((q0[start:end], q1[start:end]), axis=1).astype(np.float32)
            xp = torch.from_numpy(xp_np).float().to(self.device)
            tau = self.model.function.TravelTimes(xp)
            parts.append(tau.detach().cpu().numpy().astype(np.float32))
        self.model.network.train(was_training)
        return np.concatenate(parts)

    def _predict_replay_gradients(self, rows: np.ndarray, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.asarray(rows, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 26 or len(data) == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, np.zeros((0, 6), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
        pred0_parts: list[np.ndarray] = []
        pred1_parts: list[np.ndarray] = []
        dt0_parts: list[np.ndarray] = []
        dt1_parts: list[np.ndarray] = []
        was_training = bool(self.model.network.training)
        self.model.network.train(False)
        for start in range(0, len(data), max(1, int(batch_size))):
            batch = data[start:start + max(1, int(batch_size)), : 2 * self.dim]
            xp = torch.from_numpy(batch).float().to(self.device)
            xp.requires_grad_(True)
            tau, _w, xp_grad = self.model.network.out(xp)
            arrival_time = self.model.function.arrival_time(tau, xp_grad)
            dtime = self.model.function.gradient(arrival_time, xp_grad, create_graph=False)
            dt0 = dtime[:, : self.dim]
            dt1 = dtime[:, self.dim :]
            pred0 = torch.rsqrt(torch.sum(dt0 * dt0, dim=1) + 1.0e-8)
            pred1 = torch.rsqrt(torch.sum(dt1 * dt1, dim=1) + 1.0e-8)
            pred0_parts.append(pred0.detach().cpu().numpy().astype(np.float32))
            pred1_parts.append(pred1.detach().cpu().numpy().astype(np.float32))
            dt0_parts.append(dt0.detach().cpu().numpy().astype(np.float32))
            dt1_parts.append(dt1.detach().cpu().numpy().astype(np.float32))
        self.model.network.train(was_training)
        return (
            np.concatenate(pred0_parts),
            np.concatenate(pred1_parts),
            np.concatenate(dt0_parts, axis=0),
            np.concatenate(dt1_parts, axis=0),
        )

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(finite) < 3:
            return 0.0
        a = a[finite]
        b = b[finite]
        if float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def evaluate_replay_diagnostics(
        self,
        max_rows: int = 4096,
        *,
        rows: np.ndarray | None = None,
        prediction: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        if rows is None and self.replay_buffer is not None and self.replay_buffer.shape[1] == 17:
            state_rows = self._valid_replay_rows()
            if len(state_rows) > max_rows:
                state_rows = state_rows[np.random.choice(len(state_rows), int(max_rows), replace=False)]
            if len(state_rows) == 0:
                return {"diag_rows": 0.0}
            target = np.clip(state_rows[:, 7], 0.0, 1.0)
            pred, unsafe, conservative = self.predict_normalized_state_geometry(state_rows[:, :6])
            low = target <= 0.20
            free = target >= 0.50
            diag = {
                "diag_rows": float(len(state_rows)),
                "speed_mae": float(np.mean(np.abs(pred - target))),
                "speed_rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
                "speed_corr": self._safe_corr(pred, target),
                "pred_speed_mean": float(np.mean(pred)),
                "target_speed_mean": float(np.mean(target)),
                "low_target_threshold": 0.20,
                "low_target_overpred_frac": float(
                    np.mean((conservative[low] >= 0.20) & (unsafe[low] < 0.10))
                ) if np.any(low) else 0.0,
                "low_target_count": float(np.count_nonzero(low)),
                "unsafe_probability_mean": float(np.mean(unsafe)),
                "unsafe_probability_low_mean": float(np.mean(unsafe[low])) if np.any(low) else 1.0,
                "unsafe_probability_free_mean": float(np.mean(unsafe[free])) if np.any(free) else 1.0,
                "free_target_count": float(np.count_nonzero(free)),
                "free_speed_pass_rate": float(np.mean(conservative[free] >= 0.20)) if np.any(free) else 0.0,
                "free_unsafe_pass_rate": float(np.mean(unsafe[free] < 0.10)) if np.any(free) else 0.0,
                "free_state_recall": float(
                    np.mean((conservative[free] >= 0.20) & (unsafe[free] < 0.10))
                ) if np.any(free) else 0.0,
                "false_blocked_free_rate": float(
                    np.mean((conservative[free] < 0.20) | (unsafe[free] >= 0.10))
                ) if np.any(free) else 1.0,
                "normal_cos_mean": 0.0,
                "normal_cos_near_mean": 0.0,
            }
            self.last_diagnostics = diag
            return diag
        if rows is None:
            # Training independently reshuffles labelled endpoints on every
            # update. Saved sampler rows are intentionally local and critical
            # rows may even repeat q0 as q1. Evaluating those original pairs
            # confounds clearance with pair distance and made a good field
            # appear inversely correlated. Match the optimizer distribution.
            rows = self.reshuffle_pair_endpoints(self._valid_replay_rows())
        else:
            rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
            return {"diag_rows": 0.0}
        # A supplied prediction is aligned with these exact rows. Do not
        # resample it underneath the caller and silently mix array lengths.
        if len(rows) > max_rows and prediction is None:
            idx = np.random.choice(len(rows), int(max_rows), replace=False)
            rows = rows[idx]
        raw_speed = np.clip(rows[:, 12:14], 0.0, 1.0)
        target = self._training_speed_target(raw_speed)
        if prediction is None:
            pred0, pred1, dt0, dt1 = self._predict_replay_gradients(rows)
        else:
            pred0, pred1, dt0, dt1 = prediction
        pred = np.stack((pred0, pred1), axis=1)
        pred_clip = np.clip(pred, 0.0, 2.0)
        target_flat = target.reshape(-1)
        pred_flat = pred_clip.reshape(-1)
        normals0 = rows[:, 14:20]
        normals1 = rows[:, 20:26]
        dt0_norm = np.linalg.norm(dt0, axis=1, keepdims=True)
        dt1_norm = np.linalg.norm(dt1, axis=1, keepdims=True)
        grad0 = -dt0 / np.maximum(dt0_norm * dt0_norm, 1.0e-8)
        goal_delta = rows[:, 6:12] - rows[:, :6]
        grad0_norm = np.linalg.norm(grad0, axis=1)
        goal_norm = np.linalg.norm(goal_delta, axis=1)
        goal_mask = (grad0_norm > 1.0e-8) & (goal_norm > 1.0e-8)
        if np.any(goal_mask):
            grad_goal_cos = np.sum(grad0[goal_mask] * goal_delta[goal_mask], axis=1) / (
                grad0_norm[goal_mask] * goal_norm[goal_mask]
            )
        else:
            grad_goal_cos = np.zeros((0,), dtype=np.float32)
        cos0 = np.sum((-dt0 / np.maximum(dt0_norm, 1.0e-8)) * normals0, axis=1)
        cos1 = np.sum((-dt1 / np.maximum(dt1_norm, 1.0e-8)) * normals1, axis=1)
        raw_flat = raw_speed.reshape(-1)
        low_target_threshold = 0.20
        low_pred_max = float(self.loss_config.get("low_speed_pred_max", 0.35))
        near = raw_flat <= np.quantile(raw_flat, 0.25) if len(raw_flat) >= 4 else raw_flat <= 0.25
        far = raw_flat >= np.quantile(raw_flat, 0.75) if len(raw_flat) >= 4 else raw_flat >= 0.75
        near_pred = pred_flat[near] if np.any(near) else np.zeros((0,), dtype=np.float32)
        far_pred = pred_flat[far] if np.any(far) else np.zeros((0,), dtype=np.float32)
        near_cos = np.concatenate(
            (
                cos0[raw_speed[:, 0] <= low_target_threshold],
                cos1[raw_speed[:, 1] <= low_target_threshold],
            )
        )
        diag = {
            "diag_rows": float(len(rows)),
            "speed_mae": float(np.mean(np.abs(pred_flat - target_flat))),
            "speed_rmse": float(np.sqrt(np.mean((pred_flat - target_flat) ** 2))),
            "speed_corr": self._safe_corr(pred_flat, target_flat),
            "pred_speed_mean": float(np.mean(pred_flat)),
            "target_speed_mean": float(np.mean(target_flat)),
            "pred_near_mean": float(np.mean(near_pred)) if len(near_pred) else 0.0,
            "pred_far_mean": float(np.mean(far_pred)) if len(far_pred) else 0.0,
            "near_far_gap": float(np.mean(far_pred) - np.mean(near_pred)) if len(near_pred) and len(far_pred) else 0.0,
            "normal_cos_mean": float(np.mean(np.concatenate((cos0, cos1)))),
            "normal_cos_near_mean": float(np.mean(near_cos)) if len(near_cos) else 0.0,
            "grad_goal_cos_mean": float(np.mean(grad_goal_cos)) if len(grad_goal_cos) else 0.0,
            "grad_goal_cos_median": float(np.median(grad_goal_cos)) if len(grad_goal_cos) else 0.0,
            "grad_goal_neg_frac": float(np.mean(grad_goal_cos < 0.0)) if len(grad_goal_cos) else 0.0,
            "pred_nonfinite_frac": float(np.mean(~np.isfinite(pred_flat))),
            "low_target_threshold": low_target_threshold,
            "low_target_overpred_frac": float(
                np.mean(pred_flat[raw_flat <= low_target_threshold] > low_pred_max)
            )
            if np.any(raw_flat <= low_target_threshold)
            else 0.0,
            "low_target_count": float(np.count_nonzero(raw_flat <= low_target_threshold)),
        }
        self.last_diagnostics = diag
        return diag

    def save_replay_diagnostic_plots(
        self,
        output_dir: str | Path,
        kinematics,
        *,
        step_label: str,
        q_start: np.ndarray | None = None,
        max_rows: int = 4096,
        grid_size: int = 61,
        include_joint_slices: bool = True,
    ) -> list[Path]:
        if self.replay_buffer is not None and self.replay_buffer.shape[1] == 17:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            diag = self.evaluate_replay_diagnostics(max_rows=max_rows)
            summary_path = out_dir / f"diagnostic_summary_{step_label}.json"
            summary_path.write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf-8")
            return [summary_path]
        rows = self.reshuffle_pair_endpoints(self._valid_replay_rows())
        if rows.ndim != 2 or rows.shape[1] != 26 or len(rows) == 0:
            return []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(rows) > max_rows:
            idx = np.random.choice(len(rows), int(max_rows), replace=False)
            rows = rows[idx]
        raw_speed = np.clip(rows[:, 12:14], 0.0, 1.0)
        target = self._training_speed_target(raw_speed)
        pred0, pred1, dt0, dt1 = self._predict_replay_gradients(rows)
        pred = np.clip(np.stack((pred0, pred1), axis=1), 0.0, 2.0)
        saved: list[Path] = []
        # Reuse the gradient pass already required for the plots. Running a
        # second full autograd diagnostic here doubled checkpoint overhead.
        diag = self.evaluate_replay_diagnostics(
            max_rows=max_rows,
            rows=rows,
            prediction=(pred0, pred1, dt0, dt1),
        )
        summary_path = out_dir / f"diagnostic_summary_{step_label}.json"
        summary_path.write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf-8")
        saved.append(summary_path)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        ax.scatter(target.reshape(-1), pred.reshape(-1), s=5, alpha=0.35, c=raw_speed.reshape(-1), cmap="turbo")
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0)
        ax.set_xlabel("target speed used by loss")
        ax.set_ylabel("predicted speed = 1 / ||grad T||")
        ax.set_title(
            f"Replay Speed Fit {step_label}\n"
            f"MAE={diag.get('speed_mae', 0.0):.3f}, corr={diag.get('speed_corr', 0.0):.3f}, "
            f"low-overpred={diag.get('low_target_overpred_frac', 0.0):.2f}"
        )
        ax.grid(alpha=0.25)
        path = out_dir / f"replay_speed_fit_{step_label}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

        states = np.concatenate((rows[:, :6], rows[:, 6:12]), axis=0).astype(np.float64)
        speeds = raw_speed.reshape(-1)
        pred_speeds = pred.reshape(-1)
        states_center = states - np.mean(states, axis=0, keepdims=True)
        try:
            _u, _s, vh = np.linalg.svd(states_center, full_matrices=False)
            coords = states_center @ vh[:2].T
        except np.linalg.LinAlgError:
            coords = states_center[:, :2]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True, sharey=True)
        sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=speeds, s=5, alpha=0.45, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[0].set_title("Training labels: low speed near obstacles")
        sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=np.clip(pred_speeds, 0.0, 1.0), s=5, alpha=0.45, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[1].set_title("Field predicted speed")
        for ax_i in axes:
            ax_i.set_xlabel("replay PCA 1")
            ax_i.grid(alpha=0.2)
        axes[0].set_ylabel("replay PCA 2")
        fig.colorbar(sc0, ax=axes[0], label="raw speed label")
        fig.colorbar(sc1, ax=axes[1], label="predicted speed")
        path = out_dir / f"replay_pca_speed_{step_label}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

        if include_joint_slices:
            if q_start is not None and np.all(np.isfinite(q_start)):
                q0n = kinematics.normalize(np.asarray(q_start, dtype=np.float64)).astype(np.float32)
            else:
                q0n = rows[0, :6].astype(np.float32)
            q1n = rows[int(np.argmax(np.max(np.abs(rows[:, 6:12] - q0n[None, :]), axis=1))), 6:12].astype(np.float32)
            for ji, jj in ((0, 1), (1, 2), (2, 3), (3, 4)):
                path = self._save_joint_slice_plot(out_dir, step_label, q0n, q1n, ji, jj, int(grid_size))
                if path is not None:
                    saved.append(path)
        return saved

    def _save_joint_slice_plot(
        self,
        out_dir: Path,
        step_label: str,
        q0n: np.ndarray,
        q1n: np.ndarray,
        ji: int,
        jj: int,
        grid_size: int,
    ) -> Path | None:
        center = np.asarray(q0n, dtype=np.float32).copy()
        goal = np.asarray(q1n, dtype=np.float32).copy()
        min_span = 0.18
        pad = 0.05
        x_lo = max(-0.5, float(min(center[ji], goal[ji]) - pad))
        x_hi = min(0.5, float(max(center[ji], goal[ji]) + pad))
        y_lo = max(-0.5, float(min(center[jj], goal[jj]) - pad))
        y_hi = min(0.5, float(max(center[jj], goal[jj]) + pad))
        if x_hi - x_lo < min_span:
            mid = 0.5 * (x_lo + x_hi)
            x_lo = max(-0.5, mid - 0.5 * min_span)
            x_hi = min(0.5, mid + 0.5 * min_span)
        if y_hi - y_lo < min_span:
            mid = 0.5 * (y_lo + y_hi)
            y_lo = max(-0.5, mid - 0.5 * min_span)
            y_hi = min(0.5, mid + 0.5 * min_span)
        xs = np.linspace(x_lo, x_hi, grid_size, dtype=np.float32)
        ys = np.linspace(y_lo, y_hi, grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        samples = np.repeat(center[None, :], grid_size * grid_size, axis=0)
        samples[:, ji] = xx.reshape(-1)
        samples[:, jj] = yy.reshape(-1)
        targets = np.repeat(goal[None, :], len(samples), axis=0)
        xp = np.concatenate((samples, targets), axis=1).astype(np.float32)
        was_training = bool(self.model.network.training)
        self.model.network.train(False)
        xpt = torch.from_numpy(xp).float().to(self.device)
        xpt.requires_grad_(True)
        tau, _w, xpg = self.model.network.out(xpt)
        arrival_time = self.model.function.arrival_time(tau, xpg)
        dtime = self.model.function.gradient(arrival_time, xpg, create_graph=False)
        dt0 = dtime[:, : self.dim]
        pred_speed = torch.rsqrt(torch.sum(dt0 * dt0, dim=1) + 1.0e-8).detach().cpu().numpy().reshape(grid_size, grid_size)
        time_np = arrival_time.detach().cpu().numpy().reshape(grid_size, grid_size)
        self.model.network.train(was_training)
        rollout = self.gradient_rollout(center, goal, step_size=0.03, max_steps=180, tol=0.01)
        if rollout.ndim != 2 or rollout.shape[1] != self.dim:
            rollout = np.zeros((0, self.dim), dtype=np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
        c0 = axes[0].contourf(xs, ys, time_np, levels=28, cmap="viridis")
        c1 = axes[1].contourf(xs, ys, np.clip(pred_speed, 0.0, 1.0), levels=28, cmap="turbo", vmin=0.0, vmax=1.0)
        for ax in axes:
            ax.scatter([center[ji]], [center[jj]], c="lime", s=70, label="start")
            ax.scatter([goal[ji]], [goal[jj]], c="red", s=70, label="goal")
            if len(rollout):
                ax.plot(
                    rollout[:, ji],
                    rollout[:, jj],
                    color="white",
                    linewidth=2.0,
                    alpha=0.95,
                    label="gradient rollout",
                )
                ax.scatter(
                    rollout[-1:, ji],
                    rollout[-1:, jj],
                    c="black",
                    s=35,
                    label="rollout end",
                )
            ax.set_xlim(float(xs[0]), float(xs[-1]))
            ax.set_ylim(float(ys[0]), float(ys[-1]))
            ax.set_xlabel(f"q{ji} normalized")
            ax.set_ylabel(f"q{jj} normalized")
            ax.grid(alpha=0.2)
        axes[0].set_title("factorized T contour, 2D cut through 6D")
        axes[1].set_title("predicted speed contour, same 2D cut")
        axes[1].legend(loc="upper right")
        fig.colorbar(c0, ax=axes[0], label="T")
        fig.colorbar(c1, ax=axes[1], label="pred speed")
        path = out_dir / f"field_slice_{step_label}_{ji}_{jj}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    def save_checkpoint(self, checkpoint_path: str | Path, metadata: dict[str, object] | None = None):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = dict(self.checkpoint_metadata)
        if metadata:
            checkpoint_metadata.update(metadata)
        checkpoint_metadata.setdefault("certification_passed", bool(self.certification_passed))
        persist_coverage = checkpoint_path.name in ("weights_final.pt", "weights_partial.pt")
        coverage_states = self.coverage_states if persist_coverage else np.zeros((0, self.dim), dtype=np.float32)
        payload = {
                "network_state_dict": self.model.network.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
                "B": self.model.B.detach().cpu(),
                "architecture_version": str(
                    getattr(self.model.network, "ARCHITECTURE_VERSION", "unknown")
                ),
                "device": self.device,
                "dim": self.dim,
                "learning_rate": self.learning_rate,
                "loss_config": dict(self.loss_config),
                "loss_history": list(self.loss_history),
                "total_epochs_trained": self.total_epochs_trained,
                "calibration_margins": dict(self.calibration_margins),
                "coverage_states": torch.from_numpy(coverage_states.astype(np.float32)),
                "shell_coverage_radius": float(self.shell_coverage_radius),
                "free_coverage_radius": float(self.free_coverage_radius),
                **checkpoint_metadata,
            }
        if checkpoint_path.name == "weights_final.pt" and not bool(payload.get("certification_passed")):
            raise RuntimeError("weights_final.pt may only be saved after balanced certification passes")
        torch.save(payload, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        certified_execution: bool = False,
        current_scene_signature: str | None = None,
        current_map_version: int | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=self.device)
        expected_architecture = str(
            getattr(self.model.network, "ARCHITECTURE_VERSION", "unknown")
        )
        checkpoint_architecture = payload.get("architecture_version")
        legacy_architectures = {"pinn_factorized_t_v2"}
        self.legacy_checkpoint = checkpoint_architecture in legacy_architectures
        if checkpoint_architecture != expected_architecture and not self.legacy_checkpoint:
            raise RuntimeError(f"Unsupported checkpoint architecture_version={checkpoint_architecture!r}")
        if certified_execution and (
            self.legacy_checkpoint or not bool(payload.get("certification_passed", False))
        ):
            raise RuntimeError("Certified learned-only execution requires a v3 state-geometry checkpoint with certification_passed=true")
        B = payload.get("B")
        if B is not None:
            self.model.B = B.to(self.device).float()
            if self.model.network is not None:
                # B is intentionally not a registered parameter/buffer in the
                # original network, so state_dict loading does not restore the
                # Fourier input map. Keep checkpoint reloads numerically
                # identical to the trained model.
                self.model.network.B = self.model.B.T.to(self.device)
                self.model.network._fourier_w = 2.0 * np.pi * self.model.network.B
        network_state = payload.get("network_state_dict")
        if network_state is None:
            raise KeyError(f"Checkpoint missing network_state_dict: {checkpoint_path}")
        self.model.network.load_state_dict(network_state, strict=not self.legacy_checkpoint)
        self.calibration_margins = {
            str(k): float(v) for k, v in dict(payload.get("calibration_margins", {})).items()
        }
        self.certification_passed = bool(payload.get("certification_passed", False))
        self.checkpoint_metadata = {
            key: payload.get(key)
            for key in (
                "scene_signature", "voxel_map_path", "map_version",
                "certification_metrics", "training_wall_time", "sample_source_counts",
            )
            if key in payload
        }
        if certified_execution:
            self._validate_scene_artifact(
                checkpoint_path,
                current_scene_signature=current_scene_signature,
                current_map_version=current_map_version,
            )
        coverage_payload = payload.get("coverage_states", torch.zeros((0, 6)))
        if isinstance(coverage_payload, torch.Tensor):
            coverage_payload = coverage_payload.detach().cpu().numpy()
        self.set_coverage_support(
            np.asarray(coverage_payload, dtype=np.float32),
            float(payload.get("shell_coverage_radius", 0.0)),
            float(payload.get("free_coverage_radius", 0.0)),
        )
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None and self.model.optimizer is not None and not self.legacy_checkpoint:
            self.model.optimizer.load_state_dict(optimizer_state)
        self.loss_history = [float(v) for v in payload.get("loss_history", [])]
        self.total_epochs_trained = int(payload.get("total_epochs_trained", 0))

    def _validate_scene_artifact(
        self,
        checkpoint_path: Path,
        *,
        current_scene_signature: str | None,
        current_map_version: int | None,
    ) -> None:
        from ur_mntfields_arm.voxel_map import SparseVoxelMap

        stored_path = self.checkpoint_metadata.get("voxel_map_path")
        stored_signature = self.checkpoint_metadata.get("scene_signature")
        stored_version = self.checkpoint_metadata.get("map_version")
        if not stored_path or not stored_signature or stored_version is None:
            raise RuntimeError("Certified checkpoint is missing its voxel map, scene signature, or map version")
        artifact = Path(str(stored_path))
        if not artifact.is_absolute():
            artifact = checkpoint_path.parent / artifact
        if not artifact.exists():
            raise RuntimeError(f"Certified voxel map artifact does not exist: {artifact}")
        voxel_map = SparseVoxelMap.load(artifact)
        if voxel_map.scene_signature() != str(stored_signature) or voxel_map.map_version != int(stored_version):
            raise RuntimeError("Certified checkpoint does not match its voxel map artifact")
        if current_scene_signature is not None and str(current_scene_signature) != str(stored_signature):
            raise RuntimeError("Current scene signature does not match the certified checkpoint")
        if current_map_version is not None and int(current_map_version) != int(stored_version):
            raise RuntimeError("Current scene map version does not match the certified checkpoint")

    def save_loss_plot(self, plot_path: str | Path):
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        if self.loss_history:
            xs = np.arange(1, len(self.loss_history) + 1, dtype=np.int32)
            plt.plot(xs, np.asarray(self.loss_history, dtype=np.float32), color="#d55e00", linewidth=2.0)
        plt.xlabel("Train Step")
        plt.ylabel("Loss")
        plt.title("Online Arm MNTFields Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def gradient_rollout(self, q_start_norm: np.ndarray, q_goal_norm: np.ndarray, step_size: float = 0.04, max_steps: int = 160, tol: float = 0.01) -> np.ndarray:
        """Integrate the learned field without a GPU/CPU round trip per step."""

        q_src = torch.as_tensor(q_start_norm, dtype=torch.float32, device=self.device)
        q_tar = torch.as_tensor(q_goal_norm, dtype=torch.float32, device=self.device)
        xp = torch.cat((q_src, q_tar), dim=0)
        pts = [q_src.detach().clone()]
        reached = bool(torch.linalg.vector_norm(q_src - q_tar).item() < tol)
        for _ in range(max_steps):
            grad = self.model.function.Gradient(xp[None, :])[0].detach()
            with torch.no_grad():
                xp = xp.detach().clone()
                xp[:6].add_(float(step_size) * grad[:6])
                xp[:6].clamp_(-0.5, 0.5)
                valid = torch.all(torch.isfinite(grad[:6])) & torch.all(torch.isfinite(xp[:6]))
                reached_now = torch.linalg.vector_norm(xp[:6] - q_tar) < float(tol)
                status = torch.stack((valid, reached_now)).detach().cpu().numpy()
            if not bool(status[0]):
                break
            pts.append(xp[:6].detach().clone())
            if bool(status[1]):
                reached = True
                break
        if reached and bool(torch.all(torch.isfinite(q_tar)).item()):
            pts.append(q_tar.detach().clone())
        pts = torch.stack(pts, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        if pts.ndim != 2:
            return np.zeros((0, 6), dtype=np.float32)
        finite_mask = np.all(np.isfinite(pts), axis=1)
        return pts[finite_mask]
