from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import torch
from rclpy.node import Node

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.ur5_kinematics import JOINT_NAMES, UR5Kinematics


class FieldSliceVisualizer(Node):
    def __init__(self):
        super().__init__("field_slice_visualizer")
        self.declare_parameter("ur_type", "ur5")
        self.declare_parameter("model_dir", "/home/mayank/ur_ws/src/ur5_sim_training_factorized_v2/model")
        self.declare_parameter("checkpoint_path", "/home/mayank/ur_ws/src/ur5_sim_training_factorized_v2/model/weights_final.pt")
        self.declare_parameter("output_dir", "/home/mayank/ur_ws/src/ur5_sim_training_factorized_v2/model/field_slices")
        self.declare_parameter("start_q", [0.12, -2.3, 1.9, -2.5, -1.57, 0.0])
        self.declare_parameter("goal_q", [0.12, -2.3, 1.9, -2.5, -1.57, 0.0])
        self.declare_parameter("slice_joint_pairs", [0, 1, 1, 2, 2, 3, 3, 4])
        self.declare_parameter("grid_size", 81)
        self.declare_parameter("slice_span_norm", 0.18)
        self.declare_parameter("quiver_stride", 4)

        self.kinematics = UR5Kinematics(str(self.get_parameter("ur_type").value))
        self.model_dir = str(self.get_parameter("model_dir").value)
        self.checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_q = np.asarray(self.get_parameter("start_q").value, dtype=np.float64).reshape(6)
        self.goal_q = np.asarray(self.get_parameter("goal_q").value, dtype=np.float64).reshape(6)
        pair_vals = [int(v) for v in self.get_parameter("slice_joint_pairs").value]
        self.slice_joint_pairs = [(pair_vals[i], pair_vals[i + 1]) for i in range(0, len(pair_vals), 2)]
        self.grid_size = int(self.get_parameter("grid_size").value)
        self.slice_span_norm = float(self.get_parameter("slice_span_norm").value)
        self.quiver_stride = int(self.get_parameter("quiver_stride").value)

        self.field_model = ArmFieldModel(self.model_dir)
        self.field_model.load_checkpoint(self.checkpoint_path)

    def _evaluate_slice(self, q_start: np.ndarray, q_goal: np.ndarray, idx_i: int, idx_j: int):
        q_start_n = self.kinematics.normalize(q_start)
        q_goal_n = self.kinematics.normalize(q_goal)
        center = q_start_n.copy()
        span = self.slice_span_norm
        xs = np.linspace(center[idx_i] - span, center[idx_i] + span, self.grid_size, dtype=np.float32)
        ys = np.linspace(center[idx_j] - span, center[idx_j] + span, self.grid_size, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")

        samples = np.repeat(center[None, :], self.grid_size * self.grid_size, axis=0).astype(np.float32)
        samples[:, idx_i] = XX.reshape(-1)
        samples[:, idx_j] = YY.reshape(-1)
        samples = np.clip(samples, -0.5, 0.5)
        targets = np.repeat(q_goal_n[None, :].astype(np.float32), len(samples), axis=0)
        xp = np.concatenate((samples, targets), axis=1)

        with torch.no_grad():
            tau = self.field_model.model.function.TravelTimes(torch.from_numpy(xp).to(self.field_model.device))
            tau_np = tau.detach().cpu().numpy().reshape(self.grid_size, self.grid_size)

        xp_t = torch.from_numpy(xp).float().to(self.field_model.device)
        grad = self.field_model.model.function.Gradient(xp_t).detach().cpu().numpy()[:, :6]
        grad_i = grad[:, idx_i].reshape(self.grid_size, self.grid_size)
        grad_j = grad[:, idx_j].reshape(self.grid_size, self.grid_size)

        goal_xy = (float(q_goal_n[idx_i]), float(q_goal_n[idx_j]))
        start_xy = (float(q_start_n[idx_i]), float(q_start_n[idx_j]))
        return xs, ys, tau_np, grad_i, grad_j, start_xy, goal_xy

    def _plot_slice(self, q_start: np.ndarray, q_goal: np.ndarray, idx_i: int, idx_j: int, suffix: str):
        xs, ys, tau_np, grad_i, grad_j, start_xy, goal_xy = self._evaluate_slice(q_start, q_goal, idx_i, idx_j)
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xs, ys, tau_np, levels=24, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Travel-time potential")

        stride = max(1, self.quiver_stride)
        ax.quiver(
            xs[::stride],
            ys[::stride],
            grad_i[::stride, ::stride],
            grad_j[::stride, ::stride],
            color="white",
            alpha=0.75,
            scale=18.0,
            width=0.003,
        )
        ax.scatter([start_xy[0]], [start_xy[1]], c=["lime"], s=90, label="slice center / start")
        ax.scatter([goal_xy[0]], [goal_xy[1]], c=["red"], s=90, label="goal")
        ax.set_xlabel(f"{JOINT_NAMES[idx_i]} (normalized)")
        ax.set_ylabel(f"{JOINT_NAMES[idx_j]} (normalized)")
        ax.set_title(f"Field Slice {suffix}: {JOINT_NAMES[idx_i]} vs {JOINT_NAMES[idx_j]}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
        out = self.output_dir / f"field_slice_{suffix}_{idx_i}_{idx_j}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        self.get_logger().info(f"Saved field slice: {out}")

    def run(self):
        for idx_i, idx_j in self.slice_joint_pairs:
            self._plot_slice(self.start_q, self.goal_q, idx_i, idx_j, "start_to_goal")


def main():
    rclpy.init()
    node = FieldSliceVisualizer()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# [arm_mntfields_explorer-6] [INFO] [1781233579.133019205] [arm_mntfields_explorer]: step=37 training complete: train_steps=80 loss=0.006199 state_batch_size=512 pair_batch_size=512 replay_size=1024 train_ms=3917.7
