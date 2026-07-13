import numpy as np
import torch

from ur_mntfields_arm.arm_field_model import ArmFieldModel


def _pair_rows(count: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    q0 = rng.uniform(-0.45, 0.45, size=(count, 6)).astype(np.float32)
    q1 = rng.uniform(-0.45, 0.45, size=(count, 6)).astype(np.float32)
    speed = rng.uniform(0.05, 1.0, size=(count, 2)).astype(np.float32)
    normals = rng.normal(size=(count, 2, 6)).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1.0e-6)
    normals = normals.reshape(count, 12)
    return np.concatenate((q0, q1, speed, normals), axis=1)


def test_paper_loss_has_finite_unscaled_updates(tmp_path):
    torch.manual_seed(4)
    model = ArmFieldModel(
        model_dir=str(tmp_path / "model"),
        device="cpu",
        minibatch_size=32,
        td_loss_weight=0.0,
        speed_loss_weight=1.0,
        log_speed_loss_weight=0.5,
        direct_speed_loss_weight=0.0,
        normal_loss_weight=0.0,
        normal_cos_loss_weight=0.0,
        near_obstacle_loss_weight=0.0,
        low_speed_penalty_weight=0.0,
    )
    rows = _pair_rows()
    losses = [model.train_step(rows, 1) for _ in range(3)]

    assert all(loss is not None and np.isfinite(loss) for loss in losses)
    # A previous large diagnostic value must not rescale the next backward
    # pass; train_batch now always sends beta=1 to the objective.
    model.model.last_loss = 1.0e8
    captured = {}
    original_loss = model.model.function.Loss

    def recording_loss(points, speed, normal, beta, gamma, epoch):
        captured["beta"] = beta
        return original_loss(points, speed, normal, beta, gamma, epoch)

    model.model.function.Loss = recording_loss
    assert model.train_step(rows, 1) is not None
    assert captured["beta"] == 1.0
