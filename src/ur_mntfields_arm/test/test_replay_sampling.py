import numpy as np

from ur_mntfields_arm.arm_field_model import ArmFieldModel


def _rows_with_speed(values):
    rows = np.zeros((len(values), 26), dtype=np.float32)
    rows[:, 12] = values
    rows[:, 13] = values
    rows[:, 14] = 1.0
    rows[:, 20] = 1.0
    return rows


def test_replay_sampler_is_uniform_without_speed_quota(tmp_path):
    model = ArmFieldModel(
        model_dir=str(tmp_path / "model"),
        device="cpu",
        minibatch_size=16,
    )
    rows = np.concatenate(
        (
            _rows_with_speed(np.full((4,), 0.10, dtype=np.float32)),
            _rows_with_speed(np.full((8,), 0.45, dtype=np.float32)),
            _rows_with_speed(np.full((64,), 1.00, dtype=np.float32)),
        ),
        axis=0,
    )
    rows[:, 0] = np.arange(len(rows), dtype=np.float32)

    np.random.seed(7)
    batch = model._sample_rows_random(rows, 16)

    assert len(batch) == 16
    assert len(np.unique(batch, axis=0)) == 16


def test_recombined_pairs_remain_transient_in_train_step(tmp_path):
    model = ArmFieldModel(model_dir=str(tmp_path / "model"), device="cpu", minibatch_size=4)
    persistent = _rows_with_speed(np.full((8,), 0.50, dtype=np.float32))
    transient = _rows_with_speed(np.full((8,), 0.10, dtype=np.float32))

    # Avoid a costly autograd update; this test verifies replay ownership.
    model.model.train_batch = lambda *_args, **_kwargs: 0.1
    loss = model.train_step(persistent, epochs=1, transient_rows=transient)

    assert loss == 0.1
    assert model.replay_size == len(persistent)


def test_priority_rows_receive_reserved_minibatch_fraction(tmp_path):
    model = ArmFieldModel(
        model_dir=str(tmp_path / "model"),
        device="cpu",
        minibatch_size=20,
        replay_ratio=0.75,
        priority_ratio=0.20,
    )
    ordinary = _rows_with_speed(np.full((80,), 1.0, dtype=np.float32))
    priority = _rows_with_speed(np.full((20,), 0.1, dtype=np.float32))
    model.add_rows(ordinary)

    np.random.seed(11)
    batch = model.sample_training_batch(ordinary, priority_rows=priority)

    assert len(batch) == 20
    assert np.count_nonzero(batch[:, 12] == np.float32(0.1)) == 4


def test_diagnostics_use_the_configured_low_speed_band(tmp_path):
    model = ArmFieldModel(
        model_dir=str(tmp_path / "model"),
        device="cpu",
    )
    rows = _rows_with_speed(np.asarray([0.10, 0.15, 0.19, 0.80], dtype=np.float32))
    model.add_rows(rows)

    def _predicted(batch):
        count = len(batch)
        pred = np.full((count,), 0.25, dtype=np.float32)
        grad = np.zeros((count, 6), dtype=np.float32)
        grad[:, 0] = -1.0
        return pred, pred, grad, grad

    model._predict_replay_gradients = _predicted
    diag = model.evaluate_replay_diagnostics()

    # Three pair rows have both endpoints within the configured 0.20 band.
    assert diag["low_target_count"] == 6.0
    assert diag["low_target_threshold"] == 0.20


def test_pair_reshuffle_preserves_labelled_states_and_changes_local_pairs(tmp_path):
    model = ArmFieldModel(model_dir=str(tmp_path / "model"), device="cpu")
    rows = np.zeros((8, 26), dtype=np.float32)
    for index in range(len(rows)):
        rows[index, 0] = float(index)
        rows[index, 6] = float(index) + 0.01
        rows[index, 12] = float(index) / 20.0
        rows[index, 13] = (float(index) + 0.01) / 20.0
        rows[index, 14] = 1.0
        rows[index, 20] = 1.0

    np.random.seed(12)
    shuffled = model.reshuffle_pair_endpoints(rows)
    original_states = sorted(
        (float(q), float(speed))
        for q, speed in zip(
            np.concatenate((rows[:, 0], rows[:, 6])),
            np.concatenate((rows[:, 12], rows[:, 13])),
        )
    )
    shuffled_states = sorted(
        (float(q), float(speed))
        for q, speed in zip(
            np.concatenate((shuffled[:, 0], shuffled[:, 6])),
            np.concatenate((shuffled[:, 12], shuffled[:, 13])),
        )
    )

    assert shuffled_states == original_states
    assert np.any(np.abs(shuffled[:, 0] - shuffled[:, 6]) > 0.1)
