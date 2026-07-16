import numpy as np
import pytest

from ur_mntfields_arm.test_trained_field import (
    _checkpoint_requires_certification,
    FieldPathTest,
    _requires_certified_learned_checkpoint,
)


class _IdentityKinematics:
    @staticmethod
    def normalize(q):
        return np.asarray(q, dtype=np.float32)

    @staticmethod
    def denormalize(q):
        return np.asarray(q, dtype=np.float64)


class _V3Model:
    @staticmethod
    def predict_normalized_state_geometry(q):
        raise AssertionError("shortcut validation must use the planner's conservative oracle")


class _RecordingPlanner:
    def __init__(self):
        self.calls = []
        self.reject_middle = False

    def learned_state_speeds(self, q_batch, q_target=None):
        q = np.asarray(q_batch, dtype=np.float64)
        self.calls.append(q.copy())
        speed = np.full((len(q),), 0.8, dtype=np.float32)
        if self.reject_middle and len(speed) > 2:
            speed[len(speed) // 2] = 0.0
        return speed


def test_anchor_route_expands_requested_cabinet_topology():
    goals = np.arange(18, dtype=np.float64).reshape(3, 6)
    anchor = np.full((6,), -7.0, dtype=np.float64)

    sequence, labels = FieldPathTest._anchor_routed_joint_sequence(
        goals,
        anchor,
        [1, 0, 2, 3, 0, 1],
    )

    np.testing.assert_allclose(sequence, np.vstack((goals[0], anchor, goals[1], goals[2], anchor, goals[0])))
    assert labels == [
        "goal 1",
        "cabinet transition anchor",
        "goal 2",
        "goal 3",
        "cabinet transition anchor",
        "goal 1",
    ]


def test_anchor_route_rejects_invalid_goal_index():
    with pytest.raises(ValueError, match="valid values"):
        FieldPathTest._anchor_routed_joint_sequence(
            np.zeros((3, 6), dtype=np.float64),
            np.zeros((6,), dtype=np.float64),
            [1, 4],
        )


def test_anchor_aliases_require_certified_checkpoint():
    for planner_type in ("field_anchor", "network_anchor", "budgeted_anchor"):
        assert _requires_certified_learned_checkpoint(planner_type)
    assert not _requires_certified_learned_checkpoint("field_search")


def test_only_explicit_anchor_diagnostic_override_accepts_partial_checkpoint():
    assert not _checkpoint_requires_certification(
        "field_anchor", allow_uncertified_anchor_checkpoint=True
    )
    assert _checkpoint_requires_certification(
        "field_anchor", allow_uncertified_anchor_checkpoint=False
    )
    assert _checkpoint_requires_certification(
        "learned_speed_search", allow_uncertified_anchor_checkpoint=True
    )


def test_v3_shortcuts_use_dense_conservative_state_oracle():
    node = object.__new__(FieldPathTest)
    node.kinematics = _IdentityKinematics()
    node.field_model = _V3Model()
    node.planner = _RecordingPlanner()
    node.path_shortcut_interp_step_rad = 0.04
    node.learned_speed_search_min_speed = 0.20

    qa = np.zeros((6,), dtype=np.float64)
    targets = np.full((1, 6), 0.20, dtype=np.float64)
    goal = targets[0]
    accepted = FieldPathTest._field_safe_shortcut_targets(node, qa, targets, goal)
    assert accepted.tolist() == [True]
    assert len(node.planner.calls) == 1
    assert len(node.planner.calls[0]) == 6

    node.planner.reject_middle = True
    rejected = FieldPathTest._field_safe_shortcut_targets(node, qa, targets, goal)
    assert rejected.tolist() == [False]
