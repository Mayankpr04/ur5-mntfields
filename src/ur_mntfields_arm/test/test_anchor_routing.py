import numpy as np
import pytest

from ur_mntfields_arm.test_trained_field import FieldPathTest


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
