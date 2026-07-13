import numpy as np

from ur_mntfields_arm.exploration_manager import ArmMNTFieldsExplorer


class _Kinematics:
    joint_min = np.full(6, -1.0, dtype=np.float64)
    joint_max = np.full(6, 1.0, dtype=np.float64)

    def normalize(self, q):
        return (0.5 * np.asarray(q, dtype=np.float64)).astype(np.float32)


class _Checker:
    def __init__(self, clearance):
        self.clearance = float(clearance)

    def clearance_batch(self, q):
        return np.full(len(q), self.clearance, dtype=np.float32)


class _Field:
    def __init__(self, prediction):
        self.prediction = float(prediction)

    def predict_normalized_pair_speeds(self, q0, q1, batch_size=1024):
        del q1, batch_size
        pred = np.full(len(q0), self.prediction, dtype=np.float32)
        return pred, pred.copy()


def _explorer(prediction):
    node = object.__new__(ArmMNTFieldsExplorer)
    node.field_false_free_audit_enabled = True
    node.field_false_free_audit_samples = 128
    node.field_false_free_audit_goals_per_state = 4
    node.field_false_free_target_speed_max = 0.20
    node.field_false_free_pred_speed_min = 0.20
    node.field_false_free_max_rate = 0.05
    node.field_false_free_min_low_states = 16
    node.clearance_margin_m = 0.10
    node.clearance_offset_m = 0.01
    node.clearance_label_floor = 0.0
    node.clearance_label_power = 1.0
    node.kinematics = _Kinematics()
    node.field_model = _Field(prediction)
    node.step_idx = 7
    node.last_false_free_audit_step = -1
    node.last_false_free_audit = {}
    node.hard_failed_anchor_qs = np.zeros((0, 6), dtype=np.float64)
    node.hard_failed_anchor_buffer_limit = 256
    return node


def test_false_free_audit_blocks_and_hard_mines_overprediction():
    node = _explorer(prediction=0.35)
    passed, reason = node._field_false_free_audit(_Checker(clearance=0.0))

    assert not passed
    assert "false_free=512/512" in reason
    assert node.hard_failed_anchor_qs.shape == (64, 6)

    # A second finish path in the same map step reuses the audit and does not
    # duplicate hard examples.
    passed_cached, _ = node._field_false_free_audit(_Checker(clearance=0.0))
    assert not passed_cached
    assert node.hard_failed_anchor_qs.shape == (64, 6)


def test_false_free_audit_accepts_low_prediction_on_obstacle_states():
    node = _explorer(prediction=0.12)
    passed, reason = node._field_false_free_audit(_Checker(clearance=0.0))

    assert passed
    assert "false_free=0/512" in reason
    assert len(node.hard_failed_anchor_qs) == 0
