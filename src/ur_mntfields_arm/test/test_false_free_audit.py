import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped

from ur_mntfields_arm.exploration_manager import ArmMNTFieldsExplorer


class _TFBuffer:
    def __init__(self):
        self.requested_time = None

    def lookup_transform(self, target, source, requested_time):
        assert target == "base_link"
        assert source == "camera_color_optical_frame"
        self.requested_time = requested_time
        transform = TransformStamped()
        transform.transform.rotation.w = 1.0
        transform.transform.translation.x = 0.25
        return transform


class _PoseKinematics:
    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=np.float64)
        return pose

    def tool_to_camera_pose(self, tool_pose, camera_in_tool):
        return np.asarray(tool_pose) @ np.asarray(camera_in_tool)


class _Logger:
    def info(self, _message):
        pass


def test_camera_pose_lookup_uses_depth_exposure_timestamp():
    node = object.__new__(ArmMNTFieldsExplorer)
    node.base_frame = "base_link"
    node.camera_frame = "camera_color_optical_frame"
    node.tf_buffer = _TFBuffer()
    stamp = TimeMsg(sec=123, nanosec=456_000_000)

    pose = node._lookup_camera_pose(stamp)

    assert node.tf_buffer.requested_time.nanoseconds == 123_456_000_000
    np.testing.assert_allclose(pose[:3, 3], [0.25, 0.0, 0.0])


def test_depth_mapping_uses_nearest_timestamped_joint_state_with_tolerance():
    node = object.__new__(ArmMNTFieldsExplorer)
    node.current_joints = np.full(6, 9.0)
    node.mapping_joint_sync_tolerance_s = 0.10
    node.joint_state_history = [
        (1_000_000_000, np.full(6, 1.0)),
        (1_080_000_000, np.full(6, 2.0)),
        (1_300_000_000, np.full(6, 3.0)),
    ]

    matched = node._joint_state_at_stamp(TimeMsg(sec=1, nanosec=70_000_000))
    missing = node._joint_state_at_stamp(TimeMsg(sec=2, nanosec=0))

    np.testing.assert_array_equal(matched, np.full(6, 2.0))
    assert missing is None


def test_base_link_mapping_reconstructs_camera_pose_from_timestamped_joints():
    node = object.__new__(ArmMNTFieldsExplorer)
    node.base_frame = "base_link"
    node.kinematics = _PoseKinematics()
    node.camera_in_tool = np.eye(4)
    node.camera_in_tool[:3, 3] = [0.0, -0.08, 0.02]
    node.mapping_pose_source_logged = False
    node.get_logger = lambda: _Logger()
    node._lookup_camera_pose = lambda _stamp: (_ for _ in ()).throw(
        AssertionError("dynamic TF must not be used for base_link mapping")
    )

    pose = node._camera_pose_at_exposure(
        TimeMsg(sec=4), np.asarray([0.4, 0.3, 0.2, 0.0, 0.0, 0.0])
    )

    np.testing.assert_allclose(pose[:3, 3], [0.4, 0.22, 0.22])
    assert node.mapping_pose_source_logged


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


def test_empty_focused_recovery_sampler_stats_are_loggable():
    node = object.__new__(ArmMNTFieldsExplorer)
    stats = node._merge_sampler_stats([])

    assert stats["sampling_mode"] == "none"
    assert stats["attempts"] == 0.0
    assert stats["ik_seed_tries"] == 0.0
    assert stats["accepted_pairs"] == 0.0
    assert stats["acceptance_rate"] == 0.0
    assert stats["accepted_per_seed"] == 0.0
