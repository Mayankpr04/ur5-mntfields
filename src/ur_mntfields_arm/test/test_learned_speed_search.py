import numpy as np

from ur_mntfields_arm.planner import ArmFieldPlanner, JointSpaceRRTConnectPlanner, _LearnedSpeedChecker


class _IdentityKinematics:
    joint_min = np.full(6, -1.0, dtype=np.float64)
    joint_max = np.full(6, 1.0, dtype=np.float64)

    def normalize(self, q):
        return np.asarray(q, dtype=np.float32)

    def denormalize(self, q):
        return np.asarray(q, dtype=np.float32)

    def clamp(self, q):
        return np.clip(np.asarray(q, dtype=np.float64), self.joint_min, self.joint_max)


class _RecordingChecker:
    def __init__(self):
        self.rows = []

    def clearance_batch(self, q):
        q = np.asarray(q, dtype=np.float32)
        self.rows.append(q.copy())
        return 1.0 - q[:, 0]


class _FreeChecker:
    def clearance_batch(self, q):
        return np.ones(len(np.asarray(q)), dtype=np.float32)


class _SpeedField:
    def __init__(self):
        self.calls = 0

    def predict_normalized_pair_speeds(self, q0, q1):
        self.calls += 1
        # Deliberately depend on both inputs so the test also verifies that the
        # requested search target is supplied to every neural query.
        speed = np.clip(q0[:, 0] + q1[:, 0], 0.0, 1.0).astype(np.float32)
        return speed, speed.copy()


def test_learned_speed_checker_uses_only_batched_network_inference():
    field = _SpeedField()
    checker = _LearnedSpeedChecker(
        field,
        _IdentityKinematics(),
        q_target=np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    values = checker.clearance_batch(
        np.asarray(
            [[0.1, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(values, [0.5, 0.9], atol=1.0e-6)
    assert field.calls == 1
    assert checker.query_calls == 1
    assert checker.query_states == 2

    # Repeated edge endpoints and candidate-state scoring must reuse the same
    # inference rather than invoking the network a second time.
    repeated = checker.clearance_batch(
        np.asarray(
            [[0.5, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(repeated, [0.9, 0.9], atol=1.0e-6)
    assert field.calls == 1
    assert checker.query_calls == 1
    assert checker.query_states == 2
    assert checker.requested_states == 4


def test_edge_min_clearances_interpolates_in_normalized_cspace():
    planner = object.__new__(ArmFieldPlanner)
    planner.kinematics = _IdentityKinematics()
    planner.last_debug = {}
    checker = _RecordingChecker()
    candidates = np.asarray(
        [[0.03, 0, 0, 0, 0, 0], [0.06, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )

    values = planner._edge_min_clearances(
        checker,
        np.zeros(6, dtype=np.float32),
        candidates,
        step_size_q=0.03,
    )

    np.testing.assert_allclose(values, [0.97, 0.94], atol=1.0e-6)
    # Half-step interpolation gives 3 and 5 points for these two edges.
    assert checker.rows[0].shape == (8, 6)
    assert planner.last_debug["edge_check_edges"] == 2
    assert planner.last_debug["edge_check_states"] == 8


def test_rrt_connect_reports_direct_edge_query_work():
    planner = JointSpaceRRTConnectPlanner(_IdentityKinematics(), rng=np.random.default_rng(3))
    path = planner.plan(
        _FreeChecker(),
        np.zeros(6, dtype=np.float32),
        np.full(6, 0.4, dtype=np.float32),
        step_size_q=0.2,
        max_iters=10,
        clearance_margin_m=0.02,
        edge_check_step_rad=0.04,
    )
    assert path.shape == (2, 6)
    assert planner.last_debug["status"] == "direct_edge"
    assert planner.last_debug["edge_query_calls"] == 1
    assert 11 <= planner.last_debug["edge_query_states"] <= 12
