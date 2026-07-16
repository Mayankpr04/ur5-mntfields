import numpy as np
import time
from scipy.spatial import cKDTree

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


class _ExpiredChecker(_FreeChecker):
    expired = True


class _SpeedField:
    def __init__(self):
        self.calls = 0

    def predict_normalized_pair_speeds(self, q0, q1, **_kwargs):
        self.calls += 1
        # Deliberately depend on both inputs so the test also verifies that the
        # requested search target is supplied to every neural query.
        speed = np.clip(q0[:, 0] + q1[:, 0], 0.0, 1.0).astype(np.float32)
        return speed, speed.copy()


class _StateField:
    def __init__(self):
        x = np.linspace(0.0, 0.2, 6, dtype=np.float32)
        self.coverage_states = np.zeros((len(x), 6), dtype=np.float32)
        self.coverage_states[:, 0] = x
        self.coverage_tree = cKDTree(self.coverage_states)
        self.shell_coverage_radius = 0.08
        self.free_coverage_radius = 0.08

    def predict_normalized_state_geometry(self, q):
        speed = np.full(len(q), 0.8, dtype=np.float32)
        unsafe = np.zeros(len(q), dtype=np.float32)
        return speed, unsafe, speed.copy()


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


def test_learned_speed_checker_fails_closed_after_deadline():
    field = _SpeedField()
    checker = _LearnedSpeedChecker(
        field,
        _IdentityKinematics(),
        q_target=np.zeros(6, dtype=np.float32),
        deadline_at=time.perf_counter() - 1.0,
    )
    values = checker.clearance_batch(np.zeros((3, 6), dtype=np.float32))
    np.testing.assert_array_equal(values, np.zeros(3, dtype=np.float32))
    assert checker.deadline_rejections == 3
    assert field.calls == 0


def test_graph_search_stops_immediately_when_learned_deadline_expires():
    planner = ArmFieldPlanner(_SpeedField(), _IdentityKinematics())
    planner.path_joint_edge_weight = 0.1
    path = planner._plan_collision_aware_one_way(
        _ExpiredChecker(),
        np.zeros(6, dtype=np.float32),
        np.full(6, 0.4, dtype=np.float32),
        step_size_q=0.03,
        max_steps=120,
        allow_direct_edge=False,
    )
    assert path.shape == (0, 6)
    assert planner.last_debug["status"] == "learned_time_budget"


def test_edge_min_clearances_interpolates_in_physical_joint_space():
    planner = object.__new__(ArmFieldPlanner)
    planner.kinematics = _IdentityKinematics()
    planner.last_debug = {}
    planner.planning_edge_step_rad = 0.04
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
    # A 0.04-rad physical step gives 2 and 3 points for these two edges.
    assert checker.rows[0].shape == (5, 6)
    assert planner.last_debug["edge_check_edges"] == 2
    assert planner.last_debug["edge_check_states"] == 5


def test_edge_batch_reuses_endpoint_clearances_and_smoothing_reduces_curvature():
    planner = ArmFieldPlanner(_SpeedField(), _IdentityKinematics())
    checker = _FreeChecker()
    candidates = np.asarray(
        [[0.03, 0, 0, 0, 0, 0], [0.06, 0, 0, 0, 0, 0]], dtype=np.float32
    )
    minimum, endpoints = planner._edge_min_clearances(
        checker, np.zeros(6, dtype=np.float32), candidates, 0.03, return_endpoints=True
    )
    np.testing.assert_allclose(minimum, 1.0)
    np.testing.assert_allclose(endpoints, 1.0)

    path = np.asarray(
        [[0, 0, 0, 0, 0, 0], [0.25, 0.35, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    before = float(np.linalg.norm(path[0] - 2.0 * path[1] + path[2]))
    smoothed = planner._smooth_collision_path(checker, path, clearance_margin_m=0.02)
    after = float(np.linalg.norm(smoothed[0] - 2.0 * smoothed[1] + smoothed[2]))
    np.testing.assert_allclose(smoothed[[0, -1]], path[[0, -1]])
    assert after < before


def test_field_planner_returns_learned_safe_direct_edge_without_rollout():
    planner = ArmFieldPlanner(_SpeedField(), _IdentityKinematics())
    q_start = np.asarray([0.1, 0, 0, 0, 0, 0], dtype=np.float32)
    q_goal = np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32)

    path = planner.plan(
        q_start,
        q_goal,
        step_size_q=0.03,
        max_steps=120,
        mode="forward",
        allow_direct_edge=True,
        min_predicted_speed=0.4,
        edge_check_step_rad=0.04,
    )

    np.testing.assert_allclose(path, np.asarray([q_start, q_goal]), atol=1.0e-6)
    assert planner.last_debug["status"] == "learned_direct_edge"
    assert planner.last_debug["steps"] == 0
    assert planner.last_debug["direct_edge_min_speed"] >= 0.4


def test_learned_search_uses_one_batch_replay_bridge():
    planner = ArmFieldPlanner(_StateField(), _IdentityKinematics())
    q_start = np.zeros(6, dtype=np.float32)
    q_goal = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
    path = planner.plan_learned_speed_search(
        q_start,
        q_goal,
        step_size_q=0.04,
        max_steps=120,
        min_predicted_speed=0.20,
        allow_direct_edge=True,
        time_budget_ms=90.0,
    )
    np.testing.assert_allclose(path[[0, -1]], np.asarray([q_start, q_goal]))
    assert planner.last_debug["status"] == "learned_replay_bridge"
    assert planner.last_debug["learned_speed_query_calls"] == 2


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
