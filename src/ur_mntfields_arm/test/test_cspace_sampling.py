import numpy as np

from ur_mntfields_arm.cspace_sampling import (
    _project_out_of_collision_to_clearance_band,
    _project_toward_clearance_band,
    _sample_near_target_clearances,
    sample_cspace_training_batch,
    sample_path_centered_training_batch,
)


class _UnitBoxKinematics:
    joint_min = np.full((6,), -1.0, dtype=np.float64)
    joint_max = np.full((6,), 1.0, dtype=np.float64)

    def normalize(self, q):
        return (0.5 * np.asarray(q, dtype=np.float64)).astype(np.float32)

    def denormalize(self, qn):
        return (2.0 * np.asarray(qn, dtype=np.float64)).astype(np.float32)

    def clamp(self, q):
        return np.clip(np.asarray(q, dtype=np.float64), self.joint_min, self.joint_max)

    def solve_ik(self, _pose, _seed):
        # A deterministic reachable state is sufficient to exercise the
        # bounded ROI-seed path without bringing ROS IK into this unit test.
        return np.full((6,), 0.40, dtype=np.float64)


class _PlanarClearanceChecker:
    """A monotonic q[0] clearance model with a recorded query cap."""

    def __init__(self):
        self.batch_sizes = []

    def clearance_and_normal_batch(self, q_batch):
        q = np.asarray(q_batch, dtype=np.float32)
        self.batch_sizes.append(len(q))
        clearance = np.maximum(q[:, 0], 0.0).astype(np.float32)
        normals = np.zeros_like(q)
        normals[:, 0] = 1.0
        return clearance, normals


class _NoProjectionClearanceChecker(_PlanarClearanceChecker):
    def clearance_and_normal_batch(self, q_batch):
        q = np.asarray(q_batch, dtype=np.float32)
        self.batch_sizes.append(len(q))
        clearance = np.full((len(q),), 0.10, dtype=np.float32)
        normals = np.zeros_like(q)
        normals[:, 0] = 1.0
        return clearance, normals


class _NoValidClearanceChecker(_PlanarClearanceChecker):
    def clearance_and_normal_batch(self, q_batch):
        q = np.asarray(q_batch, dtype=np.float32)
        self.batch_sizes.append(len(q))
        return np.zeros((len(q),), dtype=np.float32), np.zeros_like(q)


def test_collision_projection_recovers_a_verified_free_boundary_shell():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    q, clearance, normal = _project_out_of_collision_to_clearance_band(
        checker,
        kinematics,
        q_seed=np.array([[-0.50, 0, 0, 0, 0, 0]], dtype=np.float32),
        normal_seed_q=np.array([[1.0, 0, 0, 0, 0, 0]], dtype=np.float32),
        target_clearance=np.array([0.01], dtype=np.float32),
        minimum_clearance_m=0.01,
    )
    assert q.shape == (1, 6)
    assert 0.01 <= clearance[0] <= 0.03
    assert normal[0, 0] == 1.0


def test_near_sampler_retains_obstacle_side_floor_speed_endpoints():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=96,
        clearance_margin_m=0.10,
        clearance_offset_m=0.01,
        rng=np.random.default_rng(43),
        sampling_mode="joint_local_6d",
        proposal_batch_size=16,
        near_boundary_only=True,
    )
    _assert_valid_rows(rows, checker, cap=16)
    assert np.any(rows[:, 13] <= 0.1001)
    assert stats["q1_obstacle_side_frac"] > 0.0


def test_near_sampler_also_retains_the_free_side_of_the_boundary_shell():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=96,
        clearance_margin_m=0.10,
        clearance_offset_m=0.01,
        rng=np.random.default_rng(43),
        sampling_mode="joint_local_6d",
        proposal_batch_size=16,
        near_boundary_only=True,
    )
    _assert_valid_rows(rows, checker, cap=16)
    # Shell pairs must teach both sides of the transition.  Regressing to an
    # always-inward q1 makes safe but sparsely covered states collapse to the
    # field floor during late online training.
    assert np.any(rows[:, 13] >= 0.80)
    assert 0.0 < stats["q1_obstacle_side_frac"] < 0.60
    # Reproduce v6's balanced local shell. Global goal conditioning is created
    # by endpoint reshuffling, not by making the labelled local proposal wide.
    displacement = np.linalg.norm(rows[:, 6:12] - rows[:, :6], axis=1)
    assert np.max(displacement) > 0.06
    assert np.max(displacement) <= 0.1201


def _assert_valid_rows(rows, checker, cap):
    assert rows.ndim == 2 and rows.shape[1] == 26 and len(rows) > 0
    assert max(checker.batch_sizes) <= cap
    # Retained q1 samples must be genuinely in the domain, not clipped atoms.
    assert np.all(rows[:, 6:12] > -0.5)
    assert np.all(rows[:, 6:12] < 0.5)


def test_path_centered_sampler_obeys_clearance_cap_and_stays_near_anchor():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, _stats = sample_path_centered_training_batch(
        checker,
        kinematics,
        anchor_qs=np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        num_pairs=24,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(9),
        proposal_batch_size=4,
    )
    _assert_valid_rows(rows, checker, cap=4)
    # This stage now closes path-coverage holes instead of duplicating the
    # separate near-boundary sampler. q0 remains within the configured 0.025
    # normalized radius of the supplied anchor.
    anchor_n = kinematics.normalize(np.array([0.75, 0, 0, 0, 0, 0], dtype=np.float32))
    assert np.max(np.linalg.norm(rows[:, :6] - anchor_n[None, :], axis=1)) <= 0.0251


def test_joint_local_sampler_keeps_in_bounds_endpoints_under_the_query_cap():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, _stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=24,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(11),
        seed_hint_q=np.array([0.70, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sampling_mode="joint_local_6d",
        proposal_batch_size=4,
        near_boundary_only=True,
    )
    _assert_valid_rows(rows, checker, cap=4)
    assert np.any(rows[:, 12] <= 0.35)


def test_joint_local_near_sampler_retains_a_critical_speed_quota():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=48,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(29),
        seed_hint_q=np.array([0.70, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sampling_mode="joint_local_6d",
        proposal_batch_size=4,
        near_boundary_only=True,
    )

    _assert_valid_rows(rows, checker, cap=4)
    assert stats["speed0_critical_frac"] >= 0.50


def test_joint_local_near_sampler_handles_a_round_without_projectable_seeds():
    checker = _NoProjectionClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, _stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=1,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(31),
        sampling_mode="joint_local_6d",
        proposal_batch_size=4,
        near_boundary_only=True,
    )

    assert rows.ndim == 2 and rows.shape[1] == 26
    assert max(checker.batch_sizes) <= 4


def test_joint_local_sampler_does_not_fall_back_to_an_unbounded_rejection_loop():
    checker = _NoValidClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, _stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=4,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(37),
        sampling_mode="joint_local_6d",
        proposal_batch_size=4,
        near_boundary_only=True,
    )

    assert rows.shape == (0, 26)
    # The bounded fast sampler allows 24 proposals per requested near row.
    # The former legacy fallback would add another 60 proposals per row.
    assert sum(checker.batch_sizes) <= 4 * 24


def test_joint_local_sampler_uses_bounded_roi_ik_seeds_when_roi_is_available():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    _raw, rows, stats = sample_cspace_training_batch(
        checker,
        kinematics,
        num_pairs=24,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
        rng=np.random.default_rng(17),
        samples_per_seed=2,
        roi_min=np.array([0.2, -0.2, 0.2], dtype=np.float64),
        roi_max=np.array([0.8, 0.2, 0.8], dtype=np.float64),
        roi_seed_fraction=0.50,
        sampling_mode="joint_local_6d",
        proposal_batch_size=4,
    )

    _assert_valid_rows(rows, checker, cap=4)
    assert stats["roi_seed_tries"] > 0.0
    assert stats["roi_seed_success"] > 0.0


def test_near_clearance_targets_prioritize_the_low_speed_label_band():
    targets = _sample_near_target_clearances(
        np.random.default_rng(23),
        1000,
        clearance_margin_m=0.15,
        clearance_offset_m=0.015,
    )
    labels = targets / 0.15
    assert float(np.mean(labels <= 0.20)) >= 0.50
    assert float(np.mean(labels <= 0.60)) >= 0.80


def test_clearance_projection_uses_a_bounded_two_query_slope_estimate():
    checker = _PlanarClearanceChecker()
    kinematics = _UnitBoxKinematics()
    q_seed = np.full((4, 6), 0.75, dtype=np.float32)
    clearance_seed, normals = checker.clearance_and_normal_batch(q_seed)

    q_projected, clearance_projected, _normals_projected = _project_toward_clearance_band(
        checker,
        kinematics,
        q_seed,
        clearance_seed,
        normals,
        np.full((4,), 0.02, dtype=np.float32),
        minimum_clearance_m=0.015,
    )

    assert q_projected.shape == q_seed.shape
    assert np.all((clearance_projected >= 0.015) & (clearance_projected <= 0.03))
    # One initial label query, then one probe and one final re-query.
    assert checker.batch_sizes == [4, 4, 4]
