import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import torch

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.online_training import (
    CertificationMetrics,
    SampleSource,
    StateReplay,
    assign_clearance_sources,
    calibrate_conservative_speed,
    derive_coverage_radius,
    exact_label_states,
    make_state_rows,
    paired_shell_states,
    checkpoint_scene_metadata,
    save_certified_checkpoint,
    scrambled_sobol_states,
)
from ur_mntfields_arm.planner import _LearnedSpeedChecker
from ur_mntfields_arm.voxel_map import SparseVoxelMap


def _rows(q, clearance, source, version=1, speed=None):
    q = np.asarray(q, dtype=np.float32).reshape(-1, 6)
    clearance = np.broadcast_to(clearance, (len(q),))
    if speed is None:
        speed = np.clip(clearance / 0.1, 0.0, 1.0)
    normal = np.zeros_like(q)
    normal[:, 0] = 1.0
    return make_state_rows(q, clearance, speed, normal, clearance <= 0, source, version)


def test_scrambled_sobol_is_deterministic_and_bounded():
    first = scrambled_sobol_states(128, seed=9)
    second = scrambled_sobol_states(128, seed=9)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= -0.5) and np.all(first <= 0.5)


def test_replay_dedup_keeps_false_free_and_invalidates_old_map_versions():
    replay = StateReplay(capacity=20, grid_size=0.01)
    q = np.zeros((1, 6), dtype=np.float32)
    replay.add(_rows(q, 0.08, SampleSource.BROAD, version=1))
    replay.add(_rows(q + 0.001, 0.0, SampleSource.FALSE_FREE, version=1))
    assert len(replay) == 1
    assert int(replay.rows[0, 15]) == int(SampleSource.FALSE_FREE)
    replay.set_map_version(2)
    assert len(replay.valid_rows()) == 0
    assert len(replay.stale_rows()) == 1


def test_source_reservoir_caps_apply_before_total_capacity():
    replay = StateReplay(capacity=100, grid_size=1.0e-5)
    rng = np.random.default_rng(4)
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(80, 6)), 0.01, SampleSource.BOUNDARY_SHELL))
    counts = replay.source_counts(valid_only=False)
    assert counts["boundary_shell"] == 25
    assert len(replay) == 25


def test_balanced_batch_redistributes_missing_hard_sources_away_from_shell():
    replay = StateReplay(capacity=1000, grid_size=1.0e-5)
    replay.set_map_version(1)
    rng = np.random.default_rng(7)
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(250, 6)), 0.01, SampleSource.BOUNDARY_SHELL))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(250, 6)), 0.12, SampleSource.BROAD))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(200, 6)), 0.06, SampleSource.FREE_BAND))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(50, 6)), 0.08, SampleSource.COVERAGE))

    batch = replay.sample_balanced(1000)
    source = np.rint(batch[:, 15]).astype(np.int32)
    assert np.count_nonzero(source == int(SampleSource.BOUNDARY_SHELL)) == 250
    assert np.count_nonzero(source == int(SampleSource.BROAD)) > 250
    assert np.count_nonzero(source == int(SampleSource.FREE_BAND)) > 200
    assert np.count_nonzero(source == int(SampleSource.COVERAGE)) > 50


def test_tiny_hard_reservoir_is_not_repeated_more_than_four_times():
    replay = StateReplay(capacity=10_000, grid_size=1.0e-5)
    replay.set_map_version(1)
    rng = np.random.default_rng(17)
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(3000, 6)), 0.12, SampleSource.BROAD))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(2000, 6)), 0.06, SampleSource.FREE_BAND))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(500, 6)), 0.08, SampleSource.COVERAGE))
    hard_q = rng.uniform(-0.49, 0.49, size=(4, 6))
    replay.add(_rows(hard_q, 0.0, SampleSource.FALSE_FREE))

    batch = replay.sample_balanced(2048)
    hard = batch[np.rint(batch[:, 15]).astype(int) == int(SampleSource.FALSE_FREE)]
    assert len(hard) == 16
    _unique, counts = np.unique(np.round(hard[:, :6], 6), axis=0, return_counts=True)
    assert np.max(counts) <= 4


def test_optimizer_batch_balances_labels_as_well_as_replay_sources():
    replay = StateReplay(capacity=20_000, grid_size=1.0e-5)
    replay.set_map_version(1)
    rng = np.random.default_rng(23)
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(4000, 6)), 0.12, SampleSource.BROAD))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(3000, 6)), 0.06, SampleSource.FREE_BAND))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(4000, 6)), 0.0, SampleSource.BOUNDARY_SHELL))
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(2000, 6)), 0.08, SampleSource.TRAJECTORY))

    batch = replay.sample_learning_balanced(2048)
    safe = batch[:, 14] < 0.5
    high_free = safe & (batch[:, 7] >= 0.5)
    hard = np.isin(
        np.rint(batch[:, 15]).astype(int),
        (int(SampleSource.FALSE_FREE), int(SampleSource.TRAJECTORY)),
    )
    assert len(batch) == 2048
    assert np.mean(safe) >= 0.55
    assert np.mean(high_free) >= 0.45
    assert np.mean(hard) >= 0.09


def test_stale_relabelling_returns_every_row_when_budget_is_sufficient():
    replay = StateReplay(capacity=100, grid_size=1.0e-5)
    rng = np.random.default_rng(31)
    replay.add(_rows(rng.uniform(-0.49, 0.49, size=(20, 6)), 0.08, SampleSource.BROAD, version=1))
    replay.set_map_version(2)
    stale = replay.stale_rows_balanced(100)
    assert len(stale) == len(replay.stale_rows())


def test_clearance_source_assignment_preserves_verified_free_band():
    q = np.zeros((5, 6), dtype=np.float32)
    rows = _rows(
        q,
        np.asarray([-0.01, 0.0, 0.02, 0.06, 0.12]),
        SampleSource.BROAD,
    )
    tagged = assign_clearance_sources(rows)
    assert np.rint(tagged[:, 15]).astype(int).tolist() == [
        int(SampleSource.BOUNDARY_SHELL),
        int(SampleSource.BOUNDARY_SHELL),
        int(SampleSource.BOUNDARY_SHELL),
        int(SampleSource.FREE_BAND),
        int(SampleSource.BROAD),
    ]


def test_state_readiness_false_free_respects_unsafe_rejection():
    model = object.__new__(ArmFieldModel)
    model.replay_buffer = np.zeros((2, 17), dtype=np.float32)
    model.replay_buffer[:, 7] = 0.10
    model.replay_size = 2
    model.last_diagnostics = {}
    model.predict_normalized_state_geometry = lambda _q: (
        np.asarray([0.8, 0.8], dtype=np.float32),
        np.asarray([0.2, 0.0], dtype=np.float32),
        np.asarray([0.8, 0.8], dtype=np.float32),
    )
    diag = model.evaluate_replay_diagnostics(max_rows=2)
    assert np.isclose(diag["low_target_overpred_frac"], 0.5)


def test_state_head_learns_synthetic_boundary_without_depressing_free_space(tmp_path):
    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    q = rng.uniform(-0.5, 0.5, size=(2048, 6)).astype(np.float32)
    unsafe = q[:, 0] >= 0.0
    speed = np.where(unsafe, 0.0, 1.0).astype(np.float32)
    clearance = np.where(unsafe, -0.01, 0.12).astype(np.float32)
    normal = np.zeros_like(q)
    normal[:, 0] = -1.0
    source = np.where(
        unsafe, int(SampleSource.BOUNDARY_SHELL), int(SampleSource.BROAD)
    )
    rows = make_state_rows(
        q, clearance, speed, normal, unsafe.astype(np.float32), source, 1
    )
    model = ArmFieldModel(str(tmp_path / "synthetic_boundary"), device="cpu", minibatch_size=128)
    model.state_replay.set_map_version(1)
    model.train_step(rows, epochs=80)
    predicted_speed, unsafe_probability, _conservative = (
        model.predict_normalized_state_geometry(q)
    )
    assert float(np.mean(predicted_speed[~unsafe])) >= 0.80
    assert float(np.mean(predicted_speed[unsafe])) <= 0.20
    assert float(np.mean(unsafe_probability[~unsafe] < 0.10)) >= 0.75
    assert float(np.mean(unsafe_probability[unsafe] >= 0.10)) >= 0.95


def test_shell_pair_has_verified_orientation_distance():
    boundary = np.zeros((2, 6), dtype=np.float32)
    normal = np.zeros_like(boundary)
    normal[:, 2] = 1.0
    inside, outside = paired_shell_states(boundary, normal)
    np.testing.assert_allclose(inside[:, 2], -0.005)
    np.testing.assert_allclose(outside[:, 2], 0.005)


def test_voxel_unknown_is_not_free_and_artifact_roundtrips(tmp_path):
    voxel_map = SparseVoxelMap(
        voxel_size=0.02,
        roi_min=np.array([0.0, 0.0, 0.0]),
        roi_max=np.array([0.1, 0.1, 0.1]),
    )
    assert not voxel_map.is_observed_free(np.array([0.05, 0.05, 0.05]))
    voxel_map.integrate_points(
        np.array([-0.05, 0.01, 0.01]), np.array([[0.09, 0.01, 0.01]])
    )
    assert voxel_map.map_version == 1
    artifact = voxel_map.save(tmp_path / "scene_map.npz")
    restored = SparseVoxelMap.load(artifact)
    assert restored.scene_signature() == voxel_map.scene_signature()


def test_new_free_ray_clears_stale_occupied_voxel():
    voxel_map = SparseVoxelMap(voxel_size=0.10)
    origin = np.asarray([0.01, 0.01, 0.01])
    stale_hit = np.asarray([0.25, 0.01, 0.01])
    farther_hit = np.asarray([0.45, 0.01, 0.01])
    stale_key = voxel_map._key(stale_hit)

    voxel_map.integrate_points(origin, stale_hit[None, :])
    assert stale_key in voxel_map.occupied
    voxel_map.integrate_points(origin, farther_hit[None, :])

    assert stale_key not in voxel_map.occupied
    assert stale_key in voxel_map.free
    assert voxel_map._key(farther_hit) in voxel_map.occupied


def test_current_endpoints_win_over_same_frame_free_rays_independent_of_order():
    origin = np.asarray([0.01, 0.01, 0.01])
    endpoints = np.asarray([[0.25, 0.01, 0.01], [0.45, 0.01, 0.01]])
    maps = []
    for points in (endpoints, endpoints[::-1]):
        voxel_map = SparseVoxelMap(voxel_size=0.10)
        voxel_map.integrate_points(origin, points)
        maps.append(voxel_map)
        assert voxel_map._key(endpoints[0]) in voxel_map.occupied
        assert voxel_map._key(endpoints[1]) in voxel_map.occupied
    assert maps[0].occupied == maps[1].occupied
    assert maps[0].free == maps[1].free


def test_known_robot_centers_clear_self_filtered_unknown_voxels():
    voxel_map = SparseVoxelMap(
        voxel_size=0.02,
        roi_min=np.array([0.0, 0.0, 0.0]),
        roi_max=np.array([0.2, 0.2, 0.2]),
    )
    center = np.array([0.11, 0.09, 0.07])
    key = voxel_map._key(center)
    voxel_map.occupied.add(key)

    assert not voxel_map.is_observed_free(center)
    assert voxel_map.integrate_known_free_points(center[None, :]) == 1
    assert voxel_map.is_observed_free(center)
    assert key not in voxel_map.occupied

    # Re-observing the same robot center does not churn map versions.
    version = voxel_map.map_version
    assert voxel_map.integrate_known_free_points(center[None, :]) == 0
    assert voxel_map.map_version == version


def test_known_robot_sphere_carves_local_volume_and_stale_self_hits():
    voxel_map = SparseVoxelMap(
        voxel_size=0.02,
        roi_min=np.array([0.0, 0.0, 0.0]),
        roi_max=np.array([0.3, 0.3, 0.3]),
    )
    center = np.array([0.15, 0.15, 0.15])
    occupied_point = np.array([0.17, 0.15, 0.15])
    occupied_key = voxel_map._key(occupied_point)
    voxel_map.occupied.add(occupied_key)

    changed = voxel_map.integrate_known_free_spheres(center[None, :], np.array([0.06]))

    assert changed > 1
    assert voxel_map.is_observed_free(center)
    assert voxel_map.is_observed_free(occupied_point)
    assert occupied_key not in voxel_map.occupied
    assert not voxel_map.is_observed_free(np.array([0.25, 0.15, 0.15]))


def test_known_robot_volume_is_retained_outside_mapping_envelope():
    voxel_map = SparseVoxelMap(
        voxel_size=0.02,
        roi_min=np.array([0.5, 0.0, 0.0]),
        roi_max=np.array([1.0, 1.0, 1.0]),
        approach_envelope_m=0.5,
    )
    # The camera map starts at x=0, but the known proximal link extends a few
    # centimetres behind it. Arbitrary nearby space remains unknown.
    robot_center = np.array([-0.05, 0.2, 0.2])
    voxel_map.integrate_known_free_spheres(robot_center[None, :], np.array([0.06]))

    assert voxel_map.is_observed_free(robot_center)
    assert not voxel_map.is_observed_free(np.array([-0.15, 0.2, 0.2]))


def test_calibration_is_subtractive_99th_percentile_and_radius_is_capped():
    target = np.linspace(0.0, 1.0, 1000)
    prediction = target + 0.1
    margins = calibrate_conservative_speed(prediction, target, np.full(1000, 0.02))
    assert np.isclose(margins["global"], 0.1)
    distances = np.linspace(0.0, 0.2, 1000)
    false_free = distances > 0.081
    assert derive_coverage_radius(distances, false_free) <= 0.08


def test_calibration_and_inference_use_the_same_predicted_speed_bins():
    prediction = np.concatenate((np.full(100, 0.5), np.full(100, 0.8)))
    target = np.concatenate((np.full(100, 0.2), np.full(100, 0.8)))
    # Exact clearance deliberately cannot identify the inference bin. The
    # predicted 0.5 and 0.8 populations must receive separate margins.
    exact_clearance = np.full(200, 0.2)
    margins = calibrate_conservative_speed(
        prediction, target, exact_clearance, coverage_distance=np.zeros(200)
    )
    assert np.isclose(margins["clearance_2_coverage_0"], 0.3)
    assert np.isclose(margins["clearance_3_coverage_0"], 0.0)


def test_balanced_certification_rejects_fail_closed_route_set():
    metrics = CertificationMetrics(
        low_clearance_states=1200,
        low_clearance_false_free_rate=0.01,
        route_attempts=200,
        route_acceptance_rate=0.59,
        accepted_goal_reach_rate=1.0,
        accepted_collision_free_rate=1.0,
        scene_version_match=True,
        direct_edge_median_ms=10.0,
        planning_p95_ms=50.0,
    )
    assert not metrics.passed


class _Identity:
    def normalize(self, q):
        return np.asarray(q, dtype=np.float32)


class _GeometryField:
    def __init__(self):
        self.coverage_states = np.zeros((1, 6), dtype=np.float32)
        self.coverage_tree = cKDTree(self.coverage_states)
        self.shell_coverage_radius = 0.08
        self.free_coverage_radius = 0.08

    def predict_normalized_state_geometry(self, q):
        speed = np.full(len(q), 0.8, dtype=np.float32)
        unsafe = (q[:, 0] > 0.2).astype(np.float32)
        return speed, unsafe, speed.copy()


def test_geometry_oracle_rejects_unsafe_and_unsupported_states():
    checker = _LearnedSpeedChecker(_GeometryField(), _Identity(), np.zeros(6))
    values = checker.clearance_batch(
        np.asarray([[0.01, 0, 0, 0, 0, 0], [0.15, 0, 0, 0, 0, 0], [0.3, 0, 0, 0, 0, 0]], dtype=np.float32)
    )
    np.testing.assert_allclose(values, [0.8, 0.0, 0.0])
    assert checker.rejected_coverage == 2
    assert checker.rejected_unsafe == 1


def test_checkpoint_final_name_requires_certification(tmp_path):
    model = ArmFieldModel(str(tmp_path / "model"), device="cpu")
    try:
        model.save_checkpoint(tmp_path / "weights_final.pt")
    except RuntimeError as exc:
        assert "certification" in str(exc)
    else:
        raise AssertionError("uncertified final checkpoint was written")


def test_certified_checkpoint_requires_and_matches_voxel_artifact(tmp_path):
    model_dir = tmp_path / "model"
    model = ArmFieldModel(str(model_dir), device="cpu")
    model.set_coverage_support(np.zeros((1, 6), dtype=np.float32), 0.05, 0.05)
    voxel_map = SparseVoxelMap(voxel_size=0.02)
    voxel_map.free.add((0, 0, 0))
    voxel_map.map_version = 1
    artifact = voxel_map.save(model_dir / "voxel_map_final.npz")
    model.certification_passed = True
    checkpoint = model_dir / "weights_final.pt"
    model.save_checkpoint(checkpoint, metadata={
        "certification_passed": True,
        "scene_signature": voxel_map.scene_signature(),
        "voxel_map_path": artifact.name,
        "map_version": voxel_map.map_version,
    })
    restored = ArmFieldModel(str(tmp_path / "restored"), device="cpu")
    restored.load_checkpoint(
        checkpoint,
        certified_execution=True,
        current_scene_signature=voxel_map.scene_signature(),
        current_map_version=voxel_map.map_version,
    )
    assert restored.certification_passed


def test_balanced_certification_promotes_final_with_scene_and_coverage(tmp_path):
    model_dir = tmp_path / "model"
    model = ArmFieldModel(str(model_dir), device="cpu")
    model.set_coverage_support(np.zeros((4, 6), dtype=np.float32), 0.04, 0.06)
    replay = StateReplay()
    replay.set_map_version(1)
    replay.add(_rows(np.zeros((1, 6)), 0.05, SampleSource.BROAD, version=1))
    voxel_map = SparseVoxelMap(voxel_size=0.02)
    voxel_map.free.add((0, 0, 0))
    voxel_map.map_version = 1
    metadata = checkpoint_scene_metadata(voxel_map, replay, model_dir, training_wall_time=600.0)
    metrics = CertificationMetrics(
        low_clearance_states=1000,
        low_clearance_false_free_rate=0.02,
        route_attempts=200,
        route_acceptance_rate=0.60,
        accepted_goal_reach_rate=0.95,
        accepted_collision_free_rate=0.95,
        scene_version_match=True,
        direct_edge_median_ms=25.0,
        planning_p95_ms=100.0,
    )
    checkpoint = save_certified_checkpoint(model, model_dir, metrics, metadata)
    assert checkpoint.name == "weights_final.pt"
    restored = ArmFieldModel(str(tmp_path / "restored2"), device="cpu")
    restored.load_checkpoint(checkpoint, certified_execution=True)
    assert restored.certification_passed


def test_sim_training_launch_is_wired_to_online_limits():
    workspace = Path(__file__).resolve().parents[3]
    config = (workspace / "src/ur_mntfields_arm_sim/config/sim_scene.yaml").read_text()
    launch = (workspace / "src/ur_mntfields_arm_sim/launch/ur_mntfields_arm_gz.launch.py").read_text()
    assert "voxel_size_m: 0.02" in config
    assert "replay_buffer_capacity: 300000" in config
    assert "online_active_candidate_count: 65536" in config
    assert "online_certification_route_count: 200" in config
    assert "training_wall_time_limit_s: 720.0" in config
    assert "online_map_freeze_s: 90.0" in config
    assert "state_readiness_min_free_recall: 0.60" in config
    assert "field_diag_min_near_far_gap" not in config
    assert 'LaunchConfiguration("output_dir")' in launch
    assert '"require_fresh_output_dir": True' in launch


def test_exact_state_labelling_bounds_collision_and_sphere_batches():
    class Kinematics:
        joint_min = np.full(6, -1.0, dtype=np.float32)
        joint_max = np.full(6, 1.0, dtype=np.float32)

        def normalize(self, q):
            return 0.5 * np.asarray(q, dtype=np.float32)

    class Checker:
        def __init__(self):
            self.clearance_batches = []
            self.sphere_batches = []

        def clearance_and_normal_batch(self, q):
            self.clearance_batches.append(len(q))
            normal = np.zeros((len(q), 6), dtype=np.float32)
            normal[:, 0] = 1.0
            return np.full(len(q), 0.05, dtype=np.float32), normal

        def robot_spheres_batch(self, q):
            self.sphere_batches.append(len(q))
            return np.zeros((len(q), 2, 3), dtype=np.float32), np.full((len(q), 2), 0.01, dtype=np.float32)

    class ObservedMap:
        map_version = 3

        def is_observed_free(self, _point):
            return True

    checker = Checker()
    rows = exact_label_states(
        checker,
        Kinematics(),
        ObservedMap(),
        np.zeros((1000, 6), dtype=np.float32),
        state_batch_size=128,
    )
    assert rows.shape == (1000, 17)
    assert max(checker.clearance_batches) == 128
    assert max(checker.sphere_batches) == 128
    assert len(checker.clearance_batches) == 8

    class UnknownMap:
        map_version = 4

        def is_observed_free(self, _point):
            return False

    unknown_rows, observed = exact_label_states(
        checker,
        Kinematics(),
        UnknownMap(),
        np.zeros((3, 6), dtype=np.float32),
        return_observed=True,
    )
    # Unknown support is no longer conflated with physical collision. The
    # explorer excludes these free rows from replay and the coverage tree.
    assert not np.any(observed)
    assert np.all(unknown_rows[:, 14] == 0.0)
    assert np.all(unknown_rows[:, 7] > 0.0)
