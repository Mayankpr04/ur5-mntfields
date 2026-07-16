import numpy as np

from ur_mntfields_arm.offline_planner_benchmark import (
    CASES,
    GOALS,
    SAFE_TELEOP_GOALS,
    TELEOP_CASES,
    TELEOP_SEQUENCE_CASES,
    TELEOP_GOALS,
    _load_boxes,
)
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker
from pathlib import Path
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics


def test_offline_benchmark_has_two_goals_in_each_included_shelf():
    kin = UR5Kinematics()
    base_world_z = 0.50
    grouped = {"lower": [], "middle": [], "top": []}
    for name, q in GOALS.items():
        grouped[name.split("_", 1)[0]].append(float(kin.fk(q)[2, 3] + base_world_z))

    assert {name: len(values) for name, values in grouped.items()} == {
        "lower": 2,
        "middle": 2,
        "top": 2,
    }
    assert all(0.46 < z < 0.90 for z in grouped["lower"])
    assert all(0.90 < z < 1.34 for z in grouped["middle"])
    assert all(1.34 < z < 1.78 for z in grouped["top"])
    assert min(np.concatenate(tuple(np.asarray(values) for values in grouped.values()))) > 0.46


def test_offline_benchmark_balances_within_and_inter_shelf_cases():
    transitions = [case[3] for case in CASES]
    assert transitions.count("within") == 3
    assert transitions.count("inter") == 3
    assert len({case[1] for case in CASES} | {case[2] for case in CASES}) == 6


def test_teleop_benchmark_uses_all_six_goals_in_two_routes():
    assert len(TELEOP_GOALS) == 6
    assert len(TELEOP_CASES) == 12
    used = {case[1] for case in TELEOP_CASES} | {case[2] for case in TELEOP_CASES}
    assert used == set(TELEOP_GOALS)
    assert TELEOP_CASES[:3] == (
        ("route_a_01_g1_to_g2", "goal1", "goal2", "within"),
        ("route_a_02_g2_to_g5", "goal2", "goal5", "inter"),
        ("route_a_03_g5_to_g6", "goal5", "goal6", "within"),
    )


def test_live_sequence_has_startup_plus_six_goals():
    assert len(TELEOP_SEQUENCE_CASES) == 6
    assert TELEOP_SEQUENCE_CASES[0][1:3] == ("startup", "goal1")
    assert TELEOP_SEQUENCE_CASES[-1][1:3] == ("goal5", "goal6")


def test_safe_teleop_goals_have_twenty_mm_endpoint_clearance():
    kin = UR5Kinematics()
    workspace_src = Path(__file__).resolve().parents[2]
    boxes, support_count = _load_boxes(
        workspace_src / "ur_mntfields_arm_sim/config/sim_scene.yaml",
        np.array([0.15, 0.35, 0.50]),
    )
    checker = UR5PointCloudCollisionChecker(
        kin, np.zeros((0, 3), dtype=np.float32), box_obstacles=boxes,
        support_box_count=support_count,
    )
    clearance = checker.clearance_batch(np.vstack(tuple(SAFE_TELEOP_GOALS.values())).astype(np.float32))
    assert np.all(clearance >= 0.02)
    assert np.linalg.norm(SAFE_TELEOP_GOALS["goal1"] - TELEOP_GOALS["goal1"]) < 0.03
