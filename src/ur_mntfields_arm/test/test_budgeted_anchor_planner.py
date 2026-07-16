import numpy as np
import pytest

from ur_mntfields_arm.budgeted_anchor_planner import BudgetedAnchorConfig, BudgetedFieldAnchorPlanner


class _Kinematics:
    joint_min = np.full((6,), -1.0)
    joint_max = np.full((6,), 1.0)

    def clamp(self, q):
        return np.clip(np.asarray(q, dtype=np.float64), self.joint_min, self.joint_max)

    def normalize(self, q):
        return (0.5 * np.asarray(q, dtype=np.float64)).astype(np.float32)

    def fk(self, q):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        pose[:3, 3] = np.asarray(q, dtype=np.float64)[:3]
        return pose

    def solve_ik_fast(self, pose, seed):
        q = np.asarray(seed, dtype=np.float64).copy()
        q[:3] = np.asarray(pose, dtype=np.float64)[:3, 3]
        return self.clamp(q)

    def camera_to_tool_pose(self, camera_pose, camera_in_tool):
        return np.asarray(camera_pose) @ np.linalg.inv(np.asarray(camera_in_tool))

    def tool_to_camera_pose(self, tool_pose, camera_in_tool):
        return np.asarray(tool_pose) @ np.asarray(camera_in_tool)


class _Checker:
    box_obstacles = np.zeros((0, 6), dtype=np.float32)
    occupied_points = np.zeros((0, 3), dtype=np.float32)
    support_box_count = 0

    def clearance_batch(self, rows):
        return np.full((len(np.asarray(rows).reshape(-1, 6)),), 0.10, dtype=np.float32)


class _Model:
    def predict_normalized_pair_speeds(self, states, goals, batch_size=4096):
        return np.full((len(states),), 0.8, dtype=np.float32), None


class _FieldPlanner:
    def __init__(self):
        self.field_model = _Model()
        self.last_debug = {}
        self.raw_plan_calls = 0
        self.learned_plan_calls = 0

    def plan(self, q_start, q_goal, *args, **kwargs):
        self.raw_plan_calls += 1
        self.last_debug = {"status": "learned_direct_edge"}
        return np.asarray([q_start, q_goal], dtype=np.float32)

    def plan_learned_speed_search(
        self, q_start, q_goal, *args, min_predicted_speed=0.2, **kwargs
    ):
        self.learned_plan_calls += 1
        speed = self.learned_edge_min_speed(q_start, q_goal, q_goal)
        if speed < min_predicted_speed:
            self.last_debug = {"status": "learned_coverage_or_speed_rejected"}
            return np.zeros((0, 6), dtype=np.float32)
        self.last_debug = {"status": "learned_direct_edge"}
        return np.asarray([q_start, q_goal], dtype=np.float32)

    def learned_edge_min_speed(self, q_start, q_end, q_goal, max_step_rad=0.04):
        del q_goal, max_step_rad
        return 0.8 if np.linalg.norm(np.asarray(q_end) - np.asarray(q_start)) <= 0.6 else 0.1


def test_anchor_budget_is_hard_capped():
    with pytest.raises(ValueError, match="anchor_budget"):
        BudgetedAnchorConfig(anchor_budget=3)


def test_opening_method_selects_only_one_anchor_with_budget_two():
    checker = _Checker()
    checker.box_obstacles = np.asarray([[0.55, 0.0, 0.3, 0.30, 0.60, 1.20]], dtype=np.float32)
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(
            anchor_budget=2,
            max_candidates=4,
            workspace_shell_standoff_m=0.15,
        ),
    )
    rows = np.asarray([[x, 0, 0, 0, 0, 0] for x in np.linspace(-0.8, 0.8, 9)], dtype=np.float64)
    anchors = planner.select_anchors(checker, rows, rows)
    assert len(anchors) == 1
    assert planner.last_debug["anchor_budget"] == 2
    assert planner.last_debug["candidate_source"] == "opening_center_ik"


def test_scene_geometry_selects_retreat_shell_not_arbitrary_joint_hub():
    checker = _Checker()
    checker.box_obstacles = np.asarray([[0.55, 0.0, 0.0, 0.30, 0.60, 0.60]], dtype=np.float32)
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(
            anchor_budget=1,
            max_candidates=16,
            max_probes=8,
            workspace_shell_standoff_m=0.15,
        ),
    )
    rows = np.asarray([[0.5, y, z, 0, 0, 0] for y in (-0.1, 0.0, 0.1) for z in (-0.1, 0.1)])
    anchors = planner.select_anchors(checker, rows, rows)
    assert len(anchors) == 1
    assert planner.last_debug["candidate_source"] == "opening_center_ik"
    # The nearest obstacle face is x=0.40, so the anchor must retreat toward
    # the robot base instead of being selected from the supplied x=0.50 rows.
    assert anchors[0, 0] < 0.40


def test_opening_face_cannot_be_replaced_by_closer_bottom_face():
    checker = _Checker()
    # Bottom z=-0.10 is closer to the base origin than opening x=0.40. The
    # retreat direction must nevertheless be horizontal and use the opening.
    checker.box_obstacles = np.asarray([[0.55, 0.0, 0.50, 0.30, 0.60, 1.20]], dtype=np.float32)
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(max_candidates=4, workspace_shell_standoff_m=0.15),
    )
    rows = np.zeros((4, 6), dtype=np.float64)
    anchors = planner.select_anchors(checker, rows, rows)
    assert len(anchors) == 1
    assert planner.retreat_face_axis == 0
    assert planner.last_debug["target_camera_xyz"][0] == pytest.approx(0.25)


def test_same_region_route_rejects_weak_learned_edge_without_fallback():
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(anchor_budget=1, learned_min_speed=0.2),
    )
    planner.anchors = np.asarray([[0.5, 0, 0, 0, 0, 0]], dtype=np.float64)
    path = planner.plan(
        np.zeros((6,), dtype=np.float64),
        np.asarray([1.0, 0, 0, 0, 0, 0], dtype=np.float64),
    )
    assert path.shape == (0, 6)
    assert planner.last_debug["status"] == "budgeted_anchor_learned_safety_rejected"


def test_explicit_diagnostic_anchor_uses_raw_field_without_learned_gate():
    field = _FieldPlanner()
    planner = BudgetedFieldAnchorPlanner(
        field,
        _Kinematics(),
        BudgetedAnchorConfig(
            anchor_budget=1,
            learned_min_speed=0.2,
            enforce_learned_safety=False,
        ),
    )
    planner.anchors = np.asarray([[0.2, 0, 0, 0, 0, 0]], dtype=np.float64)
    path = planner.plan(
        np.zeros((6,), dtype=np.float64),
        np.asarray([1.0, 0, 0, 0, 0, 0], dtype=np.float64),
        force_anchor=True,
    )
    assert path.shape == (3, 6)
    assert field.raw_plan_calls == 2
    assert field.learned_plan_calls == 0
    assert planner.last_debug["status"] == "budgeted_anchor_diagnostic_raw_field_reached"
    assert planner.last_debug["learned_safety_enforced"] is False


def test_internal_scene_separator_forces_shared_anchor_route():
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(anchor_budget=1, learned_min_speed=0.2),
    )
    planner.anchors = np.asarray([[0.2, 0.0, 0.0, 0, 0, 0]], dtype=np.float64)
    planner.scene_bounds_min = np.asarray([0.4, -0.5, -0.5], dtype=np.float64)
    planner.scene_bounds_max = np.asarray([0.8, 0.5, 0.5], dtype=np.float64)
    planner.retreat_face_axis = 0
    planner.retreat_face_value = 0.4
    planner.retreat_outward_sign = -1.0
    planner.separator_planes = [(2, 0.0)]
    path = planner.plan(
        np.asarray([0.5, 0.0, -0.3, 0, 0, 0], dtype=np.float64),
        np.asarray([0.5, 0.0, 0.3, 0, 0, 0], dtype=np.float64),
    )
    assert path.shape == (3, 6)
    assert np.allclose(path[1], planner.anchors[0])
    assert planner.last_debug["crosses_scene_separator"] is True
    assert planner.last_debug["routing_reason"] == "scene_separator"


def test_separator_uses_camera_position_in_opening_approach_band():
    camera_in_tool = np.eye(4, dtype=np.float64)
    camera_in_tool[:3, 3] = [0.0, -0.08, 0.02]
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(
            anchor_budget=1,
            learned_min_speed=0.2,
            workspace_shell_standoff_m=0.15,
        ),
        camera_in_tool=camera_in_tool,
    )
    planner.anchors = np.asarray([[0.2, 0.0, 0.0, 0, 0, 0]], dtype=np.float64)
    planner.scene_bounds_min = np.asarray([0.4, -0.5, -0.5], dtype=np.float64)
    planner.scene_bounds_max = np.asarray([0.8, 0.5, 0.5], dtype=np.float64)
    planner.retreat_face_axis = 0
    planner.retreat_face_value = 0.4
    planner.retreat_outward_sign = -1.0
    planner.separator_planes = [(2, 0.0)]

    # Both tool0 origins remain below the separator, but the top-mounted
    # camera crosses it.  The endpoint is also 0.10 m in front of the opening,
    # within the deliberately bounded 0.15 m approach shell.
    path = planner.plan(
        np.asarray([0.30, 0.0, -0.12, 0, 0, 0], dtype=np.float64),
        np.asarray([0.30, 0.0, -0.04, 0, 0, 0], dtype=np.float64),
    )

    assert path.shape == (3, 6)
    assert np.allclose(path[1], planner.anchors[0])
    assert planner.last_debug["crosses_scene_separator"] is True
    assert planner.last_debug["routing_reason"] == "scene_separator"


def test_forced_startup_route_uses_anchor_without_scene_separator():
    planner = BudgetedFieldAnchorPlanner(
        _FieldPlanner(),
        _Kinematics(),
        BudgetedAnchorConfig(anchor_budget=1, learned_min_speed=0.2),
    )
    planner.anchors = np.asarray([[0.2, 0.0, 0.0, 0, 0, 0]], dtype=np.float64)
    path = planner.plan(
        np.zeros((6,), dtype=np.float64),
        np.asarray([0.4, 0.0, 0.0, 0, 0, 0], dtype=np.float64),
        force_anchor=True,
    )
    assert path.shape == (3, 6)
    assert np.allclose(path[1], planner.anchors[0])
    assert planner.last_debug["forced_anchor"] is True
    assert planner.last_debug["routing_reason"] == "startup_anchor"
