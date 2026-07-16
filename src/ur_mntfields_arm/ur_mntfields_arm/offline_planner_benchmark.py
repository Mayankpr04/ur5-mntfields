from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.budgeted_anchor_planner import BudgetedAnchorConfig, BudgetedFieldAnchorPlanner
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker, wrist_camera_collision_spheres
from ur_mntfields_arm.planner import ArmFieldPlanner, JointSpaceRRTConnectPlanner
from ur_mntfields_arm.ur5_kinematics import UR5Kinematics
from ur_mntfields_arm.view_field_trajectories_3d import _edge_clearance_details, _plan_length
from ur_mntfields_arm.voxel_map import SparseVoxelMap


# Six fixed, reproducible camera viewpoints: two per included cabinet shelf.
# Values are generated from opening-facing camera poses and deliberately omit
# the bottom compartment, which is outside the training/evaluation ROI.
GOALS = {
    "lower_center": np.array([-0.000276, -0.800665, 1.707345, -4.048268, -1.570518, 0.0]),
    "lower_left": np.array([-0.547524, -0.762365, 1.622610, -4.001831, -1.023270, -0.000001]),
    "middle_left": np.array([-0.547524, -1.149088, 1.025320, -3.017824, -1.023273, -0.000001]),
    "middle_center": np.array([-0.000279, -1.217074, 1.121319, -3.045837, -1.570517, 0.0]),
    "top_center": np.array([-0.002971, -1.107264, 0.001139, -2.043682, -1.564929, 0.000037]),
    "top_left": np.array([-0.765910, -1.068571, -0.000474, -2.085418, -0.826350, 0.008825]),
}

CASES = (
    ("lower_within", "lower_center", "lower_left", "within"),
    ("lower_to_middle", "lower_left", "middle_left", "inter"),
    ("middle_within", "middle_left", "middle_center", "within"),
    ("middle_to_top", "middle_center", "top_center", "inter"),
    ("top_within", "top_center", "top_left", "within"),
    ("top_to_lower", "top_left", "lower_center", "inter"),
)

# Joint configurations measured with joint_teleop. The first sequence follows
# goal1 -> goal2 -> goal5 as requested and alternates within/inter-shelf legs.
# The second sequence stresses cross-shelf routing with the same six goals.
TELEOP_GOALS = {
    "goal1": np.array([0.40327, -0.55131, 1.26900, -3.85961, -1.57000, 0.0]),
    "goal2": np.array([-0.60558, -0.76626, 1.46891, -3.82272, -1.56999, 0.0]),
    "goal3": np.array([-0.26256, -1.21705, 0.11368, -2.11169, -1.56999, 0.0]),
    "goal4": np.array([-0.75557, -1.21983, 0.11368, -2.06169, -1.56999, 0.0]),
    "goal5": np.array([-0.75557, -1.01123, 0.75179, -2.44670, -1.56999, 0.0]),
    "goal6": np.array([0.51738, -0.95577, 0.75179, -2.62537, -1.56999, 0.0]),
}

# Minimally adjusted safe set. Goals 2-6 already have more than 50 mm
# endpoint clearance. Goal 1 is moved only 0.028 rad in joint space (20.8 mm
# at tool0), raising its analytic cabinet clearance from 12.6 to 30.5 mm.
SAFE_TELEOP_GOALS = {
    **TELEOP_GOALS,
    "goal1": np.array([0.407674, -0.573785, 1.257008, -3.851825, -1.572289, 0.006915]),
}

TELEOP_CASES = (
    ("route_a_01_g1_to_g2", "goal1", "goal2", "within"),
    ("route_a_02_g2_to_g5", "goal2", "goal5", "inter"),
    ("route_a_03_g5_to_g6", "goal5", "goal6", "within"),
    ("route_a_04_g6_to_g3", "goal6", "goal3", "inter"),
    ("route_a_05_g3_to_g4", "goal3", "goal4", "within"),
    ("route_a_06_g4_to_g1", "goal4", "goal1", "inter"),
    ("route_b_01_g1_to_g5", "goal1", "goal5", "inter"),
    ("route_b_02_g5_to_g3", "goal5", "goal3", "inter"),
    ("route_b_03_g3_to_g2", "goal3", "goal2", "inter"),
    ("route_b_04_g2_to_g6", "goal2", "goal6", "inter"),
    ("route_b_05_g6_to_g4", "goal6", "goal4", "inter"),
    ("route_b_06_g4_to_g1", "goal4", "goal1", "inter"),
)

STARTUP_Q = np.array([0.0, -2.74850, 1.50004, -1.71994, -1.57000, 0.03334])
TELEOP_SEQUENCE_CASES = (
    ("sequence_01_start_to_g1", "startup", "goal1", "inter"),
    ("sequence_02_g1_to_g2", "goal1", "goal2", "within"),
    ("sequence_03_g2_to_g3", "goal2", "goal3", "inter"),
    ("sequence_04_g3_to_g4", "goal3", "goal4", "within"),
    ("sequence_05_g4_to_g5", "goal4", "goal5", "inter"),
    ("sequence_06_g5_to_g6", "goal5", "goal6", "within"),
)


def _load_boxes(
    config_path: Path,
    base_world: np.ndarray,
    *,
    include_scene_boxes: bool = True,
) -> tuple[np.ndarray, int]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = payload["test_trained_field"]["ros__parameters"]
    rows: list[np.ndarray] = []
    support_count = 0
    for key in ("scene_boxes", "support_boxes"):
        if key == "scene_boxes" and not include_scene_boxes:
            continue
        entries = params.get(key, [])
        for entry in entries:
            values = entry.split(",") if isinstance(entry, str) else entry
            row = np.asarray([float(value) for value in values], dtype=np.float64)
            if row.size != 6:
                continue
            row[:3] -= base_world
            rows.append(row)
        if key == "support_boxes":
            support_count = len(entries)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6), int(support_count)


def _anchor_rows(training_root: Path, goals: dict[str, np.ndarray]) -> np.ndarray:
    samples = sorted((training_root / "samples").glob("step_*.npz"))
    if not samples:
        return np.vstack(tuple(goals.values()))
    with np.load(samples[-1]) as payload:
        frame = np.asarray(payload["frame_data"], dtype=np.float64)
    rows = frame[:, :12].reshape(-1, 6) if frame.ndim == 2 and frame.shape[1] >= 12 else np.zeros((0, 6))
    return np.vstack((rows, np.vstack(tuple(goals.values()))))


def _join(parts: list[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(part, dtype=np.float32).reshape(-1, 6) for part in parts if len(part)]
    if not valid:
        return np.zeros((0, 6), dtype=np.float32)
    return np.vstack([part if index == 0 else part[1:] for index, part in enumerate(valid)])


def _timed_route(plan_leg, q_start: np.ndarray, q_goal: np.ndarray, anchor: np.ndarray | None):
    started = time.perf_counter()
    targets = [q_goal] if anchor is None else [anchor, q_goal]
    parts: list[np.ndarray] = []
    statuses: list[str] = []
    current = q_start
    for target in targets:
        path, status = plan_leg(current, target)
        statuses.append(status)
        if np.asarray(path).ndim != 2 or len(path) < 2:
            return np.zeros((0, 6), dtype=np.float32), statuses, (time.perf_counter() - started) * 1.0e3
        parts.append(np.asarray(path, dtype=np.float32))
        current = target
    return _join(parts), statuses, (time.perf_counter() - started) * 1.0e3


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for planner_name in ("field", "field_collision", "rrt_connect"):
        selected = [row for row in rows if row["planner"] == planner_name]
        times = np.asarray([row["generation_ms"] for row in selected], dtype=np.float64)
        planned_rows = [row for row in selected if bool(row["planned"])]
        output.append(
            {
                "planner": planner_name,
                "runs": len(selected),
                "planned": int(sum(bool(row["planned"]) for row in selected)),
                "collision_free": int(sum(bool(row["collision_free"]) for row in selected)),
                "safe": int(sum(bool(row["geometrically_safe"]) for row in selected)),
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "max_ms": float(np.max(times)),
                "mean_path_length_rad": float(np.mean([row["path_length_rad"] for row in planned_rows])),
                "mean_joint_curvature": float(np.mean([row["joint_curvature"] for row in planned_rows])),
                "mean_max_turn_deg": float(np.mean([row["max_turn_deg"] for row in planned_rows])),
            }
        )
    return output


def _path_smoothness(path: np.ndarray) -> tuple[float, float]:
    q = np.asarray(path, dtype=np.float64)
    if q.ndim != 2 or len(q) < 3:
        return 0.0, 0.0
    edges = np.diff(q, axis=0)
    curvature = float(np.sum(np.linalg.norm(np.diff(edges, axis=0), axis=1)))
    left = edges[:-1]
    right = edges[1:]
    denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1.0e-12)
    turns = np.degrees(np.arccos(np.clip(np.sum(left * right, axis=1) / denom, -1.0, 1.0)))
    return curvature, float(np.max(turns)) if len(turns) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline six-goal field/field+collision/RRTConnect benchmark.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("src/ur5_sim_training_factorized_v6_balanced_shell/model/weights_final.pt"),
    )
    parser.add_argument(
        "--scene-config", type=Path, default=Path("src/ur_mntfields_arm_sim/config/sim_scene.yaml")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-speed", type=float, default=0.15)
    parser.add_argument("--collision-margin", type=float, default=0.02)
    parser.add_argument(
        "--routing", choices=("direct", "anchor"), default="direct",
        help="Directly compare planner modes, or route declared inter-shelf legs through one anchor.",
    )
    parser.add_argument(
        "--voxel-map", type=Path, default=None,
        help="Accumulated voxel-map artifact; defaults to voxel_map_final.npz beside the checkpoint.",
    )
    parser.add_argument(
        "--goal-set", choices=("shelf6", "teleop6", "teleop6_safe"), default="shelf6"
    )
    parser.add_argument(
        "--case-set", choices=("sequence6", "stress12"), default="stress12",
        help="Run the live startup-to-goal1-to-goal6 sequence or two stress-test route cycles.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/offline_planner_benchmark_6goals.json"))
    parser.add_argument("--base-world", type=float, nargs=3, default=(0.15, 0.35, 0.50))
    args = parser.parse_args()

    goals = dict({
        "shelf6": GOALS,
        "teleop6": TELEOP_GOALS,
        "teleop6_safe": SAFE_TELEOP_GOALS,
    }[args.goal_set])
    if args.case_set == "sequence6" and args.goal_set in ("teleop6", "teleop6_safe"):
        goals["startup"] = STARTUP_Q.copy()
        cases = TELEOP_SEQUENCE_CASES
    else:
        cases = CASES if args.goal_set == "shelf6" else TELEOP_CASES

    checkpoint = args.checkpoint.expanduser().resolve()
    training_root = checkpoint.parent.parent
    kinematics = UR5Kinematics()
    # ROI scene boxes are sampling/mapping bounds, never solid obstacles. Use
    # the exact accumulated occupancy artifact plus only configured support
    # geometry, matching online training and certification.
    support_boxes, support_count = _load_boxes(
        args.scene_config.expanduser().resolve(),
        np.asarray(args.base_world),
        include_scene_boxes=False,
    )
    voxel_map_path = (
        args.voxel_map.expanduser().resolve()
        if args.voxel_map is not None
        else checkpoint.parent / "voxel_map_final.npz"
    )
    voxel_map = SparseVoxelMap.load(voxel_map_path)
    camera_in_tool = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.08], [0.0, 0.0, 1.0, 0.02], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    checker = UR5PointCloudCollisionChecker(
        kinematics,
        voxel_map.occupied_points(),
        box_obstacles=support_boxes,
        support_box_count=support_count,
        attached_spheres_local=wrist_camera_collision_spheres(camera_in_tool),
    )
    model = ArmFieldModel(model_dir=str(checkpoint.parent))
    model.load_checkpoint(checkpoint)
    field_planner = ArmFieldPlanner(model, kinematics)
    anchor_planner = BudgetedFieldAnchorPlanner(
        field_planner,
        kinematics,
        BudgetedAnchorConfig(learned_min_speed=float(args.min_speed)),
        camera_in_tool=camera_in_tool,
    )
    selection_ms = 0.0
    anchor = None
    if args.routing == "anchor":
        selection_started = time.perf_counter()
        anchors = anchor_planner.select_anchors(checker, _anchor_rows(training_root, goals))
        selection_ms = (time.perf_counter() - selection_started) * 1.0e3
        if len(anchors) != 1:
            raise RuntimeError(f"Expected one opening anchor, got {len(anchors)}: {anchor_planner.last_debug}")
        anchor = anchors[0]

    # Warm model inference so timings exclude one-time framework setup.
    warmup_names = list(goals)
    field_planner.learned_edge_min_speed(
        goals[warmup_names[0]], goals[warmup_names[1]], goals[warmup_names[1]]
    )
    raw: list[dict[str, object]] = []
    for repeat in range(max(1, int(args.repeats))):
        rrt = JointSpaceRRTConnectPlanner(kinematics, rng=np.random.default_rng(1000 + repeat))
        for case_name, start_name, goal_name, transition in cases:
            q_start, q_goal = goals[start_name], goals[goal_name]
            detected_separator = bool(anchor_planner._crosses_scene_separator(q_start, q_goal))
            # This is an intentionally labelled shelf benchmark. Some high
            # shelf camera poses place tool0 just outside the opening even
            # though the camera itself is in the compartment, so route the
            # declared inter-shelf cases through the shared anchor explicitly.
            use_anchor = args.routing == "anchor" and transition == "inter"

            def field_leg(qa, qb):
                path = field_planner.plan(
                    qa, qb, 0.03, 120, mode="forward", allow_direct_edge=True,
                    min_predicted_speed=float(args.min_speed), edge_check_step_rad=0.04,
                )
                return path, str(field_planner.last_debug.get("status", ""))

            def collision_leg(qa, qb):
                path = field_planner.plan_collision_aware(
                    checker, qa, qb, 0.03, 120, clearance_margin_m=float(args.collision_margin),
                    max_local_candidates=32, allow_direct_edge=True, shortcut_path=True, mode="forward",
                )
                return path, str(field_planner.last_debug.get("status", ""))

            def rrt_leg(qa, qb):
                path = rrt.plan(
                    checker, qa, qb, step_size_q=0.20, max_iters=4000, goal_bias=0.20,
                    clearance_margin_m=float(args.collision_margin), edge_check_step_rad=0.04,
                )
                return path, str(rrt.last_debug.get("status", ""))

            for planner_name, callback in (
                ("field", field_leg), ("field_collision", collision_leg), ("rrt_connect", rrt_leg)
            ):
                path, statuses, generation_ms = _timed_route(
                    callback, q_start, q_goal, anchor if use_anchor else None
                )
                details = _edge_clearance_details(checker, path, 0.02)
                min_clearance = float(details["min_clearance"]) if len(path) else -1.0
                planned = bool(len(path) >= 2)
                collision_free = bool(planned and min_clearance > 1.0e-6)
                safe = bool(planned and min_clearance >= float(args.collision_margin))
                joint_curvature, max_turn_deg = _path_smoothness(path)
                raw.append(
                    {
                        "repeat": repeat,
                        "case": case_name,
                        "transition": transition,
                        "start": start_name,
                        "goal": goal_name,
                        "planner": planner_name,
                        "used_anchor": use_anchor,
                        "separator_detected_from_tool": detected_separator,
                        "planned": planned,
                        "collision_free": collision_free,
                        "geometrically_safe": safe,
                        "generation_ms": float(generation_ms),
                        "min_clearance_m": min_clearance,
                        "waypoints": int(len(path)),
                        "path_length_rad": float(_plan_length(path)),
                        "joint_curvature": float(joint_curvature),
                        "max_turn_deg": float(max_turn_deg),
                        "statuses": statuses,
                        "path_q": np.asarray(path, dtype=np.float64).tolist(),
                    }
                )
                print(
                    f"repeat={repeat} case={case_name} planner={planner_name} anchor={use_anchor} "
                    f"ms={generation_ms:.1f} planned={planned} safe={safe} clearance={min_clearance:.4f}"
                )

    goal_payload = {
        name: {
            "q": q.tolist(),
            "tool_xyz_base": kinematics.fk(q)[:3, 3].tolist(),
            "tool_xyz_world": (kinematics.fk(q)[:3, 3] + np.asarray(args.base_world)).tolist(),
        }
        for name, q in goals.items()
    }
    if anchor is not None:
        goal_payload["anchor"] = {
            "q": anchor.tolist(),
            "tool_xyz_base": kinematics.fk(anchor)[:3, 3].tolist(),
            "tool_xyz_world": (kinematics.fk(anchor)[:3, 3] + np.asarray(args.base_world)).tolist(),
        }
    result = {
        "checkpoint": str(checkpoint),
        "voxel_map": str(voxel_map_path),
        "scene_signature": voxel_map.scene_signature(),
        "goal_set": args.goal_set,
        "case_set": args.case_set,
        "routing": args.routing,
        "min_speed": float(args.min_speed),
        "collision_margin_m": float(args.collision_margin),
        "repeats": int(args.repeats),
        "anchor_selection_ms": float(selection_ms),
        "anchor_debug": anchor_planner.last_debug,
        "goals": goal_payload,
        "cases": [list(case) for case in cases],
        "summary": _summary(raw),
        "raw": raw,
    }
    args.output.expanduser().resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nplanner         runs planned collision_free margin_safe mean_ms median_ms p95_ms max_ms")
    for row in result["summary"]:
        print(
            f"{row['planner']:<16} {row['runs']:>4} {row['planned']:>7} "
            f"{row['collision_free']:>14} {row['safe']:>11} "
            f"{row['mean_ms']:>7.1f} {row['median_ms']:>9.1f} {row['p95_ms']:>6.1f} {row['max_ms']:>6.1f}"
        )
    print(f"anchor_selection_ms={selection_ms:.1f} output={args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
