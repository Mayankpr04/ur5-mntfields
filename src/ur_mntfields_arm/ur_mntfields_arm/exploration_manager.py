from __future__ import annotations

import json
from pathlib import Path as FsPath
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker, make_ur5_collision_checker
from ur_mntfields_arm.cspace_sampling import (
    make_cspace_pair_rows_from_q_pairs,
    sample_cspace_training_batch,
    sample_path_centered_training_batch,
)
from ur_mntfields_arm.frontier_bank import FrontierBank
from ur_mntfields_arm.goal_selector import ViewGoalSelector
from ur_mntfields_arm.planner import ArmFieldPlanner, JointSpaceRRTConnectPlanner
from ur_mntfields_arm.ur5_kinematics import JOINT_NAMES, UR5Kinematics, ViewGoal, _transform, look_at_rotation
from ur_mntfields_arm.voxel_map import SparseVoxelMap


def _transform_to_matrix(tf_msg) -> np.ndarray:
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    x, y, z, w = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = [t.x, t.y, t.z]
    return out


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class ArmMNTFieldsExplorer(Node):
    def __init__(self):
        super().__init__("arm_mntfields_explorer")
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("fallback_camera_width", 424)
        self.declare_parameter("fallback_camera_height", 240)
        self.declare_parameter("fallback_fx", 212.0)
        self.declare_parameter("fallback_fy", 212.0)
        self.declare_parameter("fallback_cx", 212.0)
        self.declare_parameter("fallback_cy", 120.0)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("ur_type", "ur5")
        self.declare_parameter("output_dir", "/tmp/ur_mntfields_arm")
        self.declare_parameter("camera_in_tool", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("voxel_size_m", 0.05)
        self.declare_parameter("frontier_match_radius_m", 0.18)
        self.declare_parameter("frontier_visit_radius_m", 0.20)
        self.declare_parameter("max_frontier_failures", 4)
        self.declare_parameter("raycast_stride_px", 8)
        self.declare_parameter("depth_min_m", 0.20)
        self.declare_parameter("depth_max_m", 2.00)
        self.declare_parameter("enable_robot_self_filter", True)
        self.declare_parameter("robot_self_filter_padding_m", 0.04)
        self.declare_parameter("robot_self_filter_tool_radius_m", 0.08)
        self.declare_parameter("robot_self_filter_mount_radius_m", 0.07)
        self.declare_parameter("clearance_backend", "original")
        # Sampling and view ranking may use the fast voxel SDF while execution
        # continues to use the exact point-cloud checker.
        self.declare_parameter("sampling_clearance_backend", "")
        self.declare_parameter("nbv_clearance_backend", "")
        self.declare_parameter("sdf_voxel_size_m", 0.04)
        self.declare_parameter("sdf_padding_m", 0.75)
        self.declare_parameter("sdf_max_cells", 4000000)
        self.declare_parameter("sample_pairs_per_step", 2000)
        self.declare_parameter("samples_per_ik_seed", 64) 
        self.declare_parameter("sampling_mode", "joint_local_6d")
        self.declare_parameter("sampling_proposal_batch_size", 256)
        self.declare_parameter("clearance_margin_m", 0.16)
        self.declare_parameter("clearance_offset_m", 0.0)
        self.declare_parameter("clearance_label_floor", 0.0)
        self.declare_parameter("clearance_label_power", 1.0)
        self.declare_parameter("anchor_seed_probability", 0.35)
        self.declare_parameter("roi_sampling_seed_fraction", 0.25)
        self.declare_parameter("use_field_eval_anchors_for_sampling", False)
        self.declare_parameter("use_configured_training_anchors", True)
        self.declare_parameter("training_anchor_joint_goals", [0.0] * 6)
        self.declare_parameter("train_epochs_per_step", 5)
        self.declare_parameter("train_every_n_frames", 1)
        self.declare_parameter("replay_buffer_capacity", 100000)
        self.declare_parameter("train_minibatch_size", 2000)
        self.declare_parameter("train_gradient_accumulation_steps", 1)
        self.declare_parameter("train_replay_ratio", 0.75)
        self.declare_parameter("hard_example_train_ratio", 0.0)
        self.declare_parameter("replay_recombine_pairs_per_step", 1000)
        self.declare_parameter("adaptive_training_enabled", True)
        self.declare_parameter("field_ready_sample_pairs_per_step", 1500)
        self.declare_parameter("field_ready_train_epochs_per_step", 8)
        self.declare_parameter("field_ready_train_every_n_frames", 6)
        self.declare_parameter("field_ready_replay_recombine_pairs_per_step", 500)
        self.declare_parameter("field_ready_hard_failed_pairs_per_step", 256)
        self.declare_parameter("path_centered_pair_fraction", 0.35)
        self.declare_parameter("near_boundary_pair_fraction", 0.40)
        self.declare_parameter("path_anchor_stride", 4)
        self.declare_parameter("path_anchor_buffer_limit", 2048)
        self.declare_parameter("hard_failed_pairs_per_step", 512)
        self.declare_parameter("hard_failed_anchor_buffer_limit", 2048)
        self.declare_parameter("hard_failed_clearance_window_m", 0.08)
        self.declare_parameter("step_size_q", 0.04)
        self.declare_parameter("rollout_max_steps", 160)
        self.declare_parameter("execution_planner", "field_then_rrt")
        self.declare_parameter("collision_aware_field_rollout", True)
        self.declare_parameter("field_local_rollout_candidates", 32)
        self.declare_parameter("rrt_max_iters", 4000)
        self.declare_parameter("rrt_goal_bias", 0.20)
        self.declare_parameter("enable_trajectory_publish", False)
        self.declare_parameter("trajectory_topic", "/ur_mntfields_arm/joint_trajectory")
        self.declare_parameter("save_every_n_steps", 1)
        self.declare_parameter("checkpoint_every_epochs", 20)
        self.declare_parameter("enable_field_diagnostics", True)
        self.declare_parameter("field_diagnostics_max_rows", 4096)
        self.declare_parameter("field_diagnostics_grid_size", 61)
        self.declare_parameter("field_diagnostics_every_n_train_updates", 5)
        self.declare_parameter("field_diagnostics_save_joint_slices", False)
        # Paper Eq. 12 is the default objective.  The TD, direct-speed and
        # normal losses are experimental auxiliaries and remain opt-in.
        self.declare_parameter("field_td_loss_weight", 0.0)
        self.declare_parameter("field_speed_loss_weight", 1.0)
        self.declare_parameter("field_log_speed_loss_weight", 0.0)
        self.declare_parameter("field_direct_speed_loss_weight", 0.0)
        self.declare_parameter("field_normal_loss_weight", 0.0)
        self.declare_parameter("field_normal_cos_loss_weight", 0.0)
        self.declare_parameter("field_near_obstacle_loss_weight", 0.0)
        self.declare_parameter("field_low_speed_threshold", 0.20)
        self.declare_parameter("field_low_speed_pred_max", 0.30)
        self.declare_parameter("field_low_speed_penalty_weight", 0.0)
        self.declare_parameter("field_effective_speed_floor", 0.05)
        self.declare_parameter("field_diag_gate_enabled", True)
        self.declare_parameter("field_diag_gate_min_replay_pairs", 12000)
        self.declare_parameter("field_diag_max_low_overpred_frac", 0.45)
        self.declare_parameter("field_diag_min_speed_corr", 0.35)
        self.declare_parameter("field_diag_min_near_far_gap", 0.20)
        self.declare_parameter("field_diag_min_normal_cos_near", 0.20)
        self.declare_parameter("field_false_free_audit_enabled", True)
        self.declare_parameter("field_false_free_audit_samples", 1024)
        self.declare_parameter("field_false_free_audit_goals_per_state", 4)
        self.declare_parameter("field_false_free_target_speed_max", 0.20)
        self.declare_parameter("field_false_free_pred_speed_min", 0.20)
        self.declare_parameter("field_false_free_max_rate", 0.05)
        self.declare_parameter("field_false_free_min_low_states", 32)
        self.declare_parameter("debug_point_cloud_topic", "/debug/points/world")
        self.declare_parameter("planned_path_topic", "/ur_mntfields_arm/planned_path")
        self.declare_parameter("replan_while_executing", False)
        self.declare_parameter("trajectory_busy_margin_s", 1.0)
        self.declare_parameter("trajectory_max_joint_speed", 0.25)
        self.declare_parameter("trajectory_min_segment_dt", 0.35)
        self.declare_parameter("trajectory_collision_margin_m", 0.01)
        self.declare_parameter("trajectory_waypoint_stride", 3)
        self.declare_parameter("trajectory_smoothing_window", 5)
        self.declare_parameter("execute_prefix_waypoints", 12)
        self.declare_parameter("execute_prefix_min_duration_s", 4.0)
        self.declare_parameter("enable_frontier_roi_filter", True)
        self.declare_parameter("enable_object_roi_nbv", True)
        self.declare_parameter("object_roi_nbv_first", True)
        self.declare_parameter("rank_frontiers_when_roi_available", False)
        self.declare_parameter("fast_roi_nbv", True)
        self.declare_parameter("roi_nbv_max_pose_candidates", 18)
        self.declare_parameter("enable_view_self_occlusion_filter", True)
        self.declare_parameter("min_view_self_occlusion_free_fraction", 0.98)
        self.declare_parameter("view_self_occlusion_padding_m", 0.03)
        self.declare_parameter("view_self_occlusion_ignore_near_origin_m", 0.06)
        self.declare_parameter("frontier_roi_init_min_points", 300)
        self.declare_parameter("frontier_roi_init_min_step", 1)
        self.declare_parameter("frontier_roi_padding_xyz", [0.12, 0.12, 0.12])
        self.declare_parameter("scene_boxes", [""])
        self.declare_parameter("scene_boxes_topic", "/scene_boxes")
        self.declare_parameter("scene_boxes_frame", "")
        self.declare_parameter("support_boxes", [""])
        self.declare_parameter("support_boxes_frame", "")
        self.declare_parameter("support_point_ignore_padding_m", 0.15)
        self.declare_parameter("cabinet_bbox_padding_xyz", [0.04, 0.04, 0.04])
        self.declare_parameter("roi_clip_frame", "")
        self.declare_parameter("roi_clip_min_xyz", [-1.0e9, -1.0e9, -1.0e9])
        self.declare_parameter("roi_clip_max_xyz", [1.0e9, 1.0e9, 1.0e9])
        self.declare_parameter("finish_when_frontiers_exhausted", True)
        self.declare_parameter("finish_when_roi_covered", True)
        self.declare_parameter("finish_when_field_eval_passes", True)
        self.declare_parameter("roi_coverage_threshold", 0.82)
        self.declare_parameter("roi_unknown_stop_voxels", 40)
        self.declare_parameter("field_eval_points_frame", "world")
        self.declare_parameter("field_eval_points_xyz", [0.60, 0.35, 0.68, 0.60, 0.35, 1.12])
        self.declare_parameter("field_eval_joint_goals", [0.0] * 6)
        self.declare_parameter("field_eval_goal_clearance_min_m", 0.05)
        self.declare_parameter("field_eval_goal_tool_forward_alignment_min", 0.70)
        self.declare_parameter("field_eval_success_ratio_threshold", 1.0)
        self.declare_parameter("field_eval_every_n_steps", 1)
        self.declare_parameter("field_eval_min_train_steps", 1)
        self.declare_parameter("field_eval_min_replay_pairs", 30000)
        self.declare_parameter("field_eval_use_startup_pose", True)
        self.declare_parameter("field_eval_collision_aware_rollout", False)
        self.declare_parameter("training_wall_time_limit_s", 600.0)
        self.declare_parameter("goal_tool_orientation_yaw_offsets_rad", [0.0, 1.5708, -1.5708, 3.1416])
        self.declare_parameter("goal_tool_orientation_pitch_offsets_rad", [0.0, 0.7854, -0.7854, 1.5708, -1.5708])
        self.declare_parameter("goal_tool_orientation_roll_offsets_rad", [0.0, 1.5708, -1.5708, 3.1416])
        self.declare_parameter("frontier_completion_patience_steps", 10)
        self.declare_parameter("strict_train_move_cycle", True)
        self.declare_parameter("train_during_motion", True)
        self.declare_parameter("max_joint_state_age_s", 1.5)
        self.declare_parameter("min_goal_joint_delta_rad", 0.08)
        self.declare_parameter("min_camera_goal_delta_m", 0.10)
        self.declare_parameter("frontier_reselect_cooldown_steps", 8)
        self.declare_parameter("viewpoint_cooldown_steps", 12)
        self.declare_parameter("viewpoint_cooldown_radius_m", 0.18)
        self.declare_parameter("min_frontier_visibility_score", 0.05)
        self.declare_parameter("min_roi_coverage_ratio", 0.08)
        self.declare_parameter("min_view_alignment", 0.72)
        self.declare_parameter("min_actual_view_alignment", 0.85)
        self.declare_parameter("target_context_radius_m", 0.35)
        self.declare_parameter("frontier_pose_candidates_per_frontier", 15)
        self.declare_parameter("frontier_fallback_max_frontiers", 3)
        self.declare_parameter("camera_yaw_offsets_deg", [-12.0, 0.0, 12.0])
        self.declare_parameter("camera_pitch_offsets_deg", [-10.0, 0.0, 10.0])
        self.declare_parameter("camera_roll_offsets_deg", [0.0])
        self.declare_parameter("enable_bootstrap_recovery", True)
        self.declare_parameter("bootstrap_recovery_radius_m", 0.12)
        self.declare_parameter("bootstrap_recovery_lateral_m", 0.10)
        self.declare_parameter("bootstrap_recovery_vertical_m", 0.08)
        self.declare_parameter("enable_startup_pose_gate", True)
        self.declare_parameter("inspection_positions", [0.35, -1.10, 1.10, -1.55, -1.57, 0.0])
        self.declare_parameter("pose_reached_tolerance_rad", 0.15)
        self.declare_parameter("startup_wait_log_interval_s", 2.0)
        self.declare_parameter("startup_data_collection_delay_s", 0.0)
        self.declare_parameter("visualization_frame", "")
        self.declare_parameter("retrain_same_pose_joint_threshold_rad", 0.05)
        self.declare_parameter("local_frontier_recent_steps", 8)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.fallback_camera_width = int(self.get_parameter("fallback_camera_width").value)
        self.fallback_camera_height = int(self.get_parameter("fallback_camera_height").value)
        self.fallback_fx = float(self.get_parameter("fallback_fx").value)
        self.fallback_fy = float(self.get_parameter("fallback_fy").value)
        self.fallback_cx = float(self.get_parameter("fallback_cx").value)
        self.fallback_cy = float(self.get_parameter("fallback_cy").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.output_dir = FsPath(str(self.get_parameter("output_dir").value)).expanduser()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot create output_dir={self.output_dir}. Use a writable path such as "
                f"/tmp/ur5_train_new or /home/mayank/ur_ws/ur5_train_new."
            ) from exc
        self.images_dir = self.output_dir / "Images"
        self.pcd_dir = self.output_dir / "PCD"
        self.model_artifacts_dir = self.output_dir / "model"
        self.samples_dir = self.output_dir / "samples"
        for out_dir in (self.images_dir, self.pcd_dir, self.model_artifacts_dir, self.samples_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_in_tool = np.asarray(self.get_parameter("camera_in_tool").value, dtype=np.float64).reshape(4, 4)
        self.raycast_stride_px = int(self.get_parameter("raycast_stride_px").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.enable_robot_self_filter = bool(self.get_parameter("enable_robot_self_filter").value)
        self.robot_self_filter_padding_m = float(max(0.0, float(self.get_parameter("robot_self_filter_padding_m").value)))
        self.robot_self_filter_tool_radius_m = float(
            max(0.0, float(self.get_parameter("robot_self_filter_tool_radius_m").value))
        )
        self.robot_self_filter_mount_radius_m = float(
            max(0.0, float(self.get_parameter("robot_self_filter_mount_radius_m").value))
        )
        self.clearance_backend = str(self.get_parameter("clearance_backend").value).strip().lower()
        configured_sampling_backend = str(
            self.get_parameter("sampling_clearance_backend").value
        ).strip().lower()
        self.sampling_clearance_backend = configured_sampling_backend or self.clearance_backend
        configured_nbv_backend = str(self.get_parameter("nbv_clearance_backend").value).strip().lower()
        self.nbv_clearance_backend = configured_nbv_backend or self.clearance_backend
        self.sdf_voxel_size_m = float(max(1.0e-3, float(self.get_parameter("sdf_voxel_size_m").value)))
        self.sdf_padding_m = float(max(0.0, float(self.get_parameter("sdf_padding_m").value)))
        self.sdf_max_cells = max(10_000, int(self.get_parameter("sdf_max_cells").value))
        self.sample_pairs_per_step = int(self.get_parameter("sample_pairs_per_step").value)
        self.samples_per_ik_seed = int(self.get_parameter("samples_per_ik_seed").value)
        self.sampling_mode = str(self.get_parameter("sampling_mode").value)
        self.sampling_proposal_batch_size = int(self.get_parameter("sampling_proposal_batch_size").value)
        self.clearance_margin_m = float(self.get_parameter("clearance_margin_m").value)
        configured_offset = float(self.get_parameter("clearance_offset_m").value)
        self.clearance_offset_m = (
            configured_offset
            if configured_offset > 0.0
            else max(1.0e-6, self.clearance_margin_m / 10.0)
        )
        self.clearance_label_floor = float(np.clip(float(self.get_parameter("clearance_label_floor").value), 0.0, 1.0))
        self.clearance_label_power = float(max(1.0e-6, float(self.get_parameter("clearance_label_power").value)))
        self.anchor_seed_probability = float(np.clip(float(self.get_parameter("anchor_seed_probability").value), 0.0, 1.0))
        self.roi_sampling_seed_fraction = float(
            np.clip(float(self.get_parameter("roi_sampling_seed_fraction").value), 0.0, 1.0)
        )
        self.use_field_eval_anchors_for_sampling = bool(self.get_parameter("use_field_eval_anchors_for_sampling").value)
        training_anchor_values = [float(v) for v in self.get_parameter("training_anchor_joint_goals").value]
        if len(training_anchor_values) % 6 != 0:
            raise ValueError(
                "training_anchor_joint_goals must contain a multiple of 6 values, "
                f"got {len(training_anchor_values)}"
            )
        use_configured_training_anchors = bool(self.get_parameter("use_configured_training_anchors").value)
        training_anchor_goals = (
            np.asarray(training_anchor_values, dtype=np.float64).reshape(-1, 6)
            if use_configured_training_anchors
            else np.zeros((0, 6), dtype=np.float64)
        )
        anchor_waypoints = [
            np.asarray(self.get_parameter("inspection_positions").value, dtype=np.float64).reshape(6)
        ] if len(training_anchor_goals) else []
        anchor_waypoints.extend(training_anchor_goals)
        training_anchors: list[np.ndarray] = []
        for qa, qb in zip(anchor_waypoints[:-1], anchor_waypoints[1:]):
            max_delta = float(np.max(np.abs(qb - qa)))
            count = max(2, int(np.ceil(max_delta / 0.08)) + 1)
            training_anchors.extend(
                qa + alpha * (qb - qa) for alpha in np.linspace(0.0, 1.0, count)
            )
        self.training_anchor_qs = (
            np.asarray(training_anchors, dtype=np.float64)
            if training_anchors
            else np.zeros((0, 6), dtype=np.float64)
        )
        self.train_epochs_per_step = int(self.get_parameter("train_epochs_per_step").value)
        self.train_every_n_frames = int(self.get_parameter("train_every_n_frames").value)
        self.replay_buffer_capacity = int(self.get_parameter("replay_buffer_capacity").value)
        self.train_minibatch_size = int(self.get_parameter("train_minibatch_size").value)
        self.train_gradient_accumulation_steps = max(
            1, int(self.get_parameter("train_gradient_accumulation_steps").value)
        )
        self.train_replay_ratio = float(self.get_parameter("train_replay_ratio").value)
        self.hard_example_train_ratio = float(self.get_parameter("hard_example_train_ratio").value)
        self.replay_recombine_pairs_per_step = max(0, int(self.get_parameter("replay_recombine_pairs_per_step").value))
        self.adaptive_training_enabled = bool(self.get_parameter("adaptive_training_enabled").value)
        self.field_ready_sample_pairs_per_step = max(0, int(self.get_parameter("field_ready_sample_pairs_per_step").value))
        self.field_ready_train_epochs_per_step = max(1, int(self.get_parameter("field_ready_train_epochs_per_step").value))
        self.field_ready_train_every_n_frames = max(1, int(self.get_parameter("field_ready_train_every_n_frames").value))
        self.field_ready_replay_recombine_pairs_per_step = max(0, int(self.get_parameter("field_ready_replay_recombine_pairs_per_step").value))
        self.field_ready_hard_failed_pairs_per_step = max(0, int(self.get_parameter("field_ready_hard_failed_pairs_per_step").value))
        self.path_centered_pair_fraction = float(np.clip(float(self.get_parameter("path_centered_pair_fraction").value), 0.0, 1.0))
        self.near_boundary_pair_fraction = float(np.clip(float(self.get_parameter("near_boundary_pair_fraction").value), 0.0, 1.0))
        self.path_anchor_stride = max(1, int(self.get_parameter("path_anchor_stride").value))
        self.path_anchor_buffer_limit = max(64, int(self.get_parameter("path_anchor_buffer_limit").value))
        self.hard_failed_pairs_per_step = max(0, int(self.get_parameter("hard_failed_pairs_per_step").value))
        self.hard_failed_anchor_buffer_limit = max(64, int(self.get_parameter("hard_failed_anchor_buffer_limit").value))
        self.hard_failed_clearance_window_m = max(0.0, float(self.get_parameter("hard_failed_clearance_window_m").value))
        self.step_size_q = float(self.get_parameter("step_size_q").value)
        self.rollout_max_steps = int(self.get_parameter("rollout_max_steps").value)
        self.execution_planner = str(self.get_parameter("execution_planner").value).strip().lower()
        self.collision_aware_field_rollout = bool(self.get_parameter("collision_aware_field_rollout").value)
        self.field_local_rollout_candidates = max(4, int(self.get_parameter("field_local_rollout_candidates").value))
        self.rrt_max_iters = int(self.get_parameter("rrt_max_iters").value)
        self.rrt_goal_bias = float(self.get_parameter("rrt_goal_bias").value)
        self.enable_trajectory_publish = bool(self.get_parameter("enable_trajectory_publish").value)
        self.save_every_n_steps = int(self.get_parameter("save_every_n_steps").value)
        self.checkpoint_every_epochs = int(self.get_parameter("checkpoint_every_epochs").value)
        self.enable_field_diagnostics = bool(self.get_parameter("enable_field_diagnostics").value)
        self.field_diagnostics_max_rows = max(128, int(self.get_parameter("field_diagnostics_max_rows").value))
        self.field_diagnostics_grid_size = max(21, int(self.get_parameter("field_diagnostics_grid_size").value))
        self.field_diagnostics_every_n_train_updates = max(
            1, int(self.get_parameter("field_diagnostics_every_n_train_updates").value)
        )
        self.field_diagnostics_save_joint_slices = bool(
            self.get_parameter("field_diagnostics_save_joint_slices").value
        )
        self.field_td_loss_weight = float(self.get_parameter("field_td_loss_weight").value)
        self.field_speed_loss_weight = float(self.get_parameter("field_speed_loss_weight").value)
        self.field_log_speed_loss_weight = float(self.get_parameter("field_log_speed_loss_weight").value)
        self.field_direct_speed_loss_weight = float(self.get_parameter("field_direct_speed_loss_weight").value)
        self.field_normal_loss_weight = float(self.get_parameter("field_normal_loss_weight").value)
        self.field_normal_cos_loss_weight = float(self.get_parameter("field_normal_cos_loss_weight").value)
        self.field_near_obstacle_loss_weight = float(self.get_parameter("field_near_obstacle_loss_weight").value)
        self.field_low_speed_threshold = float(self.get_parameter("field_low_speed_threshold").value)
        self.field_low_speed_pred_max = float(self.get_parameter("field_low_speed_pred_max").value)
        self.field_low_speed_penalty_weight = float(self.get_parameter("field_low_speed_penalty_weight").value)
        self.field_effective_speed_floor = float(self.get_parameter("field_effective_speed_floor").value)
        self.field_diag_gate_enabled = bool(self.get_parameter("field_diag_gate_enabled").value)
        self.field_diag_gate_min_replay_pairs = max(0, int(self.get_parameter("field_diag_gate_min_replay_pairs").value))
        self.field_diag_max_low_overpred_frac = float(self.get_parameter("field_diag_max_low_overpred_frac").value)
        self.field_diag_min_speed_corr = float(self.get_parameter("field_diag_min_speed_corr").value)
        self.field_diag_min_near_far_gap = float(self.get_parameter("field_diag_min_near_far_gap").value)
        self.field_diag_min_normal_cos_near = float(self.get_parameter("field_diag_min_normal_cos_near").value)
        self.field_false_free_audit_enabled = bool(
            self.get_parameter("field_false_free_audit_enabled").value
        )
        self.field_false_free_audit_samples = max(
            128, int(self.get_parameter("field_false_free_audit_samples").value)
        )
        self.field_false_free_audit_goals_per_state = max(
            1, int(self.get_parameter("field_false_free_audit_goals_per_state").value)
        )
        self.field_false_free_target_speed_max = float(
            self.get_parameter("field_false_free_target_speed_max").value
        )
        self.field_false_free_pred_speed_min = float(
            self.get_parameter("field_false_free_pred_speed_min").value
        )
        self.field_false_free_max_rate = float(
            self.get_parameter("field_false_free_max_rate").value
        )
        self.field_false_free_min_low_states = max(
            1, int(self.get_parameter("field_false_free_min_low_states").value)
        )
        self.debug_point_cloud_topic = str(self.get_parameter("debug_point_cloud_topic").value)
        self.planned_path_topic = str(self.get_parameter("planned_path_topic").value)
        self.replan_while_executing = bool(self.get_parameter("replan_while_executing").value)
        self.trajectory_busy_margin_s = float(self.get_parameter("trajectory_busy_margin_s").value)
        self.trajectory_max_joint_speed = float(self.get_parameter("trajectory_max_joint_speed").value)
        self.trajectory_min_segment_dt = float(self.get_parameter("trajectory_min_segment_dt").value)
        self.trajectory_collision_margin_m = float(self.get_parameter("trajectory_collision_margin_m").value)
        self.trajectory_waypoint_stride = int(self.get_parameter("trajectory_waypoint_stride").value)
        self.trajectory_smoothing_window = int(self.get_parameter("trajectory_smoothing_window").value)
        self.execute_prefix_waypoints = int(self.get_parameter("execute_prefix_waypoints").value)
        self.execute_prefix_min_duration_s = float(self.get_parameter("execute_prefix_min_duration_s").value)
        self.enable_frontier_roi_filter = bool(self.get_parameter("enable_frontier_roi_filter").value)
        self.enable_object_roi_nbv = bool(self.get_parameter("enable_object_roi_nbv").value)
        self.object_roi_nbv_first = bool(self.get_parameter("object_roi_nbv_first").value)
        self.rank_frontiers_when_roi_available = bool(self.get_parameter("rank_frontiers_when_roi_available").value)
        self.fast_roi_nbv = bool(self.get_parameter("fast_roi_nbv").value)
        self.roi_nbv_max_pose_candidates = int(self.get_parameter("roi_nbv_max_pose_candidates").value)
        self.enable_view_self_occlusion_filter = bool(self.get_parameter("enable_view_self_occlusion_filter").value)
        self.min_view_self_occlusion_free_fraction = float(
            np.clip(float(self.get_parameter("min_view_self_occlusion_free_fraction").value), 0.0, 1.0)
        )
        self.view_self_occlusion_padding_m = float(
            max(0.0, float(self.get_parameter("view_self_occlusion_padding_m").value))
        )
        self.view_self_occlusion_ignore_near_origin_m = float(
            max(0.0, float(self.get_parameter("view_self_occlusion_ignore_near_origin_m").value))
        )
        self.frontier_roi_init_min_points = int(self.get_parameter("frontier_roi_init_min_points").value)
        self.frontier_roi_init_min_step = int(self.get_parameter("frontier_roi_init_min_step").value)
        self.frontier_roi_padding_xyz = np.asarray(
            self.get_parameter("frontier_roi_padding_xyz").value, dtype=np.float64
        ).reshape(3)
        self.scene_boxes = self._parse_scene_boxes(self.get_parameter("scene_boxes").value)
        self.scene_boxes_topic = str(self.get_parameter("scene_boxes_topic").value)
        scene_boxes_frame = str(self.get_parameter("scene_boxes_frame").value)
        self.scene_boxes_frame = scene_boxes_frame if scene_boxes_frame else self.base_frame
        self.support_boxes = self._parse_scene_boxes(self.get_parameter("support_boxes").value)
        support_boxes_frame = str(self.get_parameter("support_boxes_frame").value)
        self.support_boxes_frame = support_boxes_frame if support_boxes_frame else self.base_frame
        self.support_point_ignore_padding_m = float(
            max(0.0, float(self.get_parameter("support_point_ignore_padding_m").value))
        )
        self.cabinet_bbox_padding_xyz = np.asarray(
            self.get_parameter("cabinet_bbox_padding_xyz").value, dtype=np.float64
        ).reshape(3)
        roi_clip_frame = str(self.get_parameter("roi_clip_frame").value)
        self.roi_clip_frame = roi_clip_frame if roi_clip_frame else self.base_frame
        self.roi_clip_min_xyz = np.asarray(
            self.get_parameter("roi_clip_min_xyz").value, dtype=np.float64
        ).reshape(3)
        self.roi_clip_max_xyz = np.asarray(
            self.get_parameter("roi_clip_max_xyz").value, dtype=np.float64
        ).reshape(3)
        self.finish_when_frontiers_exhausted = bool(self.get_parameter("finish_when_frontiers_exhausted").value)
        self.finish_when_roi_covered = bool(self.get_parameter("finish_when_roi_covered").value)
        self.finish_when_field_eval_passes = bool(self.get_parameter("finish_when_field_eval_passes").value)
        self.roi_coverage_threshold = float(np.clip(float(self.get_parameter("roi_coverage_threshold").value), 0.0, 1.0))
        self.roi_unknown_stop_voxels = max(0, int(self.get_parameter("roi_unknown_stop_voxels").value))
        self.field_eval_points_frame = str(self.get_parameter("field_eval_points_frame").value)
        self.field_eval_points_xyz = np.asarray(
            [float(v) for v in self.get_parameter("field_eval_points_xyz").value], dtype=np.float64
        ).reshape(-1, 3)
        field_eval_joint_values = [float(v) for v in self.get_parameter("field_eval_joint_goals").value]
        if len(field_eval_joint_values) % 6 != 0:
            raise ValueError(
                f"field_eval_joint_goals must contain a multiple of 6 values, got {len(field_eval_joint_values)}"
            )
        self.field_eval_joint_goals = np.asarray(field_eval_joint_values, dtype=np.float64).reshape(-1, 6)
        self.field_eval_goal_clearance_min_m = float(max(0.0, float(self.get_parameter("field_eval_goal_clearance_min_m").value)))
        self.field_eval_goal_tool_forward_alignment_min = float(
            self.get_parameter("field_eval_goal_tool_forward_alignment_min").value
        )
        self.field_eval_success_ratio_threshold = float(np.clip(float(self.get_parameter("field_eval_success_ratio_threshold").value), 0.0, 1.0))
        self.field_eval_every_n_steps = max(1, int(self.get_parameter("field_eval_every_n_steps").value))
        self.field_eval_min_train_steps = max(1, int(self.get_parameter("field_eval_min_train_steps").value))
        self.field_eval_min_replay_pairs = max(1, int(self.get_parameter("field_eval_min_replay_pairs").value))
        self.field_eval_use_startup_pose = bool(self.get_parameter("field_eval_use_startup_pose").value)
        self.field_eval_collision_aware_rollout = bool(
            self.get_parameter("field_eval_collision_aware_rollout").value
        )
        self.training_wall_time_limit_s = max(
            0.0, float(self.get_parameter("training_wall_time_limit_s").value)
        )
        self.goal_tool_orientation_yaw_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_yaw_offsets_rad").value
        ]
        self.goal_tool_orientation_pitch_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_pitch_offsets_rad").value
        ]
        self.goal_tool_orientation_roll_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_roll_offsets_rad").value
        ]
        self.frontier_completion_patience_steps = max(
            1, int(self.get_parameter("frontier_completion_patience_steps").value)
        )
        self.strict_train_move_cycle = bool(self.get_parameter("strict_train_move_cycle").value)
        self.train_during_motion = bool(self.get_parameter("train_during_motion").value)
        self.max_joint_state_age_s = float(self.get_parameter("max_joint_state_age_s").value)
        self.min_goal_joint_delta_rad = float(self.get_parameter("min_goal_joint_delta_rad").value)
        self.min_camera_goal_delta_m = float(self.get_parameter("min_camera_goal_delta_m").value)
        self.frontier_reselect_cooldown_steps = int(self.get_parameter("frontier_reselect_cooldown_steps").value)
        self.viewpoint_cooldown_steps = max(0, int(self.get_parameter("viewpoint_cooldown_steps").value))
        self.viewpoint_cooldown_radius_m = float(max(0.0, float(self.get_parameter("viewpoint_cooldown_radius_m").value)))
        self.min_frontier_visibility_score = float(self.get_parameter("min_frontier_visibility_score").value)
        self.min_roi_coverage_ratio = float(self.get_parameter("min_roi_coverage_ratio").value)
        self.min_view_alignment = float(self.get_parameter("min_view_alignment").value)
        self.min_actual_view_alignment = float(
            np.clip(float(self.get_parameter("min_actual_view_alignment").value), -1.0, 1.0)
        )
        self.target_context_radius_m = float(self.get_parameter("target_context_radius_m").value)
        self.frontier_pose_candidates_per_frontier = max(
            3, int(self.get_parameter("frontier_pose_candidates_per_frontier").value)
        )
        self.frontier_fallback_max_frontiers = max(
            1, int(self.get_parameter("frontier_fallback_max_frontiers").value)
        )
        self.camera_yaw_offsets_deg = tuple(float(v) for v in self.get_parameter("camera_yaw_offsets_deg").value)
        self.camera_pitch_offsets_deg = tuple(float(v) for v in self.get_parameter("camera_pitch_offsets_deg").value)
        self.camera_roll_offsets_deg = tuple(float(v) for v in self.get_parameter("camera_roll_offsets_deg").value)
        self.enable_bootstrap_recovery = bool(self.get_parameter("enable_bootstrap_recovery").value)
        self.bootstrap_recovery_radius_m = float(self.get_parameter("bootstrap_recovery_radius_m").value)
        self.bootstrap_recovery_lateral_m = float(self.get_parameter("bootstrap_recovery_lateral_m").value)
        self.bootstrap_recovery_vertical_m = float(self.get_parameter("bootstrap_recovery_vertical_m").value)
        self.enable_startup_pose_gate = bool(self.get_parameter("enable_startup_pose_gate").value)
        self.startup_positions = np.asarray(self.get_parameter("inspection_positions").value, dtype=np.float64).reshape(6)
        self.startup_pose_tolerance_rad = float(self.get_parameter("pose_reached_tolerance_rad").value)
        self.startup_wait_log_interval_s = float(self.get_parameter("startup_wait_log_interval_s").value)
        self.startup_data_collection_delay_s = max(0.0, float(self.get_parameter("startup_data_collection_delay_s").value))
        visualization_frame = str(self.get_parameter("visualization_frame").value)
        self.visualization_frame = visualization_frame if visualization_frame else self.base_frame
        self.retrain_same_pose_joint_threshold_rad = float(
            self.get_parameter("retrain_same_pose_joint_threshold_rad").value
        )
        self.local_frontier_recent_steps = int(self.get_parameter("local_frontier_recent_steps").value)

        self.kinematics = UR5Kinematics(str(self.get_parameter("ur_type").value))
        self.voxel_map = SparseVoxelMap(float(self.get_parameter("voxel_size_m").value))
        self.frontier_bank = FrontierBank(
            float(self.get_parameter("frontier_match_radius_m").value),
            float(self.get_parameter("frontier_visit_radius_m").value),
            int(self.get_parameter("max_frontier_failures").value),
        )
        self.field_model = ArmFieldModel(
            str(self.output_dir / "model"),
            replay_capacity=self.replay_buffer_capacity,
            minibatch_size=self.train_minibatch_size,
            gradient_accumulation_steps=self.train_gradient_accumulation_steps,
            replay_ratio=self.train_replay_ratio,
            priority_ratio=self.hard_example_train_ratio,
            td_loss_weight=self.field_td_loss_weight,
            speed_loss_weight=self.field_speed_loss_weight,
            log_speed_loss_weight=self.field_log_speed_loss_weight,
            direct_speed_loss_weight=self.field_direct_speed_loss_weight,
            normal_loss_weight=self.field_normal_loss_weight,
            normal_cos_loss_weight=self.field_normal_cos_loss_weight,
            near_obstacle_loss_weight=self.field_near_obstacle_loss_weight,
            low_speed_threshold=self.field_low_speed_threshold,
            low_speed_pred_max=self.field_low_speed_pred_max,
            low_speed_penalty_weight=self.field_low_speed_penalty_weight,
            effective_speed_floor=self.field_effective_speed_floor,
        )
        self.goal_selector = ViewGoalSelector(
            self.kinematics,
            self.camera_in_tool,
            min_goal_joint_delta_rad=self.min_goal_joint_delta_rad,
            min_camera_goal_delta_m=self.min_camera_goal_delta_m,
            frontier_reselect_cooldown_steps=self.frontier_reselect_cooldown_steps,
            min_frontier_visibility_score=self.min_frontier_visibility_score,
            min_roi_coverage_ratio=self.min_roi_coverage_ratio,
            min_view_alignment=self.min_view_alignment,
            min_actual_view_alignment=self.min_actual_view_alignment,
            target_context_radius_m=self.target_context_radius_m,
            frontier_pose_candidates_per_frontier=self.frontier_pose_candidates_per_frontier,
            yaw_offsets_deg=self.camera_yaw_offsets_deg,
            pitch_offsets_deg=self.camera_pitch_offsets_deg,
            roll_offsets_deg=self.camera_roll_offsets_deg,
            fast_roi_nbv=self.fast_roi_nbv,
            roi_nbv_max_pose_candidates=self.roi_nbv_max_pose_candidates,
            enable_self_occlusion_filter=self.enable_view_self_occlusion_filter,
            min_self_occlusion_free_fraction=self.min_view_self_occlusion_free_fraction,
            self_occlusion_padding_m=self.view_self_occlusion_padding_m,
            self_occlusion_ignore_near_origin_m=self.view_self_occlusion_ignore_near_origin_m,
            self_occlusion_tool_radius_m=self.robot_self_filter_tool_radius_m,
            self_occlusion_mount_radius_m=self.robot_self_filter_mount_radius_m,
        )
        self.rng = np.random.default_rng(7)
        self.planner = ArmFieldPlanner(self.field_model, self.kinematics)
        self.exec_planner = JointSpaceRRTConnectPlanner(self.kinematics, rng=self.rng)
        self.robot_self_filter_checker = UR5PointCloudCollisionChecker(
            self.kinematics, np.zeros((0, 3), dtype=np.float32)
        )

        self.current_joints: np.ndarray | None = None
        self.path_training_anchor_qs = np.zeros((0, 6), dtype=np.float64)
        self.hard_failed_anchor_qs = np.zeros((0, 6), dtype=np.float64)
        self.last_false_free_audit_step = -1
        self.last_false_free_audit: dict[str, float | str] = {}
        self.current_joint_map: dict[str, float] = {}
        self.last_camera_pose: np.ndarray | None = None
        self.latest_camera_info: CameraInfo | None = None
        self.using_fallback_camera_info = False
        self.step_idx = 0
        self.frame_count = 0
        self.last_plan: np.ndarray | None = None
        self.latest_goal_meta: dict | None = None
        self.latest_color: np.ndarray | None = None
        self.trajectory_busy_until = 0.0
        self.last_sample_progress_log = 0.0
        self.frontier_roi_min: np.ndarray | None = None
        self.frontier_roi_max: np.ndarray | None = None
        self.frontier_roi_source = "uninitialized"
        self.last_joint_state_wall_time = 0.0
        self.last_joint_state_log_time = 0.0
        self.last_trajectory_publish_wall_time = 0.0
        self.startup_banner_logged = False
        self.network_initialized = False
        self.network_initialized_wall_time = 0.0
        self.training_finished = False
        self.final_artifacts_saved = False
        self.frontier_completion_empty_steps = 0
        self.frontiers_seen_since_init = False
        self.startup_pose_reached = not self.enable_startup_pose_gate
        self.startup_reached_log_emitted = False
        self.last_startup_wait_log_time = 0.0
        self.startup_pose_ready_since_wall_time = 0.0
        self.training_frame_idx = 0
        self.training_update_count = 0
        self.defer_training_until_motion = False
        self.defer_training_q: np.ndarray | None = None
        self.defer_training_publish_time = 0.0
        self.last_bootstrap_debug: dict[str, int] = {}
        self.last_roi_coverage: dict[str, float | int] = {}
        self.last_field_eval: dict[str, object] = {"success_ratio": 0.0, "success_count": 0, "evaluated": 0}
        self.recent_viewpoints: list[tuple[int, np.ndarray]] = []
        self.field_eval_anchor_cache_qs = np.zeros((0, 6), dtype=np.float64)
        self.field_eval_anchor_cache_step = -10_000
        self.field_eval_goal_candidate_cache: list[list[tuple[np.ndarray, float]]] = []
        self.field_eval_goal_candidate_cache_step = -10_000

        self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 20)
        self.create_subscription(Image, self.color_topic, self._color_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos_profile_sensor_data)
        if self.scene_boxes_topic:
            self.create_subscription(String, self.scene_boxes_topic, self._scene_boxes_cb, 5)
        self.frontier_pub = self.create_publisher(MarkerArray, "/ur_mntfields_arm/frontiers", 5)
        self.trajectory_pub = self.create_publisher(JointTrajectory, str(self.get_parameter("trajectory_topic").value), 5)
        self.debug_points_pub = self.create_publisher(PointCloud2, self.debug_point_cloud_topic, 5)
        self.path_pub = self.create_publisher(NavPath, self.planned_path_topic, 5)
        self.get_logger().info(
            "Arm explorer initialized: "
            f"base_frame={self.base_frame}, camera_frame={self.camera_frame}, "
            f"depth_topic={self.depth_topic}, color_topic={self.color_topic}, camera_info_topic={self.camera_info_topic}, "
            f"sample_pairs_per_step={self.sample_pairs_per_step}, samples_per_ik_seed={self.samples_per_ik_seed}, "
            f"train_epochs_per_step={self.train_epochs_per_step}, "
            f"train_every_n_frames={self.train_every_n_frames}, "
            f"replay_buffer_capacity={self.replay_buffer_capacity}, train_minibatch_size={self.train_minibatch_size}, "
            f"train_gradient_accumulation_steps={self.train_gradient_accumulation_steps}, "
            f"train_replay_ratio={self.train_replay_ratio:.2f}, "
            f"hard_example_train_ratio={self.hard_example_train_ratio:.2f}, "
            f"voxel_size_m={self.voxel_map.voxel_size:.3f}, "
            f"trajectory_publish={self.enable_trajectory_publish}, output_dir={self.output_dir}"
        )
        self.get_logger().info(
            "Field model initialized: "
            f"dim={self.field_model.dim}, device={self.field_model.device}, "
            f"model_dir={self.field_model.model_dir}, lr={self.field_model.learning_rate}, "
            f"replay_capacity={self.field_model.replay_capacity}, minibatch_size={self.field_model.minibatch_size}, "
            f"effective_batch_size={self.field_model.effective_minibatch_size}"
        )
        self.get_logger().info(
            f"Sampler collision device: {self.robot_self_filter_checker.device}"
        )
        self.get_logger().info(
            "Training state is 6-DoF joint space: q = "
            f"{', '.join(JOINT_NAMES)}"
        )
    def _startup_pose_is_ready(self) -> bool:
        if not self.enable_startup_pose_gate:
            return True
        if self.current_joints is None:
            return False
        err = np.abs(np.asarray(self.current_joints, dtype=np.float64) - self.startup_positions)
        return bool(np.all(err <= self.startup_pose_tolerance_rad))

    def _log_startup_wait_if_needed(self):
        if not self.enable_startup_pose_gate or self.current_joints is None:
            return
        now = time.monotonic()
        if now - self.last_startup_wait_log_time < self.startup_wait_log_interval_s:
            return
        self.last_startup_wait_log_time = now
        err = np.abs(np.asarray(self.current_joints, dtype=np.float64) - self.startup_positions)
        max_idx = int(np.argmax(err))
        self.get_logger().info(
            "Waiting for startup inspection pose before initializing network: "
            f"max_error_joint={JOINT_NAMES[max_idx]} error={err[max_idx]:.3f} "
            f"target_q={np.round(self.startup_positions, 3).tolist()} "
            f"current_q={np.round(self.current_joints, 3).tolist()}"
        )

    def _maybe_mark_startup_pose_reached(self):
        if self.startup_pose_reached:
            return
        if not self._startup_pose_is_ready():
            self.startup_pose_ready_since_wall_time = 0.0
            return
        now = time.monotonic()
        if self.startup_pose_ready_since_wall_time <= 0.0:
            self.startup_pose_ready_since_wall_time = now
        delay_remaining = self.startup_data_collection_delay_s - (now - self.startup_pose_ready_since_wall_time)
        if delay_remaining > 0.0:
            if now - self.last_startup_wait_log_time >= self.startup_wait_log_interval_s:
                self.last_startup_wait_log_time = now
                self.get_logger().info(
                    "Startup inspection pose reached; collecting initial depth before training: "
                    f"remaining_s={delay_remaining:.1f}"
                )
            return
        self.startup_pose_reached = True
        self.training_frame_idx = 0
        self.get_logger().info(
            "Startup inspection pose reached. Beginning network initialization, sample collection, and training."
        )

    def _transform_points_for_visualization(self, points: np.ndarray) -> tuple[np.ndarray, str]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0 or self.visualization_frame == self.base_frame:
            return pts, self.base_frame
        try:
            tf = self.tf_buffer.lookup_transform(self.visualization_frame, self.base_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"Visualization TF lookup failed ({self.base_frame} -> {self.visualization_frame}): {exc}"
            )
            return pts, self.base_frame
        tf_m = _transform_to_matrix(tf)
        pts_v = (tf_m[:3, :3] @ pts.T).T + tf_m[:3, 3][None, :]
        return pts_v.astype(np.float32), self.visualization_frame

    def _transform_point_for_visualization(self, xyz: np.ndarray) -> np.ndarray:
        pt = np.asarray(xyz, dtype=np.float64).reshape(1, 3)
        pts_v, _ = self._transform_points_for_visualization(pt)
        return pts_v[0].astype(np.float64)

    def _transform_point_between_frames(self, xyz: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray | None:
        pt = np.asarray(xyz, dtype=np.float64).reshape(3)
        if source_frame == target_frame:
            return pt.copy()
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"Point TF lookup failed ({source_frame} -> {target_frame}): {exc}")
            return None
        tf_m = _transform_to_matrix(tf)
        return (tf_m[:3, :3] @ pt) + tf_m[:3, 3]

    def _parse_scene_boxes(self, entries) -> list[np.ndarray]:
        boxes = []
        for entry in entries:
            if isinstance(entry, str):
                if not entry.strip():
                    continue
                values = [float(x) for x in entry.split(",")]
            else:
                values = [float(x) for x in entry]
            if len(values) != 6:
                continue
            boxes.append(np.asarray(values, dtype=np.float64))
        return boxes

    def _parse_scene_boxes_message(self, text: str) -> tuple[list[np.ndarray], str | None]:
        payload = str(text).strip()
        if not payload:
            return [], None
        frame = None
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                frame = str(data.get("frame", "")).strip() or None
                data = data.get("boxes", [])
            if isinstance(data, list):
                if data and all(isinstance(v, (int, float)) for v in data):
                    data = [data]
                return self._parse_scene_boxes(data), frame
        except json.JSONDecodeError:
            pass
        entries = [chunk.strip() for chunk in payload.replace("\n", ";").split(";") if chunk.strip()]
        return self._parse_scene_boxes(entries), frame

    def _scene_boxes_cb(self, msg: String):
        boxes, frame = self._parse_scene_boxes_message(msg.data)
        if not boxes:
            self.get_logger().warn(
                f"Ignoring empty/invalid scene_boxes message on {self.scene_boxes_topic}. "
                "Use JSON {'frame':'base_link','boxes':[[x,y,z,sx,sy,sz],...]} or 'x,y,z,sx,sy,sz;...'."
            )
            return
        self.scene_boxes = boxes
        if frame:
            self.scene_boxes_frame = frame
        self.frontier_roi_min = None
        self.frontier_roi_max = None
        self.frontier_roi_source = "scene_boxes_topic_pending"
        self.get_logger().info(
            f"Updated scene_boxes from topic {self.scene_boxes_topic}: "
            f"boxes={len(self.scene_boxes)} frame={self.scene_boxes_frame}. ROI will reinitialize on next depth frame."
        )

    def _scene_boxes_bbox(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.scene_boxes:
            return None
        tf_m = np.eye(4, dtype=np.float64)
        if self.scene_boxes_frame != self.base_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, self.scene_boxes_frame, rclpy.time.Time())
                tf_m = _transform_to_matrix(tf)
            except TransformException as exc:
                self.get_logger().warn(
                    f"Scene box TF lookup failed ({self.scene_boxes_frame} -> {self.base_frame}): {exc}"
                )
                return None
        mins = []
        maxs = []
        for box in self.scene_boxes:
            center = np.asarray(box[:3], dtype=np.float64)
            size = np.maximum(np.asarray(box[3:], dtype=np.float64), 0.0)
            lo = center - 0.5 * size
            hi = center + 0.5 * size
            corners = np.asarray(
                [
                    [x, y, z, 1.0]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ],
                dtype=np.float64,
            )
            corners_base = (tf_m @ corners.T).T[:, :3]
            mins.append(np.min(corners_base, axis=0))
            maxs.append(np.max(corners_base, axis=0))
        if not mins:
            return None
        lo = np.min(np.asarray(mins, dtype=np.float64), axis=0) - self.cabinet_bbox_padding_xyz
        hi = np.max(np.asarray(maxs, dtype=np.float64), axis=0) + self.cabinet_bbox_padding_xyz
        return self._clip_roi_bounds(lo, hi)

    def _boxes_in_base(
        self,
        boxes: list[np.ndarray],
        source_frame: str,
        label: str,
    ) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 6), dtype=np.float64)
        tf_m = np.eye(4, dtype=np.float64)
        if source_frame != self.base_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, source_frame, rclpy.time.Time())
                tf_m = _transform_to_matrix(tf)
            except TransformException as exc:
                self.get_logger().warn(
                    f"{label} box TF lookup failed ({source_frame} -> {self.base_frame}): {exc}"
                )
                return np.zeros((0, 6), dtype=np.float64)
        out = []
        for box in boxes:
            center = np.asarray(box[:3], dtype=np.float64)
            size = np.maximum(np.asarray(box[3:], dtype=np.float64), 0.0)
            lo = center - 0.5 * size
            hi = center + 0.5 * size
            corners = np.asarray(
                [
                    [x, y, z, 1.0]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ],
                dtype=np.float64,
            )
            corners_base = (tf_m @ corners.T).T[:, :3]
            base_lo = np.min(corners_base, axis=0)
            base_hi = np.max(corners_base, axis=0)
            out.append(np.concatenate((0.5 * (base_lo + base_hi), np.maximum(base_hi - base_lo, 1e-6))))
        return np.asarray(out, dtype=np.float64)

    def _scene_boxes_in_base(self) -> np.ndarray:
        return self._boxes_in_base(self.scene_boxes, self.scene_boxes_frame, "Scene")

    def _support_boxes_in_base(self) -> np.ndarray:
        return self._boxes_in_base(self.support_boxes, self.support_boxes_frame, "Support")

    def _make_collision_checker(
        self,
        occupied_points: np.ndarray,
        *,
        clearance_backend: str | None = None,
    ) -> UR5PointCloudCollisionChecker:
        scene_boxes = self._scene_boxes_in_base()
        support_boxes = self._support_boxes_in_base()
        if len(scene_boxes) and len(support_boxes):
            box_obstacles = np.concatenate((scene_boxes, support_boxes), axis=0)
        elif len(scene_boxes):
            box_obstacles = scene_boxes
        else:
            box_obstacles = support_boxes
        backend = self.clearance_backend if clearance_backend is None else str(clearance_backend).strip().lower()
        return make_ur5_collision_checker(
            self.kinematics,
            occupied_points,
            box_obstacles=box_obstacles,
            support_box_count=len(support_boxes),
            support_point_ignore_padding_m=self.support_point_ignore_padding_m,
            clearance_backend=backend,
            sdf_voxel_size_m=self.sdf_voxel_size_m,
            sdf_padding_m=self.sdf_padding_m,
            sdf_max_cells=self.sdf_max_cells,
        )

    def _clip_roi_bounds(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clip_lo = np.asarray(self.roi_clip_min_xyz, dtype=np.float64)
        clip_hi = np.asarray(self.roi_clip_max_xyz, dtype=np.float64)
        if self.roi_clip_frame != self.base_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, self.roi_clip_frame, rclpy.time.Time())
                tf_m = _transform_to_matrix(tf)
                corners = np.asarray(
                    [
                        [x, y, z, 1.0]
                        for x in (clip_lo[0], clip_hi[0])
                        for y in (clip_lo[1], clip_hi[1])
                        for z in (clip_lo[2], clip_hi[2])
                    ],
                    dtype=np.float64,
                )
                corners_base = (tf_m @ corners.T).T[:, :3]
                clip_lo = np.min(corners_base, axis=0)
                clip_hi = np.max(corners_base, axis=0)
            except TransformException as exc:
                self.get_logger().warn(
                    f"ROI clip TF lookup failed ({self.roi_clip_frame} -> {self.base_frame}): {exc}"
                )
        lo = np.maximum(np.asarray(lo, dtype=np.float64), clip_lo)
        hi = np.minimum(np.asarray(hi, dtype=np.float64), clip_hi)
        hi = np.maximum(hi, lo + 1.0e-3)
        return lo, hi

    def _maybe_initialize_network(self, occupied_points: np.ndarray) -> bool:
        if self.network_initialized:
            return True
        if self.enable_frontier_roi_filter and (self.frontier_roi_min is None or self.frontier_roi_max is None):
            return False
        if len(occupied_points) == 0:
            return False
        self.get_logger().info("Initializing network")
        self.network_initialized = True
        self.network_initialized_wall_time = time.monotonic()
        self.get_logger().info(
            "Network initialization complete: "
            f"roi_source={self.frontier_roi_source}, occupied_points={len(occupied_points)}, "
            f"support_boxes={len(self.support_boxes)}"
        )
        self.get_logger().info(
            "Field loss config: "
            f"speed_weight={self.field_speed_loss_weight:.4g}, "
            f"log_speed_weight={self.field_log_speed_loss_weight:.4g}, "
            f"direct_speed_weight={self.field_direct_speed_loss_weight:.4g}, "
            f"normal_weight={self.field_normal_loss_weight:.4g}, "
            f"normal_cos_weight={self.field_normal_cos_loss_weight:.4g}, "
            f"near_weight={self.field_near_obstacle_loss_weight:.3g}, "
            f"low_threshold={self.field_low_speed_threshold:.3f}, "
            f"low_pred_max={self.field_low_speed_pred_max:.3f}, "
            f"low_penalty={self.field_low_speed_penalty_weight:.4g}, "
            f"effective_floor={self.field_effective_speed_floor:.3f}"
        )
        return True

    def _finish_training(self, reason: str, *, certified: bool = True):
        if self.training_finished:
            return
        self.training_finished = True
        enough_training = certified and (
            self.field_model.total_epochs_trained >= self.field_eval_min_train_steps
            and self.field_model.replay_size >= self.field_eval_min_replay_pairs
        )
        if enough_training:
            self.get_logger().info("Training completed.")
        else:
            self.get_logger().info(f"Training stopped before completion: {reason}")
        checkpoint_name = "weights_final.pt" if enough_training else "weights_partial.pt"
        final_path = self.model_artifacts_dir / checkpoint_name
        self.field_model.save_checkpoint(final_path)
        self.field_model.save_loss_plot(self.model_artifacts_dir / "train_loss.png")
        self.final_artifacts_saved = True
        if not enough_training:
            self.get_logger().warn(
                "Saved partial model checkpoint instead of weights_final.pt because training stopped before "
                f"the configured minimums: epochs={self.field_model.total_epochs_trained}/"
                f"{self.field_eval_min_train_steps}, replay={self.field_model.replay_size}/"
                f"{self.field_eval_min_replay_pairs}. checkpoint={final_path}"
            )
        self._write_status()

    def _joint_state_cb(self, msg: JointState):
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        if not all(name in name_to_idx for name in JOINT_NAMES):
            return
        q = np.array([msg.position[name_to_idx[name]] for name in JOINT_NAMES], dtype=np.float64)
        self.current_joints = q
        self.current_joint_map = {name: float(val) for name, val in zip(JOINT_NAMES, q)}
        self.last_joint_state_wall_time = time.monotonic()
        now = self.last_joint_state_wall_time
        if now - self.last_joint_state_log_time > 5.0:
            self.last_joint_state_log_time = now
            self.get_logger().info(
                f"Joint state update: q={np.round(self.current_joints, 3).tolist()}"
            )

    def _joint_state_is_valid_for_planning(self) -> tuple[bool, float, str]:
        if self.current_joints is None or self.last_joint_state_wall_time <= 0.0:
            return False, -1.0, "missing"
        joint_age = time.monotonic() - self.last_joint_state_wall_time
        trajectory_was_published_since_joint_update = (
            self.last_trajectory_publish_wall_time > self.last_joint_state_wall_time
        )
        if not trajectory_was_published_since_joint_update:
            return True, joint_age, "stationary"
        if joint_age <= self.max_joint_state_age_s:
            return True, joint_age, "fresh_after_motion"
        return False, joint_age, "stale_after_motion"

    def _color_cb(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        except Exception:
            return

    def _camera_info_cb(self, msg: CameraInfo):
        self.latest_camera_info = msg
        if self.using_fallback_camera_info:
            self.get_logger().info("Received camera_info after fallback mode; switching to live camera intrinsics.")
            self.using_fallback_camera_info = False
        if self.frame_count == 0:
            self.get_logger().info(
                f"Received camera_info: frame_id={msg.header.frame_id} size={msg.width}x{msg.height}"
            )

    def _effective_camera_info(self, depth_msg: Image) -> CameraInfo | None:
        if self.latest_camera_info is not None:
            return self.latest_camera_info
        if not self.using_fallback_camera_info:
            self.get_logger().warn(
                "No camera_info received; using configured fallback intrinsics for depth processing."
            )
            self.using_fallback_camera_info = True
        info = CameraInfo()
        info.header = depth_msg.header
        info.width = int(depth_msg.width) if int(depth_msg.width) > 0 else self.fallback_camera_width
        info.height = int(depth_msg.height) if int(depth_msg.height) > 0 else self.fallback_camera_height
        info.k = [
            self.fallback_fx, 0.0, self.fallback_cx,
            0.0, self.fallback_fy, self.fallback_cy,
            0.0, 0.0, 1.0,
        ]
        return info

    def _lookup_camera_pose(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return None
        return _transform_to_matrix(tf)

    def _depth_cb(self, depth_msg: Image):
        callback_t0 = time.perf_counter()
        if self.current_joints is None:
            return
        info_msg = self._effective_camera_info(depth_msg)
        if info_msg is None:
            return
        self.frame_count += 1
        camera_pose = self._lookup_camera_pose()
        if camera_pose is None:
            return
        tf_t1 = time.perf_counter()
        self.last_camera_pose = camera_pose
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        if depth_msg.encoding == "16UC1":
            depth *= 0.001
        points = self._depth_to_world(depth, info_msg, camera_pose)
        raw_depth_points = int(len(points))
        points, self_points_removed = self._filter_robot_self_points(points)
        points_t2 = time.perf_counter()
        if points.size == 0:
            if self.frame_count % 15 == 0:
                self.get_logger().info(
                    "Depth frame had no valid points after filtering: "
                    f"raw_points={raw_depth_points} self_removed={self_points_removed}"
                )
            return

        self.step_idx += 1
        self.voxel_map.integrate_points(camera_pose[:3, 3], points)
        occupied_points = self.voxel_map.occupied_points()
        self._maybe_initialize_frontier_roi(occupied_points)
        self._publish_debug_point_cloud(occupied_points, depth_msg.header.stamp)
        clusters = self.voxel_map.frontier_clusters()
        clusters = self._filter_clusters_to_frontier_roi(clusters)
        self.frontier_bank.update(clusters, self.step_idx)
        self.frontier_bank.mark_visited_near(camera_pose[:3, 3])
        checker = self._make_collision_checker(occupied_points)
        checker.free_keys = set(self.voxel_map.free)
        # Lazily created for the expensive online-only paths below. The exact
        # checker remains the source of truth for planner validation.
        fast_clearance_checker: UR5PointCloudCollisionChecker | None = None
        map_t3 = time.perf_counter()
        if self.step_idx <= 3 or self.step_idx % 10 == 0:
            roi_status = "uninitialized"
            if self.frontier_roi_min is not None and self.frontier_roi_max is not None:
                roi_status = (
                    f"roi_min={np.round(self.frontier_roi_min, 2).tolist()} "
                    f"roi_max={np.round(self.frontier_roi_max, 2).tolist()}"
                )
            self.get_logger().info(
                f"step={self.step_idx} depth callback: tf_ms={(tf_t1 - callback_t0) * 1e3:.1f} "
                f"points_ms={(points_t2 - tf_t1) * 1e3:.1f} map_ms={(map_t3 - points_t2) * 1e3:.1f} "
                f"raw_points={raw_depth_points} filtered_points={len(points)} "
                f"self_removed={self_points_removed} occupied={len(occupied_points)} frontiers={len(clusters)} "
                f"{roi_status}"
            )

        self._maybe_mark_startup_pose_reached()
        if not self.startup_pose_reached:
            self._log_startup_wait_if_needed()
            self._publish_frontiers()
            return

        if not self._maybe_initialize_network(occupied_points):
            self._publish_frontiers()
            return
        if (
            self.training_wall_time_limit_s > 0.0
            and self.network_initialized_wall_time > 0.0
            and time.monotonic() - self.network_initialized_wall_time
            >= self.training_wall_time_limit_s
        ):
            elapsed = time.monotonic() - self.network_initialized_wall_time
            self._finish_training(
                f"wall-time limit reached ({elapsed:.1f}s) before raw field evaluation passed",
                certified=False,
            )
            self._publish_frontiers()
            return
        self.training_frame_idx += 1

        active_all = sorted(
            self._active_frontiers_in_roi(),
            key=lambda rec: (-rec.voxel_count, rec.times_failed, rec.times_selected),
        )
        local_frontiers = sorted(
            [
                rec
                for rec in self.frontier_bank.local_active_records(
                    self.step_idx, max_age_steps=self.local_frontier_recent_steps
                )
                if self._point_in_frontier_roi(rec.centroid)
            ],
            key=lambda rec: (-rec.voxel_count, rec.times_failed, rec.times_selected),
        )
        global_frontiers = sorted(
            [
                rec
                for rec in self.frontier_bank.global_active_records(
                    self.step_idx, min_age_steps=self.local_frontier_recent_steps + 1
                )
                if self._point_in_frontier_roi(rec.centroid)
            ],
            key=lambda rec: (-rec.voxel_count, rec.times_failed, rec.times_selected),
        )
        if active_all:
            self.frontiers_seen_since_init = True
            self.frontier_completion_empty_steps = 0
        elif self.frontiers_seen_since_init:
            self.frontier_completion_empty_steps += 1
            has_eval_minimums = (
                self.field_model.total_epochs_trained >= self.field_eval_min_train_steps
                and self.field_model.replay_size >= self.field_eval_min_replay_pairs
            )
            if (
                self.finish_when_frontiers_exhausted
                and not self.training_finished
                and has_eval_minimums
                and not self.finish_when_field_eval_passes
                and self.frontier_completion_empty_steps >= self.frontier_completion_patience_steps
            ):
                self._finish_training(
                    f"all frontier goal points cleared for {self.frontier_completion_empty_steps} consecutive steps"
                )
            elif (
                self.finish_when_frontiers_exhausted
                and not self.training_finished
                and not has_eval_minimums
                and self.frontier_completion_empty_steps >= self.frontier_completion_patience_steps
            ):
                self.get_logger().info(
                    f"Frontiers have been empty for {self.frontier_completion_empty_steps} steps, "
                    "but training will continue until fixed-goal eval minimums are met: "
                    f"epochs={self.field_model.total_epochs_trained}/{self.field_eval_min_train_steps}, "
                    f"replay={self.field_model.replay_size}/{self.field_eval_min_replay_pairs}."
                )
        if self.training_finished:
            self._publish_frontiers()
            return
        is_executing = (
            self.enable_trajectory_publish
            and not self.replan_while_executing
            and time.monotonic() < self.trajectory_busy_until
        )
        if not self.startup_banner_logged:
            self.startup_banner_logged = True
            self.get_logger().info(
                "Training/exploration ready: "
                f"strict_train_move_cycle={self.strict_train_move_cycle}, "
                f"train_during_motion={self.train_during_motion}, sample_pairs_per_step={self.sample_pairs_per_step}, "
                f"sampling_mode={self.sampling_mode}, sampling_proposal_batch_size={self.sampling_proposal_batch_size}, "
                f"planner_clearance_backend={self.clearance_backend}, "
                f"sampling_clearance_backend={self.sampling_clearance_backend}, "
                f"nbv_clearance_backend={self.nbv_clearance_backend}, sdf_voxel_size_m={self.sdf_voxel_size_m:.3f}, "
                f"path_centered_pair_fraction={self.path_centered_pair_fraction:.2f}, "
                f"near_boundary_pair_fraction={self.near_boundary_pair_fraction:.2f}, "
                f"execution_planner={self.execution_planner}, "
                f"support_boxes={len(self.support_boxes)}, "
                f"rank_frontiers_when_roi_available={self.rank_frontiers_when_roi_available}, "
                f"frontier_fallback_max_frontiers={self.frontier_fallback_max_frontiers}, "
                f"train_steps={self.train_epochs_per_step}, train_every_n_frames={self.train_every_n_frames}, "
                f"replay_size={self.field_model.replay_size}/{self.field_model.replay_capacity}, "
                f"minibatch_size={self.field_model.minibatch_size}"
            )

        (
            active_sample_pairs,
            active_train_steps,
            active_train_every_n_frames,
            active_recombine_pairs,
            active_hard_failed_pairs,
            adaptive_ready,
        ) = self._active_training_schedule()
        should_train_this_frame = (
            self.training_frame_idx == 1
            or (
                active_train_every_n_frames > 0
                and self.training_frame_idx > 1
                and (self.training_frame_idx - 1) % active_train_every_n_frames == 0
            )
        )
        if self.defer_training_until_motion and self.current_joints is not None and self.defer_training_q is not None:
            joint_motion = float(
                np.max(np.abs(np.asarray(self.current_joints, dtype=np.float64) - self.defer_training_q))
            )
            published_after_defer = self.last_trajectory_publish_wall_time > self.defer_training_publish_time
            if published_after_defer and joint_motion >= self.retrain_same_pose_joint_threshold_rad:
                self.defer_training_until_motion = False
                self.defer_training_q = None
                self.defer_training_publish_time = 0.0
                self.get_logger().info(
                    f"Training defer gate released after executed motion: joint_motion={joint_motion:.4f} rad"
                )
            else:
                should_train_this_frame = False
        if should_train_this_frame and (self.train_during_motion or not is_executing):
            sample_t0 = time.perf_counter()
            sampling_checker = checker
            sampling_checker_build_ms = 0.0
            if self.sampling_clearance_backend != self.clearance_backend:
                checker_t0 = time.perf_counter()
                sampling_checker = self._make_collision_checker(
                    occupied_points,
                    clearance_backend=self.sampling_clearance_backend,
                )
                sampling_checker.free_keys = checker.free_keys
                sampling_checker_build_ms = (time.perf_counter() - checker_t0) * 1e3
            fast_clearance_checker = sampling_checker
            self.get_logger().info(
                f"step={self.step_idx} post_startup_frame={self.training_frame_idx} collecting training samples: "
                f"target_pairs={active_sample_pairs} adaptive_ready={adaptive_ready} "
                f"active_frontiers={len(active_all)} occupied_points={len(occupied_points)} "
                f"samples_per_ik_seed={self.samples_per_ik_seed} "
                f"sampling_backend={self.sampling_clearance_backend} checker={type(sampling_checker).__name__} "
                f"checker_build_ms={sampling_checker_build_ms:.1f} "
                f"support_points_filtered={getattr(sampling_checker, 'ignored_support_point_count', 0)}"
            )
            self.last_sample_progress_log = 0.0

            def _sample_progress(stage_name: str, stage_target: int):
                def _report(attempts: int, accepted: int, ik_seed_tries: int):
                    now = time.monotonic()
                    if now - self.last_sample_progress_log < 2.0:
                        return
                    self.last_sample_progress_log = now
                    self.get_logger().info(
                        f"step={self.step_idx} post_startup_frame={self.training_frame_idx} "
                        f"sampling {stage_name} progress: accepted={accepted}/{stage_target} "
                        f"attempts={attempts} seed_batches={ik_seed_tries}"
                    )

                return _report

            sampler_stats_list: list[dict[str, float]] = []
            raw_rows_list: list[np.ndarray] = []
            frame_rows_list: list[np.ndarray] = []
            frame_rows_hard = np.zeros((0, 26), dtype=np.float32)
            path_anchor_sets = []
            if len(self.training_anchor_qs):
                path_anchor_sets.append(self.training_anchor_qs)
            if len(self.path_training_anchor_qs):
                path_anchor_sets.append(self.path_training_anchor_qs)
            path_anchor_qs = (
                np.concatenate(path_anchor_sets, axis=0).astype(np.float64, copy=False)
                if path_anchor_sets
                else np.zeros((0, 6), dtype=np.float64)
            )
            path_pair_target = int(round(active_sample_pairs * self.path_centered_pair_fraction))
            if len(path_anchor_qs) == 0:
                path_pair_target = 0
            path_pair_target = int(np.clip(path_pair_target, 0, active_sample_pairs))
            global_pair_target = int(active_sample_pairs - path_pair_target)
            near_pair_target = int(round(active_sample_pairs * self.near_boundary_pair_fraction))
            near_pair_target = int(np.clip(near_pair_target, 0, global_pair_target))
            broad_pair_target = int(global_pair_target - near_pair_target)
            sample_stage_ms = {
                "path": 0.0,
                "near": 0.0,
                "broad": 0.0,
                "hard": 0.0,
                "concat": 0.0,
                "recombine": 0.0,
            }
            sample_stage_rows = {
                "path_raw": 0,
                "path_frame": 0,
                "near_raw": 0,
                "near_frame": 0,
                "broad_raw": 0,
                "broad_frame": 0,
                "hard_raw": 0,
                "hard_frame": 0,
            }

            if path_pair_target > 0:
                stage_t0 = time.perf_counter()
                raw_rows_path, frame_rows_path, stats_path = sample_path_centered_training_batch(
                    sampling_checker,
                    self.kinematics,
                    path_anchor_qs,
                    path_pair_target,
                    self.clearance_margin_m,
                    self.clearance_offset_m,
                    self.rng,
                    clearance_label_floor=self.clearance_label_floor,
                    clearance_label_power=self.clearance_label_power,
                    proposal_batch_size=self.sampling_proposal_batch_size,
                )
                raw_rows_list.append(raw_rows_path)
                frame_rows_list.append(frame_rows_path)
                sampler_stats_list.append(stats_path)
                sample_stage_ms["path"] = (time.perf_counter() - stage_t0) * 1e3
                sample_stage_rows["path_raw"] = int(len(raw_rows_path))
                sample_stage_rows["path_frame"] = int(len(frame_rows_path))

            anchor_qs_for_sampling = None
            if self.anchor_seed_probability > 0.0:
                anchor_sets = []
                if self.use_field_eval_anchors_for_sampling:
                    eval_anchor_qs = self._field_eval_anchor_qs(sampling_checker)
                    if len(eval_anchor_qs):
                        anchor_sets.append(eval_anchor_qs)
                if len(self.training_anchor_qs):
                    anchor_sets.append(self.training_anchor_qs)
                if len(self.path_training_anchor_qs):
                    anchor_sets.append(self.path_training_anchor_qs)
                if len(self.hard_failed_anchor_qs):
                    anchor_sets.append(self.hard_failed_anchor_qs)
                if anchor_sets:
                    anchor_qs_for_sampling = np.concatenate(anchor_sets, axis=0).astype(np.float64, copy=False)

            if near_pair_target > 0:
                stage_t0 = time.perf_counter()
                raw_rows_near, frame_rows_near, stats_near = sample_cspace_training_batch(
                    sampling_checker,
                    self.kinematics,
                    near_pair_target,
                    self.clearance_margin_m,
                    self.clearance_offset_m,
                    self.rng,
                    samples_per_seed=self.samples_per_ik_seed,
                    roi_min=self.frontier_roi_min,
                    roi_max=self.frontier_roi_max,
                    seed_hint_q=self.current_joints,
                    anchor_qs=anchor_qs_for_sampling,
                    anchor_seed_probability=self.anchor_seed_probability,
                    roi_seed_fraction=self.roi_sampling_seed_fraction,
                    clearance_label_floor=self.clearance_label_floor,
                    clearance_label_power=self.clearance_label_power,
                    sampling_mode=self.sampling_mode,
                    proposal_batch_size=self.sampling_proposal_batch_size,
                    near_boundary_only=True,
                    progress_cb=_sample_progress("near", near_pair_target),
                )
                raw_rows_list.append(raw_rows_near)
                frame_rows_list.append(frame_rows_near)
                sampler_stats_list.append(stats_near)
                sample_stage_ms["near"] = (time.perf_counter() - stage_t0) * 1e3
                sample_stage_rows["near_raw"] = int(len(raw_rows_near))
                sample_stage_rows["near_frame"] = int(len(frame_rows_near))

            if broad_pair_target > 0:
                stage_t0 = time.perf_counter()
                raw_rows_global, frame_rows_global, stats_global = sample_cspace_training_batch(
                    sampling_checker,
                    self.kinematics,
                    broad_pair_target,
                    self.clearance_margin_m,
                    self.clearance_offset_m,
                    self.rng,
                    samples_per_seed=self.samples_per_ik_seed,
                    roi_min=self.frontier_roi_min,
                    roi_max=self.frontier_roi_max,
                    seed_hint_q=self.current_joints,
                    anchor_qs=anchor_qs_for_sampling,
                    anchor_seed_probability=self.anchor_seed_probability,
                    roi_seed_fraction=self.roi_sampling_seed_fraction,
                    clearance_label_floor=self.clearance_label_floor,
                    clearance_label_power=self.clearance_label_power,
                    sampling_mode=self.sampling_mode,
                    proposal_batch_size=self.sampling_proposal_batch_size,
                    near_boundary_only=False,
                    progress_cb=_sample_progress("broad", broad_pair_target),
                )
                raw_rows_list.append(raw_rows_global)
                frame_rows_list.append(frame_rows_global)
                sampler_stats_list.append(stats_global)
                sample_stage_ms["broad"] = (time.perf_counter() - stage_t0) * 1e3
                sample_stage_rows["broad_raw"] = int(len(raw_rows_global))
                sample_stage_rows["broad_frame"] = int(len(frame_rows_global))

            hard_pair_count = 0
            if active_hard_failed_pairs > 0 and len(self.hard_failed_anchor_qs) > 0:
                stage_t0 = time.perf_counter()
                raw_rows_hard, frame_rows_hard, stats_hard = self._hard_failed_training_rows(
                    # Hard anchors come from exact rollout validation and the
                    # held-out false-free audit. Preserve that source of truth
                    # instead of relabelling a missed collision with the
                    # approximate sampling SDF.
                    checker,
                    active_hard_failed_pairs,
                )
                sample_stage_ms["hard"] = (time.perf_counter() - stage_t0) * 1e3
                sample_stage_rows["hard_raw"] = int(len(raw_rows_hard))
                sample_stage_rows["hard_frame"] = int(len(frame_rows_hard))
                if len(frame_rows_hard) > 0:
                    raw_rows_list.append(raw_rows_hard)
                    frame_rows_list.append(frame_rows_hard)
                    sampler_stats_list.append(stats_hard)
                    hard_pair_count = int(len(frame_rows_hard))

            stage_t0 = time.perf_counter()
            raw_rows = (
                np.concatenate([rows for rows in raw_rows_list if len(rows) > 0], axis=0).astype(np.float32, copy=False)
                if any(len(rows) > 0 for rows in raw_rows_list)
                else np.zeros((0, 12), dtype=np.float32)
            )
            frame_rows = (
                np.concatenate([rows for rows in frame_rows_list if len(rows) > 0], axis=0).astype(np.float32, copy=False)
                if any(len(rows) > 0 for rows in frame_rows_list)
                else np.zeros((0, 26), dtype=np.float32)
            )
            persistent_frame_rows = frame_rows
            sample_stage_ms["concat"] = (time.perf_counter() - stage_t0) * 1e3
            recombined_pair_count = 0
            recombined_rows = np.zeros((0, 26), dtype=np.float32)
            if active_recombine_pairs > 0:
                stage_t0 = time.perf_counter()
                recombined_rows = self.field_model.recombine_replay_pairs(active_recombine_pairs)
                if len(recombined_rows) > 0:
                    frame_rows = np.concatenate((frame_rows, recombined_rows), axis=0).astype(np.float32, copy=False)
                    recombined_pair_count = int(len(recombined_rows))
                sample_stage_ms["recombine"] = (time.perf_counter() - stage_t0) * 1e3
            sample_stats = self._merge_sampler_stats(sampler_stats_list)
            sample_t1 = time.perf_counter()
            self.get_logger().info(
                f"step={self.step_idx} training batch: raw_contact_rows={len(raw_rows)} frame_pairs={len(frame_rows)} "
                f"train_steps={active_train_steps} occupied_points={len(self.voxel_map.occupied)} "
                f"sampling_backend={self.sampling_clearance_backend} checker={type(sampling_checker).__name__} "
                f"checker_build_ms={sampling_checker_build_ms:.1f} "
                f"support_points_filtered={getattr(sampling_checker, 'ignored_support_point_count', 0)} "
                f"hard_pairs={hard_pair_count} recombined_pairs={recombined_pair_count} "
                f"sample_ms={(sample_t1 - sample_t0) * 1e3:.1f}"
            )
            self.get_logger().info(
                f"step={self.step_idx} sample timing: "
                f"path_target={path_pair_target} path_rows={sample_stage_rows['path_frame']} path_ms={sample_stage_ms['path']:.1f} "
                f"near_target={near_pair_target} near_rows={sample_stage_rows['near_frame']} near_ms={sample_stage_ms['near']:.1f} "
                f"broad_target={broad_pair_target} broad_rows={sample_stage_rows['broad_frame']} broad_ms={sample_stage_ms['broad']:.1f} "
                f"hard_target={active_hard_failed_pairs} hard_rows={sample_stage_rows['hard_frame']} hard_ms={sample_stage_ms['hard']:.1f} "
                f"concat_ms={sample_stage_ms['concat']:.1f} recombine_rows={recombined_pair_count} "
                f"recombine_ms={sample_stage_ms['recombine']:.1f}"
            )
            self.get_logger().info(
                f"step={self.step_idx} post_startup_frame={self.training_frame_idx} sampler efficiency: "
                f"attempts={int(sample_stats['attempts'])} seed_batches={int(sample_stats['ik_seed_tries'])} "
                f"accepted_pairs={int(sample_stats['accepted_pairs'])} "
                f"acceptance_rate={sample_stats['acceptance_rate']:.3f} "
                f"accepted_per_seed={sample_stats['accepted_per_seed']:.2f} "
                f"refined_q0={int(sample_stats.get('refined_q0', 0.0))} "
                f"refined_q1={int(sample_stats.get('refined_q1', 0.0))} "
                f"anchor_seed_success={int(sample_stats.get('anchor_seed_success', 0.0))}/"
                f"{int(sample_stats.get('anchor_seed_tries', 0.0))} "
                f"roi_seed_success={int(sample_stats.get('roi_seed_success', 0.0))}/"
                f"{int(sample_stats.get('roi_seed_tries', 0.0))} "
                f"workspace_seed_success={int(sample_stats.get('workspace_seed_success', 0.0))}/"
                f"{int(sample_stats.get('workspace_seed_tries', 0.0))} "
                f"path_anchor_buffer={len(self.path_training_anchor_qs)}"
            )
            self.get_logger().info(
                f"step={self.step_idx} sampler diagnostics: "
                f"mode={sample_stats.get('sampling_mode', '')} "
                f"q0_clearance_mean={sample_stats.get('q0_clearance_mean', 0.0):.4f} "
                f"q1_clearance_mean={sample_stats.get('q1_clearance_mean', 0.0):.4f} "
                f"q0_near_margin_frac={sample_stats.get('q0_near_margin_frac', 0.0):.3f} "
                f"q1_near_margin_frac={sample_stats.get('q1_near_margin_frac', 0.0):.3f} "
                f"q0_boundary_shell_frac={sample_stats.get('q0_boundary_shell_frac', 0.0):.3f} "
                f"q1_obstacle_side_frac={sample_stats.get('q1_obstacle_side_frac', 0.0):.3f} "
                f"speed0_critical_frac={sample_stats.get('speed0_critical_frac', 0.0):.3f} "
                f"speed1_critical_frac={sample_stats.get('speed1_critical_frac', 0.0):.3f} "
                f"speed0_sat_frac={sample_stats.get('speed0_sat_frac', 0.0):.3f} "
                f"speed1_sat_frac={sample_stats.get('speed1_sat_frac', 0.0):.3f}"
            )
            if len(frame_rows):
                self.get_logger().info(
                    f"step={self.step_idx} post_startup_frame={self.training_frame_idx} "
                    f"starting training: train_steps={active_train_steps} "
                    f"fresh_pairs={len(persistent_frame_rows)} transient_pairs={len(recombined_rows)} "
                    f"replay_size={self.field_model.replay_size} "
                    f"eval_replay_target={self.field_eval_min_replay_pairs}"
                )
                train_t0 = time.perf_counter()
                loss = self.field_model.train_step(
                    persistent_frame_rows,
                    active_train_steps,
                    transient_rows=recombined_rows,
                    priority_rows=frame_rows_hard,
                )
                train_t1 = time.perf_counter()
                self.get_logger().info(
                    f"step={self.step_idx} training complete: train_steps={active_train_steps} "
                    f"loss={-1.0 if loss is None else loss:.6f} "
                    f"state_batch_size={self.field_model.last_train_batch_size} "
                    f"pair_batch_size={self.field_model.last_train_pair_count} "
                    f"replay_size={self.field_model.replay_size} "
                    f"train_ms={(train_t1 - train_t0) * 1e3:.1f}"
                )
                self.training_update_count += 1
                should_run_diagnostics = (
                    self.training_update_count == 1
                    or self.training_update_count % self.field_diagnostics_every_n_train_updates == 0
                )
                checkpoint_epoch = (
                    self.field_model.total_epochs_trained // self.checkpoint_every_epochs
                ) * self.checkpoint_every_epochs
                checkpoint_diagnostics_due = (
                    checkpoint_epoch > self.field_model.last_checkpoint_epoch and checkpoint_epoch > 0
                )
                # The checkpoint plot routine evaluates the same replay metric.
                # Let it own that pass so a diagnostic checkpoint does not run
                # two large second-derivative evaluations back-to-back.
                if self.enable_field_diagnostics and should_run_diagnostics and not checkpoint_diagnostics_due:
                    diag_t0 = time.perf_counter()
                    diag = self.field_model.evaluate_replay_diagnostics(self.field_diagnostics_max_rows)
                    diag_t1 = time.perf_counter()
                    if diag.get("diag_rows", 0.0) > 0.0:
                        self.get_logger().info(
                            f"step={self.step_idx} field diagnostics: rows={int(diag.get('diag_rows', 0.0))} "
                            f"speed_mae={diag.get('speed_mae', 0.0):.4f} "
                            f"speed_corr={diag.get('speed_corr', 0.0):.3f} "
                            f"pred_near={diag.get('pred_near_mean', 0.0):.3f} "
                            f"pred_far={diag.get('pred_far_mean', 0.0):.3f} "
                            f"near_far_gap={diag.get('near_far_gap', 0.0):.3f} "
                            f"normal_cos_near={diag.get('normal_cos_near_mean', 0.0):.3f} "
                            f"grad_goal_cos={diag.get('grad_goal_cos_mean', 0.0):.3f} "
                            f"grad_goal_neg_frac={diag.get('grad_goal_neg_frac', 0.0):.3f} "
                            f"low_overpred={diag.get('low_target_overpred_frac', 0.0):.3f} "
                            f"diag_ms={(diag_t1 - diag_t0) * 1e3:.1f}"
                        )
                self._save_step_artifacts(
                    step_idx=self.step_idx,
                    depth_image_m=depth,
                    camera_info=info_msg,
                    camera_pose=camera_pose,
                    points_world=points,
                    raw_rows=raw_rows,
                    # Synthetic recombinations are optimizer-only rows. Keep
                    # saved frame_data collision-labelled so offline replay
                    # cannot re-ingest transient pairs as observations.
                    frame_rows=persistent_frame_rows,
                    occupied_points=occupied_points,
                    active_frontiers=active_all,
                    loss=loss,
                )
                self._save_model_artifacts_if_needed()
            else:
                self.get_logger().info(
                    f"step={self.step_idx} post_startup_frame={self.training_frame_idx} "
                    "skipped training: no valid 6-DoF samples passed clearance filters."
                )
            self._maybe_finish_for_roi_coverage(checker)
            self._maybe_finish_for_field_eval(checker)
            if self.training_finished:
                self._publish_frontiers()
                return

        if is_executing:
            if self.step_idx % 5 == 0:
                remain_s = self.trajectory_busy_until - time.monotonic()
                self.get_logger().info(
                    f"step={self.step_idx} motion in progress: active_frontiers={len(active_all)} "
                    f"remaining_s={max(0.0, remain_s):.2f}"
                )
            self._publish_frontiers()
            return

        joint_valid, joint_age, joint_state_reason = self._joint_state_is_valid_for_planning()
        if not joint_valid:
            if self.step_idx % 5 == 0:
                self.get_logger().warn(
                    f"step={self.step_idx} skipping goal selection because joint state is stale: "
                    f"age_s={joint_age:.3f} max_age_s={self.max_joint_state_age_s:.3f} "
                    f"reason={joint_state_reason}"
                )
            self.latest_goal_meta = None
            self._publish_frontiers()
            return

        nbv_checker = checker
        nbv_checker_build_ms = 0.0
        if self.nbv_clearance_backend != self.clearance_backend:
            if (
                fast_clearance_checker is not None
                and self.nbv_clearance_backend == self.sampling_clearance_backend
            ):
                nbv_checker = fast_clearance_checker
            else:
                checker_t0 = time.perf_counter()
                nbv_checker = self._make_collision_checker(
                    occupied_points,
                    clearance_backend=self.nbv_clearance_backend,
                )
                nbv_checker.free_keys = checker.free_keys
                nbv_checker_build_ms = (time.perf_counter() - checker_t0) * 1e3

        self.get_logger().info(
            f"step={self.step_idx} starting goal selection: "
            f"active_frontiers={len(active_all)} local_frontiers={len(local_frontiers)} "
            f"global_frontiers={len(global_frontiers)} replay_size={self.field_model.replay_size} "
            f"epochs_trained={self.field_model.total_epochs_trained} "
            f"nbv_backend={self.nbv_clearance_backend} checker={type(nbv_checker).__name__} "
            f"checker_build_ms={nbv_checker_build_ms:.1f} "
            f"support_points_filtered={getattr(nbv_checker, 'ignored_support_point_count', 0)}"
        )

        current_camera_xyz = camera_pose[:3, 3].copy() if camera_pose is not None else None
        ranked_goals = []
        rejected_frontier_ids: set[int] = set()
        if (
            self.enable_object_roi_nbv
            and self.object_roi_nbv_first
            and self.frontier_roi_min is not None
            and self.frontier_roi_max is not None
            and self.current_joints is not None
        ):
            roi_rank_t0 = time.perf_counter()
            roi_goals = self.goal_selector.ranked_roi_candidates(
                self.frontier_roi_min,
                self.frontier_roi_max,
                self.current_joints,
                nbv_checker,
                camera_info=info_msg,
                current_camera_xyz=current_camera_xyz,
                max_candidates=8,
            )
            ranked_goals.extend(roi_goals)
            roi_rank_t1 = time.perf_counter()
            roi_dbg = getattr(self.goal_selector, "last_select_debug", {})
            self.get_logger().info(
                f"step={self.step_idx} object ROI NBV ranking complete: "
                f"fast_roi_nbv={self.fast_roi_nbv} "
                f"candidates_returned={len(roi_goals)} unknown_points={roi_dbg.get('unknown_points', 0)} "
                f"roi_targets={roi_dbg.get('roi_targets', 0)} "
                f"candidates={roi_dbg.get('candidates_total', 0)} "
                f"accepted={roi_dbg.get('accepted', 0)} "
                f"rejected_visibility={roi_dbg.get('rejected_visibility', 0)} "
                f"rejected_ik={roi_dbg.get('rejected_ik', 0)} "
                f"rejected_orientation={roi_dbg.get('rejected_orientation', 0)} "
                f"rejected_self_occlusion={roi_dbg.get('rejected_self_occlusion', 0)} "
                f"rank_ms={(roi_rank_t1 - roi_rank_t0) * 1e3:.1f}"
            )
        should_rank_frontiers = (len(ranked_goals) == 0) or self.rank_frontiers_when_roi_available
        if not should_rank_frontiers and (local_frontiers or global_frontiers):
            self.get_logger().info(
                f"step={self.step_idx} skipping local/global frontier ranking because ROI NBV returned "
                f"{len(ranked_goals)} candidate(s)."
            )
        if local_frontiers and should_rank_frontiers:
            rank_t0 = time.perf_counter()
            last_rank_log = [0.0]

            def _rank_progress_local(stats: dict):
                now = time.monotonic()
                if now - last_rank_log[0] < 1.0:
                    return
                last_rank_log[0] = now
                self.get_logger().info(
                    f"step={self.step_idx} ranking local frontiers: "
                    f"frontiers={stats.get('frontiers_considered', 0)} "
                    f"candidates={stats.get('candidates_total', 0)} "
                    f"accepted={stats.get('accepted', 0)}"
                )

            local_ranked = self.goal_selector.ranked_candidates(
                    local_frontiers,
                    self.current_joints,
                    nbv_checker,
                    camera_info=info_msg,
                    current_camera_xyz=current_camera_xyz,
                    step_idx=self.step_idx,
                    max_candidates=6,
                    max_frontiers=self.frontier_fallback_max_frontiers,
                    progress_cb=_rank_progress_local,
                )
            ranked_goals.extend(local_ranked)
            rejected_frontier_ids.update(
                int(fid)
                for fid in self.goal_selector.last_select_debug.get("rejected_frontier_ids", [])
            )
            rank_t1 = time.perf_counter()
            self.get_logger().info(
                f"step={self.step_idx} local frontier ranking complete: "
                f"local_frontiers={len(local_frontiers)} candidates_returned={len(ranked_goals)} "
                f"rank_ms={(rank_t1 - rank_t0) * 1e3:.1f}"
            )
        # A local fallback candidate is already a usable next view. Ranking a
        # second, disjoint global set only adds repeated IK/collision work and
        # previously accounted for several seconds of every failed ROI pass.
        should_rank_global = should_rank_frontiers and (
            not ranked_goals or self.rank_frontiers_when_roi_available
        )
        if global_frontiers and should_rank_global:
            rank_t0 = time.perf_counter()
            last_rank_log = [0.0]

            def _rank_progress_global(stats: dict):
                now = time.monotonic()
                if now - last_rank_log[0] < 1.0:
                    return
                last_rank_log[0] = now
                self.get_logger().info(
                    f"step={self.step_idx} ranking global frontiers: "
                    f"frontiers={stats.get('frontiers_considered', 0)} "
                    f"candidates={stats.get('candidates_total', 0)} "
                    f"accepted={stats.get('accepted', 0)}"
                )

            seen_frontier_ids = {int(goal.frontier_id) for goal in ranked_goals}
            global_ranked = self.goal_selector.ranked_candidates(
                global_frontiers,
                self.current_joints,
                nbv_checker,
                camera_info=info_msg,
                current_camera_xyz=current_camera_xyz,
                step_idx=self.step_idx,
                max_candidates=8,
                max_frontiers=self.frontier_fallback_max_frontiers,
                progress_cb=_rank_progress_global,
            )
            rejected_frontier_ids.update(
                int(fid)
                for fid in self.goal_selector.last_select_debug.get("rejected_frontier_ids", [])
            )
            for goal in global_ranked:
                if int(goal.frontier_id) in seen_frontier_ids and len(ranked_goals) >= 2:
                    continue
                ranked_goals.append(goal)
                seen_frontier_ids.add(int(goal.frontier_id))
            rank_t1 = time.perf_counter()
            self.get_logger().info(
                f"step={self.step_idx} global frontier ranking complete: "
                f"global_frontiers={len(global_frontiers)} candidates_returned={len(global_ranked)} "
                f"rank_ms={(rank_t1 - rank_t0) * 1e3:.1f}"
            )
        elif global_frontiers and should_rank_frontiers and ranked_goals:
            self.get_logger().info(
                f"step={self.step_idx} skipping global frontier ranking because local fallback produced "
                f"{len(ranked_goals)} candidate(s)."
            )
        ranked_goals.sort(key=lambda goal: goal.score, reverse=True)
        ranked_goals = self._filter_viewpoint_cooldown(ranked_goals)
        ranked_goals = ranked_goals[:10]
        if ranked_goals:
            dbg = getattr(self.goal_selector, "last_select_debug", {})
            self._attempt_goal_execution(ranked_goals, checker, joint_age, joint_state_reason)
        else:
            dbg = getattr(self.goal_selector, "last_select_debug", {})
            self.get_logger().info(
                f"step={self.step_idx} no feasible NBV goal selected from {len(active_all)} active frontiers."
            )
            retired_ids = []
            for frontier_id in sorted(rejected_frontier_ids):
                rec = self.frontier_bank.records.get(frontier_id)
                was_active = rec is not None and rec.status == "active"
                self.frontier_bank.mark_failed(frontier_id)
                rec = self.frontier_bank.records.get(frontier_id)
                if was_active and rec is not None and rec.status == "retired":
                    retired_ids.append(frontier_id)
            if rejected_frontier_ids:
                self.get_logger().info(
                    f"step={self.step_idx} NBV rejection accounting: failed_frontiers="
                    f"{sorted(rejected_frontier_ids)} retired={retired_ids}"
                )
            if dbg:
                self.get_logger().info(
                    f"step={self.step_idx} selector debug: frontiers={dbg.get('frontiers_considered', 0)} "
                    f"candidates={dbg.get('candidates_total', 0)} rejected_gain={dbg.get('rejected_gain', 0)} "
                    f"rejected_visibility={dbg.get('rejected_visibility', 0)} "
                    f"rejected_ik={dbg.get('rejected_ik', 0)} rejected_clearance={dbg.get('rejected_clearance', 0)} "
                    f"rejected_same_pose={dbg.get('rejected_same_pose', 0)} "
                    f"rejected_same_view={dbg.get('rejected_same_view', 0)} "
                    f"rejected_orientation={dbg.get('rejected_orientation', 0)} "
                    f"rejected_self_occlusion={dbg.get('rejected_self_occlusion', 0)} "
                    f"rejected_cooldown={dbg.get('rejected_cooldown', 0)}"
                )
            recovery_goals = self._filter_viewpoint_cooldown(
                self._bootstrap_recovery_goals(nbv_checker, info_msg, current_camera_xyz, active_all)
            )
            if not recovery_goals:
                # Task-space look-at poses can all fail single-seed IK near
                # cabinet edges even though useful nearby configurations are
                # reachable. Fall back to a bounded C-space lattice: FK makes
                # reachability exact, then the same actual-camera visibility
                # gates decide whether the view is useful.
                recovery_goals = self._filter_viewpoint_cooldown(
                    self._joint_space_recovery_goals(
                        nbv_checker, info_msg, current_camera_xyz, active_all
                    )
                )
            if recovery_goals:
                self.get_logger().info(
                    f"step={self.step_idx} attempting bootstrap recovery views: candidates={len(recovery_goals)}"
                )
                self._attempt_goal_execution(
                    recovery_goals,
                    checker,
                    joint_age,
                    joint_state_reason,
                    selection_mode_override="bootstrap_recovery",
                )
            else:
                dbg_boot = getattr(self, "last_bootstrap_debug", {})
                if dbg_boot:
                    self.get_logger().info(
                        f"step={self.step_idx} bootstrap recovery debug: "
                        f"candidates={dbg_boot.get('candidates_total', 0)} "
                        f"rejected_projection={dbg_boot.get('rejected_projection', 0)} "
                        f"rejected_ik={dbg_boot.get('rejected_ik', 0)} "
                        f"rejected_same_pose={dbg_boot.get('rejected_same_pose', 0)} "
                        f"rejected_clearance={dbg_boot.get('rejected_clearance', 0)} "
                        f"rejected_visibility={dbg_boot.get('rejected_visibility', 0)} "
                        f"rejected_self_occlusion={dbg_boot.get('rejected_self_occlusion', 0)} "
                        f"accepted={dbg_boot.get('accepted', 0)}"
                    )
                has_eval_minimums = (
                    self.field_model.total_epochs_trained >= self.field_eval_min_train_steps
                    and self.field_model.replay_size >= self.field_eval_min_replay_pairs
                )
                self.defer_training_until_motion = False
                self.defer_training_q = (
                    None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64).copy()
                )
                self.defer_training_publish_time = self.last_trajectory_publish_wall_time
                # If every candidate view is filtered out, keep improving the
                # field from the static scene immediately instead of waiting
                # for the normal frame cadence or a motion-triggered sample.
                self.training_frame_idx = 0
                self.latest_goal_meta = None
                if has_eval_minimums:
                    self.get_logger().info(
                        "No feasible motion goal is available; fixed-goal eval has not passed yet, "
                        "so static field training will continue."
                    )
                else:
                    self.get_logger().info(
                        "No feasible motion goal is available yet; continuing static field training until "
                        f"field-eval minimums are met: epochs={self.field_model.total_epochs_trained}/"
                        f"{self.field_eval_min_train_steps}, replay={self.field_model.replay_size}/"
                        f"{self.field_eval_min_replay_pairs}."
                    )
        if self.step_idx % 5 == 0:
            self.get_logger().info(
                f"step={self.step_idx} occupied={len(self.voxel_map.occupied)} "
                f"local_frontiers={len(local_frontiers)} global_frontiers={len(global_frontiers)} "
                f"active_frontiers={len(active_all)} visited_frontiers="
                f"{sum(1 for rec in self.frontier_bank.records.values() if rec.status == 'visited')} "
                f"plan_len={0 if self.last_plan is None else len(self.last_plan)}"
            )

        self._publish_frontiers()

    def _depth_to_world(self, depth: np.ndarray, info: CameraInfo, camera_pose: np.ndarray) -> np.ndarray:
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        rows = np.arange(0, depth.shape[0], self.raycast_stride_px)
        cols = np.arange(0, depth.shape[1], self.raycast_stride_px)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        z = depth[rr, cc]
        valid = np.isfinite(z) & (z >= self.depth_min_m) & (z <= self.depth_max_m)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)
        rr = rr[valid]
        cc = cc[valid]
        z = z[valid]
        x = (cc.astype(np.float32) - cx) * z / fx
        y = (rr.astype(np.float32) - cy) * z / fy
        pts_c = np.stack((x, y, z), axis=1)
        pts_w = (camera_pose[:3, :3] @ pts_c.T).T + camera_pose[:3, 3][None, :]
        return pts_w.astype(np.float32)

    def _filter_robot_self_points(self, points: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.enable_robot_self_filter or self.current_joints is None:
            return points, 0
        extra_spheres = self._tool_camera_self_filter_spheres(self.current_joints)
        return self.robot_self_filter_checker.filter_robot_self_points(
            points,
            np.asarray(self.current_joints, dtype=np.float64),
            padding_m=self.robot_self_filter_padding_m,
            extra_spheres=extra_spheres,
        )

    def _tool_camera_self_filter_spheres(self, q: np.ndarray) -> np.ndarray:
        tool_pose = self.kinematics.fk(np.asarray(q, dtype=np.float64))
        camera_pose = self.kinematics.tool_to_camera_pose(tool_pose, self.camera_in_tool)
        tool_xyz = np.asarray(tool_pose[:3, 3], dtype=np.float64)
        camera_xyz = np.asarray(camera_pose[:3, 3], dtype=np.float64)
        mount_xyz = 0.5 * (tool_xyz + camera_xyz)
        spheres = []
        if self.robot_self_filter_tool_radius_m > 0.0:
            spheres.append(np.r_[tool_xyz, self.robot_self_filter_tool_radius_m])
        if self.robot_self_filter_mount_radius_m > 0.0 and np.linalg.norm(camera_xyz - tool_xyz) > 1.0e-4:
            spheres.append(np.r_[mount_xyz, self.robot_self_filter_mount_radius_m])
        return np.asarray(spheres, dtype=np.float64).reshape(-1, 4)

    def _maybe_initialize_frontier_roi(self, occupied_points: np.ndarray):
        if not self.enable_frontier_roi_filter or self.frontier_roi_min is not None:
            return
        scene_bbox = self._scene_boxes_bbox()
        if scene_bbox is not None:
            self.frontier_roi_min, self.frontier_roi_max = scene_bbox
            self.frontier_roi_source = "scene_boxes"
            self.get_logger().info(
                "Initialized enclosed-space ROI from scene_boxes: "
                f"min={np.round(self.frontier_roi_min, 3).tolist()} "
                f"max={np.round(self.frontier_roi_max, 3).tolist()} "
                f"boxes={len(self.scene_boxes)}"
            )
            return
        pts = np.asarray(occupied_points, dtype=np.float64)
        if self.step_idx < self.frontier_roi_init_min_step or len(pts) < self.frontier_roi_init_min_points:
            return
        lo = np.min(pts, axis=0) - self.frontier_roi_padding_xyz
        hi = np.max(pts, axis=0) + self.frontier_roi_padding_xyz
        lo, hi = self._clip_roi_bounds(lo, hi)
        self.frontier_roi_min = lo
        self.frontier_roi_max = hi
        self.frontier_roi_source = "occupied_points"
        self.get_logger().info(
            "Initialized enclosed-space ROI: "
            f"min={np.round(self.frontier_roi_min, 3).tolist()} "
            f"max={np.round(self.frontier_roi_max, 3).tolist()} "
            f"from_points={len(pts)} step={self.step_idx}"
        )

    def _point_in_frontier_roi(self, xyz: np.ndarray) -> bool:
        if not self.enable_frontier_roi_filter or self.frontier_roi_min is None or self.frontier_roi_max is None:
            return True
        p = np.asarray(xyz, dtype=np.float64)
        return bool(np.all(p >= self.frontier_roi_min) and np.all(p <= self.frontier_roi_max))

    def _field_eval_targets_in_base(self) -> list[np.ndarray]:
        targets: list[np.ndarray] = []
        for xyz_src in self.field_eval_points_xyz:
            xyz_base = self._transform_point_between_frames(xyz_src, self.field_eval_points_frame, self.base_frame)
            if xyz_base is not None:
                targets.append(np.asarray(xyz_base, dtype=np.float64))
        return targets

    def _candidate_tool_point_goal_states(
        self,
        checker: UR5PointCloudCollisionChecker,
        q_start: np.ndarray,
        target_base: np.ndarray,
        max_candidates: int = 24,
    ) -> list[tuple[np.ndarray, float]]:
        target = np.asarray(target_base, dtype=np.float64).reshape(3)
        current_tool = self.kinematics.fk(q_start)
        base_rot = current_tool[:3, :3].copy()
        target_from_base = target / max(np.linalg.norm(target), 1e-8)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        approach_x = look_at_rotation(target - 0.20 * target_from_base, target, world_up)
        approach_neg_x = look_at_rotation(target + 0.20 * target_from_base, target, world_up)
        orientation_bases = [
            base_rot,
            approach_x @ _rot_y(-np.pi / 2.0),
            approach_x @ _rot_y(np.pi / 2.0),
            approach_neg_x @ _rot_y(-np.pi / 2.0),
            approach_neg_x @ _rot_y(np.pi / 2.0),
        ]
        seen: set[tuple[float, ...]] = set()
        candidates: list[tuple[np.ndarray, float, float]] = []
        for orient_base in orientation_bases:
            for yaw in self.goal_tool_orientation_yaw_offsets_rad:
                for pitch in self.goal_tool_orientation_pitch_offsets_rad:
                    for roll in self.goal_tool_orientation_roll_offsets_rad:
                        desired_tool = np.eye(4, dtype=np.float64)
                        desired_tool[:3, :3] = orient_base @ _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
                        desired_tool[:3, 3] = target
                        q_goal = self.kinematics.solve_ik_full(desired_tool, q_start)
                        if q_goal is None:
                            continue
                        q_key = tuple(np.round(q_goal, 4).tolist())
                        if q_key in seen:
                            continue
                        seen.add(q_key)
                        clearance = float(checker.clearance_batch(np.asarray([q_goal], dtype=np.float32))[0])
                        if not np.isfinite(clearance) or clearance < self.field_eval_goal_clearance_min_m:
                            continue
                        actual_tool = self.kinematics.fk(q_goal)
                        pos_err = float(np.linalg.norm(actual_tool[:3, 3] - target))
                        tool_forward = actual_tool[:3, 0]
                        alignment = float(np.dot(tool_forward, target_from_base))
                        if alignment < self.field_eval_goal_tool_forward_alignment_min:
                            continue
                        move_cost = float(np.linalg.norm(q_goal - q_start))
                        score = 2.0 * alignment + 1.5 * clearance - 0.45 * move_cost - 4.0 * pos_err
                        candidates.append((np.asarray(q_goal, dtype=np.float64), float(clearance), float(score)))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return [(q_goal, clearance) for q_goal, clearance, _score in candidates[:max_candidates]]

    def _field_eval_anchor_qs(self, checker: UR5PointCloudCollisionChecker) -> np.ndarray:
        q_ref = (
            np.asarray(self.startup_positions, dtype=np.float64)
            if self.field_eval_use_startup_pose or self.current_joints is None
            else np.asarray(self.current_joints, dtype=np.float64)
        )
        anchors: list[np.ndarray] = [q_ref.copy()]
        cache_valid = (
            len(self.field_eval_anchor_cache_qs) > 0
            and (self.step_idx - self.field_eval_anchor_cache_step) <= 20
            and self.field_eval_use_startup_pose
        )
        if cache_valid:
            clearances = checker.clearance_batch(self.field_eval_anchor_cache_qs.astype(np.float32))
            valid = np.isfinite(clearances) & (clearances >= self.field_eval_goal_clearance_min_m)
            anchors.extend([q.copy() for q in self.field_eval_anchor_cache_qs[valid]])
        else:
            computed: list[np.ndarray] = []
            for target in self._field_eval_targets_in_base():
                for q_goal, _clearance in self._candidate_tool_point_goal_states(checker, q_ref, target, max_candidates=4):
                    computed.append(np.asarray(q_goal, dtype=np.float64))
            if computed:
                self.field_eval_anchor_cache_qs = np.asarray(computed, dtype=np.float64)
                self.field_eval_anchor_cache_step = int(self.step_idx)
                anchors.extend([q.copy() for q in self.field_eval_anchor_cache_qs])
        if not anchors:
            return np.zeros((0, 6), dtype=np.float64)
        dedup: list[np.ndarray] = []
        for q in anchors:
            if not any(np.max(np.abs(q - ref)) < 1.0e-3 for ref in dedup):
                dedup.append(q)
        return np.asarray(dedup, dtype=np.float64)

    def _evaluate_field_goal_library(self, checker: UR5PointCloudCollisionChecker) -> dict[str, object]:
        if len(self.field_eval_points_xyz) == 0 and len(self.field_eval_joint_goals) == 0:
            result: dict[str, object] = {"success_ratio": 0.0, "success_count": 0, "evaluated": 0, "goals": []}
            self.last_field_eval = result
            return result
        q_start = (
            np.asarray(self.startup_positions, dtype=np.float64)
            if self.field_eval_use_startup_pose or self.current_joints is None
            else np.asarray(self.current_joints, dtype=np.float64)
        )
        goals_out: list[dict[str, object]] = []
        success_count = 0
        evaluated = 0
        targets_base = self._field_eval_targets_in_base()
        if len(self.field_eval_joint_goals) > 0:
            goal_specs = [
                (None, [(np.asarray(q, dtype=np.float64), float(checker.clearance(q)))])
                for q in self.field_eval_joint_goals
            ]
        else:
            cached_candidates = self._field_eval_goal_candidates(checker, q_start)
            goal_specs = [
                (target, cached_candidates[i] if i < len(cached_candidates) else [])
                for i, target in enumerate(targets_base)
            ]
        for idx, (target_base, candidates) in enumerate(goal_specs, start=1):
            best_goal_clearance = -1.0
            best_path_clearance = -1.0
            best_debug: dict[str, object] = {}
            success = False
            for q_goal, goal_clearance in candidates:
                if self.field_eval_collision_aware_rollout:
                    plan = self.planner.plan_collision_aware(
                        checker,
                        q_start,
                        q_goal,
                        self.step_size_q,
                        self.rollout_max_steps,
                        clearance_margin_m=self.trajectory_collision_margin_m,
                        max_local_candidates=self.field_local_rollout_candidates,
                    )
                    debug = dict(getattr(self.planner, "last_debug", {}) or {})
                else:
                    plan = self.planner.plan_learned_speed_search(
                        q_start,
                        q_goal,
                        self.step_size_q,
                        self.rollout_max_steps,
                        min_predicted_speed=0.10,
                        max_local_candidates=self.field_local_rollout_candidates,
                        allow_direct_edge=False,
                        mode="bidirectional",
                    )
                    debug = dict(getattr(self.planner, "last_debug", {}) or {})
                path_ok, min_clearance = self._validate_plan_collision(checker, plan)
                best_goal_clearance = max(best_goal_clearance, float(goal_clearance))
                best_path_clearance = max(best_path_clearance, float(min_clearance))
                if (
                    not best_debug
                    or len(plan) > int(best_debug.get("raw_waypoints", 0))
                    or float(debug.get("last_goal_dist", float("inf"))) < float(best_debug.get("last_goal_dist", float("inf")))
                ):
                    debug["raw_waypoints"] = int(len(plan))
                    debug["path_clearance_m"] = float(min_clearance)
                    best_debug = debug
                if path_ok:
                    success = True
                    success_count += 1
                    break
                if len(plan) > 0:
                    self._append_failed_rollout_anchors(checker, plan)
                    self._append_path_training_anchors(plan)
            if candidates:
                evaluated += 1
            goals_out.append(
                {
                    "goal_index": idx,
                    "target_base_xyz": None if target_base is None else np.round(target_base, 3).tolist(),
                    "target_q": None if not candidates else np.round(candidates[0][0], 3).tolist(),
                    "candidate_count": len(candidates),
                    "success": success,
                    "best_goal_clearance_m": best_goal_clearance,
                    "best_path_clearance_m": best_path_clearance,
                    "best_debug": best_debug,
                }
            )
        success_ratio = 0.0 if evaluated <= 0 else float(success_count) / float(evaluated)
        result = {
            "success_ratio": success_ratio,
            "success_count": int(success_count),
            "evaluated": int(evaluated),
            "goals": goals_out,
        }
        self.last_field_eval = result
        return result

    def _field_eval_goal_candidates(
        self, checker: UR5PointCloudCollisionChecker, q_start: np.ndarray
    ) -> list[list[tuple[np.ndarray, float]]]:
        targets = self._field_eval_targets_in_base()
        cache_valid = (
            self.field_eval_use_startup_pose
            and len(self.field_eval_goal_candidate_cache) == len(targets)
            and (self.step_idx - self.field_eval_goal_candidate_cache_step) <= 20
        )
        if cache_valid:
            refreshed: list[list[tuple[np.ndarray, float]]] = []
            for cached in self.field_eval_goal_candidate_cache:
                if not cached:
                    refreshed.append([])
                    continue
                qs = np.asarray([q for q, _clearance in cached], dtype=np.float32)
                clearances = checker.clearance_batch(qs)
                valid_candidates: list[tuple[np.ndarray, float]] = []
                for (q, _old_clearance), clearance in zip(cached, clearances):
                    clearance_f = float(clearance)
                    if np.isfinite(clearance_f) and clearance_f >= self.field_eval_goal_clearance_min_m:
                        valid_candidates.append((np.asarray(q, dtype=np.float64), clearance_f))
                refreshed.append(valid_candidates)
            return refreshed

        computed: list[list[tuple[np.ndarray, float]]] = []
        for target_base in targets:
            computed.append(self._candidate_tool_point_goal_states(checker, q_start, target_base, max_candidates=12))
        if self.field_eval_use_startup_pose:
            self.field_eval_goal_candidate_cache = [
                [(np.asarray(q, dtype=np.float64), float(clearance)) for q, clearance in candidates]
                for candidates in computed
            ]
            self.field_eval_goal_candidate_cache_step = int(self.step_idx)
        return computed

    def _field_diagnostics_gate_passed(self) -> tuple[bool, str]:
        if not self.field_diag_gate_enabled:
            return True, "disabled"
        if self.field_model.replay_size < self.field_diag_gate_min_replay_pairs:
            return (
                False,
                f"replay_size={self.field_model.replay_size} "
                f"min_required={self.field_diag_gate_min_replay_pairs}",
            )
        diag = dict(self.field_model.last_diagnostics or {})
        if float(diag.get("diag_rows", 0.0)) <= 0.0:
            diag = self.field_model.evaluate_replay_diagnostics(self.field_diagnostics_max_rows)
        failures: list[str] = []
        speed_corr = float(diag.get("speed_corr", 0.0))
        near_far_gap = float(diag.get("near_far_gap", 0.0))
        normal_cos_near = float(diag.get("normal_cos_near_mean", 0.0))
        low_count = int(diag.get("low_target_count", 0.0))
        low_overpred = float(diag.get("low_target_overpred_frac", 0.0))
        if speed_corr < self.field_diag_min_speed_corr:
            failures.append(f"speed_corr={speed_corr:.3f}<{self.field_diag_min_speed_corr:.3f}")
        if near_far_gap < self.field_diag_min_near_far_gap:
            failures.append(f"near_far_gap={near_far_gap:.3f}<{self.field_diag_min_near_far_gap:.3f}")
        normal_supervised = (
            self.field_normal_loss_weight > 0.0 or self.field_normal_cos_loss_weight > 0.0
        )
        if normal_supervised and normal_cos_near < self.field_diag_min_normal_cos_near:
            failures.append(
                f"normal_cos_near={normal_cos_near:.3f}<{self.field_diag_min_normal_cos_near:.3f}"
            )
        if low_count >= 64 and low_overpred > self.field_diag_max_low_overpred_frac:
            failures.append(
                f"low_overpred={low_overpred:.3f}>{self.field_diag_max_low_overpred_frac:.3f}"
            )
        if failures:
            return False, ", ".join(failures)
        return (
            True,
            f"speed_corr={speed_corr:.3f}, near_far_gap={near_far_gap:.3f}, "
            f"normal_cos_near={normal_cos_near:.3f}, low_overpred={low_overpred:.3f}",
        )

    def _field_false_free_audit(
        self, checker: UR5PointCloudCollisionChecker
    ) -> tuple[bool, str]:
        """Check held-out C-space states for unsafe free-space predictions.

        Replay diagnostics only measure interpolation on samples already seen by
        the optimizer.  A model can therefore have good aggregate correlation
        while assigning a high speed to an unsampled colliding configuration.
        This audit uses fresh, goal-independent joint states, exact geometric
        clearances, and independent random goals.  Failed states are retained as
        hard anchors so subsequent online updates explicitly train the missed
        obstacle region.
        """
        if not self.field_false_free_audit_enabled:
            return True, "disabled"
        if self.last_false_free_audit_step == self.step_idx:
            cached = dict(self.last_false_free_audit)
            return bool(cached.get("passed", 0.0)), str(cached.get("reason", "cached"))

        count = int(self.field_false_free_audit_samples)
        # Do not perturb the exploration/sampling RNG.  Varying the seed by map
        # step gives a new held-out audit whenever training is reconsidered.
        audit_rng = np.random.default_rng(104729 + int(self.step_idx))
        q = audit_rng.uniform(
            self.kinematics.joint_min,
            self.kinematics.joint_max,
            size=(count, 6),
        ).astype(np.float64)
        goals_per_state = int(self.field_false_free_audit_goals_per_state)
        goal_q = audit_rng.uniform(
            self.kinematics.joint_min,
            self.kinematics.joint_max,
            size=(count * goals_per_state, 6),
        ).astype(np.float64)

        clearance_parts: list[np.ndarray] = []
        for start in range(0, count, 128):
            clearance_parts.append(
                np.asarray(checker.clearance_batch(q[start : start + 128]), dtype=np.float32)
            )
        clearances = np.concatenate(clearance_parts) if clearance_parts else np.zeros((0,), dtype=np.float32)

        margin = max(float(self.clearance_margin_m), 1.0e-6)
        offset = float(np.clip(self.clearance_offset_m, 0.0, margin - 1.0e-6))
        alpha = np.clip(clearances, offset, margin) / margin
        alpha = alpha ** max(1.0e-6, float(self.clearance_label_power))
        floor = float(np.clip(self.clearance_label_floor, 0.0, 1.0))
        target = floor + (1.0 - floor) * alpha

        qn = self.kinematics.normalize(q)
        query_qn = np.repeat(qn, goals_per_state, axis=0)
        goal_qn = self.kinematics.normalize(goal_q)
        pred, _ = self.field_model.predict_normalized_pair_speeds(query_qn, goal_qn, batch_size=1024)
        query_clearances = np.repeat(clearances, goals_per_state)
        query_target = np.repeat(target, goals_per_state)
        finite = np.isfinite(query_clearances) & np.isfinite(query_target) & np.isfinite(pred)
        low = finite & (query_target <= float(self.field_false_free_target_speed_max))
        false_free = low & (pred >= float(self.field_false_free_pred_speed_min))
        low_state_count = int(
            np.count_nonzero(np.isfinite(clearances) & np.isfinite(target) & (target <= float(self.field_false_free_target_speed_max)))
        )
        low_pair_count = int(np.count_nonzero(low))
        false_free_count = int(np.count_nonzero(false_free))
        false_free_rate = float(false_free_count) / float(max(1, low_pair_count))

        enough_low = low_state_count >= int(self.field_false_free_min_low_states)
        passed = bool(enough_low and false_free_rate <= float(self.field_false_free_max_rate))
        if not enough_low:
            reason = (
                f"low_states={low_state_count}<{self.field_false_free_min_low_states} "
                f"from audit_samples={count}"
            )
        else:
            reason = (
                f"false_free={false_free_count}/{low_pair_count} rate={false_free_rate:.3f} "
                f"low_states={low_state_count} goals_per_state={goals_per_state} "
                f"max={self.field_false_free_max_rate:.3f} "
                f"target<={self.field_false_free_target_speed_max:.2f} "
                f"pred>={self.field_false_free_pred_speed_min:.2f}"
            )

        if false_free_count > 0:
            failed_pair_idx = np.flatnonzero(false_free)
            error = pred[failed_pair_idx] - query_target[failed_pair_idx]
            ordered_pairs = failed_pair_idx[np.argsort(error)[::-1]]
            # Retain at most one copy of a state even if several independent
            # goals expose the same false-free prediction.
            worst_state_idx = []
            seen_state_idx: set[int] = set()
            for pair_idx in ordered_pairs:
                state_idx = int(pair_idx) // goals_per_state
                if state_idx in seen_state_idx:
                    continue
                seen_state_idx.add(state_idx)
                worst_state_idx.append(state_idx)
                if len(worst_state_idx) >= 64:
                    break
            anchors = q[np.asarray(worst_state_idx, dtype=np.int64)]
            merged = (
                anchors
                if len(self.hard_failed_anchor_qs) == 0
                else np.vstack((self.hard_failed_anchor_qs, anchors))
            )
            if len(merged) > self.hard_failed_anchor_buffer_limit:
                merged = merged[-self.hard_failed_anchor_buffer_limit :]
            self.hard_failed_anchor_qs = merged.astype(np.float64, copy=False)

        self.last_false_free_audit_step = int(self.step_idx)
        self.last_false_free_audit = {
            "passed": float(passed),
            "audit_samples": float(count),
            "audit_pairs": float(count * goals_per_state),
            "low_states": float(low_state_count),
            "low_pairs": float(low_pair_count),
            "false_free_count": float(false_free_count),
            "false_free_rate": float(false_free_rate),
            "reason": reason,
        }
        return passed, reason

    def _adaptive_training_ready(self) -> bool:
        if not self.adaptive_training_enabled:
            return False
        if not self.field_diag_gate_enabled:
            return bool(
                self.field_model.total_epochs_trained >= self.field_eval_min_train_steps
                and self.field_model.replay_size >= self.field_eval_min_replay_pairs
                and float(self.last_field_eval.get("success_ratio", 0.0)) >= self.field_eval_success_ratio_threshold
            )
        ok, _reason = self._field_diagnostics_gate_passed()
        return bool(ok)

    def _active_training_schedule(self) -> tuple[int, int, int, int, int, bool]:
        ready = self._adaptive_training_ready()
        if not ready:
            return (
                int(self.sample_pairs_per_step),
                int(self.train_epochs_per_step),
                max(1, int(self.train_every_n_frames)),
                int(self.replay_recombine_pairs_per_step),
                int(self.hard_failed_pairs_per_step),
                False,
            )
        return (
            int(self.field_ready_sample_pairs_per_step),
            int(self.field_ready_train_epochs_per_step),
            int(self.field_ready_train_every_n_frames),
            int(self.field_ready_replay_recombine_pairs_per_step),
            int(self.field_ready_hard_failed_pairs_per_step),
            True,
        )

    def _filter_clusters_to_frontier_roi(self, clusters):
        if not self.enable_frontier_roi_filter or self.frontier_roi_min is None:
            return clusters
        return [cluster for cluster in clusters if self._point_in_frontier_roi(cluster.centroid)]

    def _active_frontiers_in_roi(self):
        active = self.frontier_bank.active_records()
        if not self.enable_frontier_roi_filter or self.frontier_roi_min is None:
            return active
        return [rec for rec in active if self._point_in_frontier_roi(rec.centroid)]

    def _roi_coverage_stats(self, checker: UR5PointCloudCollisionChecker) -> dict[str, float | int]:
        if self.frontier_roi_min is None or self.frontier_roi_max is None:
            return {"coverage": 0.0, "known": 0, "unknown": 0, "total": 0}
        voxel = float(checker.voxel_size)
        lo_key = np.floor(np.asarray(self.frontier_roi_min, dtype=np.float64) / voxel).astype(int)
        hi_key = np.floor(np.asarray(self.frontier_roi_max, dtype=np.float64) / voxel).astype(int)
        boxes = np.asarray(
            getattr(checker, "box_obstacles", np.zeros((0, 6))), dtype=np.float64
        ).reshape(-1, 6)
        box_centers = boxes[:, :3]
        box_half_extents = 0.5 * boxes[:, 3:]
        known = 0
        unknown = 0
        for ix in range(int(lo_key[0]), int(hi_key[0]) + 1):
            for iy in range(int(lo_key[1]), int(hi_key[1]) + 1):
                for iz in range(int(lo_key[2]), int(hi_key[2]) + 1):
                    key = (ix, iy, iz)
                    if len(boxes):
                        center = np.asarray(checker.key_to_center(key), dtype=np.float64)
                        inside_static_geometry = np.any(
                            np.all(np.abs(center - box_centers) <= box_half_extents, axis=1)
                        )
                        if inside_static_geometry:
                            continue
                    if key in checker.free_keys or key in checker.occupied_keys:
                        known += 1
                    else:
                        unknown += 1
        total = known + unknown
        coverage = 0.0 if total <= 0 else float(known) / float(total)
        return {"coverage": coverage, "known": known, "unknown": unknown, "total": total}

    def _maybe_finish_for_roi_coverage(self, checker: UR5PointCloudCollisionChecker):
        stats = self._roi_coverage_stats(checker)
        self.last_roi_coverage = stats
        if self.step_idx <= 3 or self.step_idx % 5 == 0:
            self.get_logger().info(
                f"step={self.step_idx} ROI coverage: "
                f"coverage={float(stats['coverage']):.3f} known={int(stats['known'])} "
                f"unknown={int(stats['unknown'])} total={int(stats['total'])}"
            )
        if not self.finish_when_roi_covered or self.training_finished:
            return
        total = int(stats["total"])
        if total <= 0:
            return
        coverage = float(stats["coverage"])
        unknown = int(stats["unknown"])
        coverage_ready = (coverage >= self.roi_coverage_threshold) or (unknown <= self.roi_unknown_stop_voxels)
        if not coverage_ready:
            return
        if self.field_model.total_epochs_trained < self.field_eval_min_train_steps:
            self.get_logger().info(
                f"ROI coverage is ready but deferring finish until minimum training epochs are met: "
                f"epochs_trained={self.field_model.total_epochs_trained} min_required={self.field_eval_min_train_steps}"
            )
            return
        if self.field_model.replay_size < self.field_eval_min_replay_pairs:
            self.get_logger().info(
                f"ROI coverage is ready but deferring finish until enough accumulated pairs are collected: "
                f"replay_size={self.field_model.replay_size} min_required={self.field_eval_min_replay_pairs}"
            )
            return
        if not self.finish_when_field_eval_passes:
            self._finish_training(
                f"ROI coverage reached coverage={coverage:.3f}, unknown_voxels={unknown}, "
                f"threshold={self.roi_coverage_threshold:.3f}"
            )
            return
        diag_ok, diag_reason = self._field_diagnostics_gate_passed()
        if not diag_ok:
            self.get_logger().info(
                "ROI coverage is ready but deferring finish until field diagnostics improve: "
                f"{diag_reason}"
            )
            return
        audit_ok, audit_reason = self._field_false_free_audit(checker)
        self.get_logger().info(f"step={self.step_idx} held-out false-free audit: {audit_reason}")
        if not audit_ok:
            self.get_logger().info(
                "ROI coverage is ready but unsafe held-out field predictions were hard-mined; "
                "continuing training."
            )
            return
        if self.step_idx % self.field_eval_every_n_steps != 0:
            return
        eval_result = self._evaluate_field_goal_library(checker)
        success_ratio = float(eval_result.get("success_ratio", 0.0))
        success_count = int(eval_result.get("success_count", 0))
        evaluated = int(eval_result.get("evaluated", 0))
        self.get_logger().info(
            f"step={self.step_idx} field eval: success_ratio={success_ratio:.3f} "
            f"success_count={success_count}/{evaluated} "
            f"threshold={self.field_eval_success_ratio_threshold:.3f}"
        )
        for goal_info in eval_result.get("goals", []):
            if not isinstance(goal_info, dict):
                continue
            debug = goal_info.get("best_debug", {})
            if not isinstance(debug, dict):
                debug = {}
            self.get_logger().info(
                f"step={self.step_idx} field eval goal {int(goal_info.get('goal_index', -1))}: "
                f"target={goal_info.get('target_base_xyz', [])} candidates={int(goal_info.get('candidate_count', 0))} "
                f"success={bool(goal_info.get('success', False))} "
                f"best_goal_clearance_m={float(goal_info.get('best_goal_clearance_m', -1.0)):.4f} "
                f"best_path_clearance_m={float(goal_info.get('best_path_clearance_m', -1.0)):.4f} "
                f"rollout_status={debug.get('status', '')} rollout_steps={int(debug.get('steps', 0))} "
                f"last_goal_dist={float(debug.get('last_goal_dist', -1.0)):.3f} "
                f"valid_edges={int(debug.get('valid_edge_count', 0))}/{int(debug.get('candidate_count', 0))} "
                f"start_clearance_m={float(debug.get('start_clearance', -1.0)):.4f} "
                f"best_cand_clearance_m={float(debug.get('best_candidate_clearance', -1.0)):.4f} "
                f"best_edge_clearance_m={float(debug.get('best_edge_min_clearance', -1.0)):.4f}"
            )
        if success_ratio >= self.field_eval_success_ratio_threshold:
            self._finish_training(
                f"ROI coverage reached and field eval passed: coverage={coverage:.3f}, "
                f"unknown_voxels={unknown}, eval_success={success_count}/{evaluated}"
            )
        else:
            self.get_logger().info(
                f"ROI coverage reached but field eval has not passed yet: "
                f"coverage={coverage:.3f}, eval_success={success_count}/{evaluated}"
            )

    def _maybe_finish_for_field_eval(self, checker: UR5PointCloudCollisionChecker):
        if self.training_finished or not self.finish_when_field_eval_passes:
            return
        if self.field_model.total_epochs_trained < self.field_eval_min_train_steps:
            return
        if self.field_model.replay_size < self.field_eval_min_replay_pairs:
            return
        diag_ok, diag_reason = self._field_diagnostics_gate_passed()
        if not diag_ok:
            self.get_logger().info(
                "Field-eval finish is ready but diagnostics have not passed yet: "
                f"{diag_reason}"
            )
            return
        audit_ok, audit_reason = self._field_false_free_audit(checker)
        self.get_logger().info(f"step={self.step_idx} held-out false-free audit: {audit_reason}")
        if not audit_ok:
            self.get_logger().info(
                "Field-eval finish is blocked by unsafe held-out predictions; "
                "the worst states were added to hard-example training."
            )
            return
        if self.step_idx % self.field_eval_every_n_steps != 0:
            return
        eval_result = self._evaluate_field_goal_library(checker)
        success_ratio = float(eval_result.get("success_ratio", 0.0))
        success_count = int(eval_result.get("success_count", 0))
        evaluated = int(eval_result.get("evaluated", 0))
        self.get_logger().info(
            f"step={self.step_idx} field eval without ROI finish: success_ratio={success_ratio:.3f} "
            f"success_count={success_count}/{evaluated} "
            f"threshold={self.field_eval_success_ratio_threshold:.3f}"
        )
        for goal_info in eval_result.get("goals", []):
            if not isinstance(goal_info, dict):
                continue
            debug = goal_info.get("best_debug", {})
            if not isinstance(debug, dict):
                debug = {}
            self.get_logger().info(
                f"step={self.step_idx} field eval goal {int(goal_info.get('goal_index', -1))}: "
                f"target={goal_info.get('target_base_xyz', [])} candidates={int(goal_info.get('candidate_count', 0))} "
                f"success={bool(goal_info.get('success', False))} "
                f"best_goal_clearance_m={float(goal_info.get('best_goal_clearance_m', -1.0)):.4f} "
                f"best_path_clearance_m={float(goal_info.get('best_path_clearance_m', -1.0)):.4f} "
                f"rollout_status={debug.get('status', '')} rollout_steps={int(debug.get('steps', 0))} "
                f"last_goal_dist={float(debug.get('last_goal_dist', -1.0)):.3f} "
                f"valid_edges={int(debug.get('valid_edge_count', 0))}/{int(debug.get('candidate_count', 0))} "
                f"start_clearance_m={float(debug.get('start_clearance', -1.0)):.4f} "
                f"best_cand_clearance_m={float(debug.get('best_candidate_clearance', -1.0)):.4f} "
                f"best_edge_clearance_m={float(debug.get('best_edge_min_clearance', -1.0)):.4f}"
            )
        if evaluated > 0 and success_ratio >= self.field_eval_success_ratio_threshold:
            self._finish_training(
                f"field eval passed before ROI coverage stop: eval_success={success_count}/{evaluated}"
            )

    def _bootstrap_focus_point(self, active_frontiers: list) -> np.ndarray | None:
        if active_frontiers:
            ranked = sorted(active_frontiers, key=lambda rec: rec.voxel_count, reverse=True)[:6]
            pts = np.asarray([rec.centroid for rec in ranked], dtype=np.float64)
            if len(pts):
                return np.mean(pts, axis=0)
        if self.frontier_roi_min is not None and self.frontier_roi_max is not None:
            return 0.5 * (self.frontier_roi_min + self.frontier_roi_max)
        return None

    def _bootstrap_aim_points(self, active_frontiers: list) -> list[np.ndarray]:
        aims: list[np.ndarray] = []
        focus = self._bootstrap_focus_point(active_frontiers)
        if focus is not None:
            aims.append(np.asarray(focus, dtype=np.float64))
        if self.frontier_roi_min is not None and self.frontier_roi_max is not None:
            center = 0.5 * (self.frontier_roi_min + self.frontier_roi_max)
            size = self.frontier_roi_max - self.frontier_roi_min
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            aims.extend(
                [
                    center,
                    center + 0.2 * size[0] * x_axis,
                    center - 0.2 * size[0] * x_axis,
                    center + 0.2 * size[2] * z_axis,
                    center - 0.2 * size[2] * z_axis,
                ]
            )
        uniq: list[np.ndarray] = []
        for a in aims:
            if not any(np.linalg.norm(a - b) < 0.03 for b in uniq):
                uniq.append(a)
        return uniq

    def _bootstrap_recovery_goals(
        self,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None,
        current_camera_xyz: np.ndarray | None,
        active_frontiers: list,
    ) -> list[ViewGoal]:
        stats = {
            "candidates_total": 0,
            "rejected_projection": 0,
            "rejected_ik": 0,
            "rejected_same_pose": 0,
            "rejected_clearance": 0,
            "rejected_visibility": 0,
            "rejected_self_occlusion": 0,
            "accepted": 0,
        }
        self.last_bootstrap_debug = stats
        if (
            not self.enable_bootstrap_recovery
            or self.current_joints is None
            or current_camera_xyz is None
        ):
            return []
        aim_points = self._bootstrap_aim_points(active_frontiers)
        if not aim_points:
            return []
        current_camera_xyz = np.asarray(current_camera_xyz, dtype=np.float64)
        focus = aim_points[0]
        view_vec = np.asarray(focus, dtype=np.float64) - current_camera_xyz
        dist = float(np.linalg.norm(view_vec))
        if dist < 1e-6:
            return []
        view_dir = view_vec / dist
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(view_dir, up)
        right /= max(np.linalg.norm(right), 1e-6)
        true_up = np.cross(right, view_dir)
        true_up /= max(np.linalg.norm(true_up), 1e-6)

        seed_frontier_id = -1
        if active_frontiers:
            seed_frontier_id = int(min(
                active_frontiers,
                key=lambda rec: float(np.linalg.norm(rec.centroid.astype(np.float64) - focus)),
            ).frontier_id)
        candidates: list[ViewGoal] = []
        offsets = [
            np.zeros(3, dtype=np.float64),
            self.bootstrap_recovery_lateral_m * right,
            -self.bootstrap_recovery_lateral_m * right,
            self.bootstrap_recovery_vertical_m * true_up,
            -self.bootstrap_recovery_vertical_m * true_up,
            -self.bootstrap_recovery_radius_m * view_dir,
            self.bootstrap_recovery_lateral_m * right - self.bootstrap_recovery_radius_m * view_dir,
            -self.bootstrap_recovery_lateral_m * right - self.bootstrap_recovery_radius_m * view_dir,
        ]
        bootstrap_candidate_budget = 24
        for focus in aim_points:
            focus = np.asarray(focus, dtype=np.float64)
            for offset in offsets:
                if stats["candidates_total"] >= bootstrap_candidate_budget:
                    break
                stats["candidates_total"] += 1
                cam_pos = current_camera_xyz + offset
                # Bootstrap should always face the ROI directly rather than using the broader NBV orientation cone.
                cam_rot = look_at_rotation(cam_pos, focus)
                cam_pose = _transform(cam_rot, cam_pos)
                if self.goal_selector._project_to_image(cam_pose, focus, camera_info) is None:
                    stats["rejected_projection"] += 1
                    continue
                tool_pose = self.kinematics.camera_to_tool_pose(cam_pose, self.camera_in_tool)
                # Full IK may try more than 1000 nonlinear restart seeds for
                # each unreachable pose. Across this 48-pose recovery sweep it
                # caused multi-minute callback stalls. Keep task-space
                # recovery bounded; the reachability-first joint lattice below
                # covers useful views that fast IK cannot realize.
                q_goal = self.kinematics.solve_ik_fast(
                    tool_pose, np.asarray(self.current_joints, dtype=np.float64)
                )
                if q_goal is None:
                    stats["rejected_ik"] += 1
                    continue
                if float(np.max(np.abs(q_goal - self.current_joints))) < self.min_goal_joint_delta_rad:
                    stats["rejected_same_pose"] += 1
                    continue
                clearance = checker.clearance(q_goal)
                if clearance <= 0.0:
                    stats["rejected_clearance"] += 1
                    continue
                # IK is approximate and may converge to a different wrist
                # orientation than the requested look-at pose. Validate and
                # retain the pose FK says the robot will actually reach. The
                # old bootstrap path scored/stored ``cam_pose`` directly, so a
                # joint solution whose optical axis faced away from the ROI
                # could still pass every projection/visibility test.
                actual_tool_pose = self.kinematics.fk(q_goal)
                actual_cam_pose = self.kinematics.tool_to_camera_pose(
                    actual_tool_pose, self.camera_in_tool
                )
                actual_cam_pos = np.asarray(actual_cam_pose[:3, 3], dtype=np.float64)
                target_ok, reject_reason = self.goal_selector._actual_target_view_is_valid(
                    checker,
                    q_goal,
                    actual_cam_pose,
                    focus,
                    camera_info,
                )
                if not target_ok:
                    if reject_reason == "self_occlusion":
                        stats["rejected_self_occlusion"] += 1
                    else:
                        stats["rejected_visibility"] += 1
                    continue
                self_free_fraction = self.goal_selector._self_occlusion_free_fraction(
                    checker, q_goal, actual_cam_pos, focus.reshape(1, 3)
                )
                if self_free_fraction < self.min_view_self_occlusion_free_fraction:
                    stats["rejected_self_occlusion"] += 1
                    continue
                target_visibility = self.goal_selector._ray_clearance_score(
                    checker, actual_cam_pos, focus, collision_radius_m=0.04
                )
                if target_visibility <= 0.02:
                    stats["rejected_visibility"] += 1
                    continue
                move_cost = float(np.linalg.norm(q_goal - self.current_joints))
                local_context = 0.0
                visible_neighbors = 0
                for rec in active_frontiers:
                    target = rec.centroid.astype(np.float64)
                    if float(np.linalg.norm(target - focus)) > self.target_context_radius_m:
                        continue
                    if self.goal_selector._project_to_image(actual_cam_pose, target, camera_info) is None:
                        continue
                    if self.goal_selector._ray_clearance_score(
                        checker, actual_cam_pos, target, collision_radius_m=0.04
                    ) <= 0.0:
                        continue
                    visible_neighbors += 1
                if visible_neighbors > 0:
                    local_context = min(1.0, 0.25 * visible_neighbors)
                score = (
                    0.85 * target_visibility
                    + 0.15 * local_context
                    - 0.25 * move_cost
                    + 0.35 * min(float(clearance), 0.20)
                )
                candidates.append(
                    ViewGoal(
                        frontier_id=seed_frontier_id,
                        centroid=focus.copy(),
                        camera_pose=actual_cam_pose.copy(),
                        tool_pose=actual_tool_pose.copy(),
                        q_goal=q_goal.copy(),
                        score=float(score),
                        pose_kind="bootstrap",
                        visibility_score=float(target_visibility),
                        local_coverage=float(local_context),
                        gain_score=0.0,
                        move_cost=move_cost,
                        clearance=float(clearance),
                    )
                )
                stats["accepted"] += 1
            if stats["candidates_total"] >= bootstrap_candidate_budget:
                break
        candidates.sort(key=lambda goal: goal.score, reverse=True)
        return candidates[:6]

    def _joint_space_recovery_goals(
        self,
        checker: UR5PointCloudCollisionChecker,
        camera_info: CameraInfo | None,
        current_camera_xyz: np.ndarray | None,
        active_frontiers: list,
    ) -> list[ViewGoal]:
        """Return useful, guaranteed-reachable views from a local C-space lattice."""
        stats = {
            "candidates_total": 0,
            "rejected_same_pose": 0,
            "rejected_clearance": 0,
            "rejected_visibility": 0,
            "rejected_self_occlusion": 0,
            "accepted": 0,
        }
        self.last_joint_recovery_debug = stats
        if self.current_joints is None or camera_info is None:
            return []

        q_base = np.asarray(self.current_joints, dtype=np.float64).reshape(6)
        targets = self._bootstrap_aim_points(active_frontiers)
        ranked_frontiers = sorted(active_frontiers, key=lambda rec: rec.voxel_count, reverse=True)
        targets.extend(np.asarray(rec.centroid, dtype=np.float64) for rec in ranked_frontiers[:8])
        unique_targets: list[np.ndarray] = []
        for target in targets:
            target = np.asarray(target, dtype=np.float64).reshape(3)
            if not any(np.linalg.norm(target - prior) < 0.03 for prior in unique_targets):
                unique_targets.append(target)

        deltas: list[np.ndarray] = []
        for axis, magnitudes in (
            (0, (0.20, 0.40, 0.60)),
            (1, (0.18, 0.35)),
            (2, (0.18, 0.35)),
            (3, (0.25, 0.50)),
            (4, (0.25, 0.50)),
            (5, (0.30, 0.60)),
        ):
            for magnitude in magnitudes:
                for sign in (-1.0, 1.0):
                    delta = np.zeros((6,), dtype=np.float64)
                    delta[axis] = sign * magnitude
                    deltas.append(delta)
        # Coupled shoulder/wrist changes preserve position better than a large
        # single-joint move while sweeping the camera optical axis.
        for shoulder in (-0.40, 0.40):
            for wrist in (-0.45, 0.45):
                delta = np.zeros((6,), dtype=np.float64)
                delta[0] = shoulder
                delta[4] = wrist
                deltas.append(delta)
                delta2 = delta.copy()
                delta2[1] = -0.20 if shoulder > 0.0 else 0.20
                deltas.append(delta2)

        candidates: list[ViewGoal] = []
        for delta in deltas:
            stats["candidates_total"] += 1
            q_goal = self.kinematics.clamp(q_base + delta)
            move_cost = float(np.linalg.norm(q_goal - q_base))
            if move_cost < self.min_goal_joint_delta_rad:
                stats["rejected_same_pose"] += 1
                continue
            clearance = float(checker.clearance(q_goal))
            if clearance <= max(0.0, self.trajectory_collision_margin_m):
                stats["rejected_clearance"] += 1
                continue
            tool_pose = self.kinematics.fk(q_goal)
            cam_pose = self.kinematics.tool_to_camera_pose(tool_pose, self.camera_in_tool)
            cam_pos = np.asarray(cam_pose[:3, 3], dtype=np.float64)
            if current_camera_xyz is not None and np.linalg.norm(
                cam_pos - np.asarray(current_camera_xyz, dtype=np.float64)
            ) < self.min_camera_goal_delta_m:
                stats["rejected_same_pose"] += 1
                continue

            best_target = None
            best_alignment = -1.0
            best_visibility = 0.0
            for target in unique_targets:
                ray = target - cam_pos
                ray_norm = float(np.linalg.norm(ray))
                if ray_norm < 1.0e-6:
                    continue
                alignment = float(np.dot(cam_pose[:3, 2], ray / ray_norm))
                if alignment < self.min_actual_view_alignment:
                    continue
                if self.goal_selector._project_to_image(cam_pose, target, camera_info) is None:
                    continue
                visibility = self.goal_selector._ray_clearance_score(
                    checker, cam_pos, target, collision_radius_m=0.035
                )
                if visibility <= 0.0:
                    continue
                if alignment + 0.35 * visibility > best_alignment + 0.35 * best_visibility:
                    best_target = target
                    best_alignment = alignment
                    best_visibility = visibility
            if best_target is None:
                stats["rejected_visibility"] += 1
                continue
            self_free = self.goal_selector._self_occlusion_free_fraction(
                checker, q_goal, cam_pos, best_target.reshape(1, 3)
            )
            if self_free < self.min_view_self_occlusion_free_fraction:
                stats["rejected_self_occlusion"] += 1
                continue

            visible_frontiers = 0
            for rec in active_frontiers:
                if self.goal_selector._project_to_image(
                    cam_pose, rec.centroid.astype(np.float64), camera_info
                ) is not None:
                    visible_frontiers += 1
            local_coverage = float(visible_frontiers) / float(max(1, len(active_frontiers)))
            score = (
                0.65 * best_visibility
                + 0.35 * best_alignment
                + 0.30 * local_coverage
                + 0.25 * min(clearance, 0.25)
                - 0.20 * move_cost
            )
            nearest_frontier_id = -1
            if active_frontiers:
                nearest_frontier_id = int(min(
                    active_frontiers,
                    key=lambda rec: float(
                        np.linalg.norm(rec.centroid.astype(np.float64) - best_target)
                    ),
                ).frontier_id)
            candidates.append(
                ViewGoal(
                    frontier_id=nearest_frontier_id,
                    centroid=best_target.copy(),
                    camera_pose=cam_pose.copy(),
                    tool_pose=tool_pose.copy(),
                    q_goal=np.asarray(q_goal, dtype=np.float64).copy(),
                    score=float(score),
                    pose_kind="joint_recovery",
                    visibility_score=float(best_visibility),
                    local_coverage=float(local_coverage),
                    gain_score=float(visible_frontiers),
                    move_cost=move_cost,
                    clearance=clearance,
                )
            )
            stats["accepted"] += 1
        candidates.sort(key=lambda goal: goal.score, reverse=True)
        if candidates:
            self.get_logger().info(
                f"step={self.step_idx} joint-space recovery: candidates={stats['candidates_total']} "
                f"accepted={stats['accepted']} rejected_visibility={stats['rejected_visibility']} "
                f"rejected_clearance={stats['rejected_clearance']} "
                f"rejected_self_occlusion={stats['rejected_self_occlusion']}"
            )
        return candidates[:6]

    def _prune_viewpoint_cooldown(self):
        if self.viewpoint_cooldown_steps <= 0 or not self.recent_viewpoints:
            self.recent_viewpoints = []
            return
        min_step = int(self.step_idx) - int(self.viewpoint_cooldown_steps)
        self.recent_viewpoints = [
            (step, xyz)
            for step, xyz in self.recent_viewpoints
            if int(step) >= min_step
        ]

    def _viewpoint_on_cooldown(self, camera_xyz: np.ndarray) -> bool:
        if self.viewpoint_cooldown_steps <= 0 or self.viewpoint_cooldown_radius_m <= 0.0:
            return False
        self._prune_viewpoint_cooldown()
        xyz = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
        for _step, prev_xyz in self.recent_viewpoints:
            if float(np.linalg.norm(xyz - np.asarray(prev_xyz, dtype=np.float64))) <= self.viewpoint_cooldown_radius_m:
                return True
        return False

    def _record_viewpoint_cooldown(self, camera_xyz: np.ndarray):
        if self.viewpoint_cooldown_steps <= 0 or self.viewpoint_cooldown_radius_m <= 0.0:
            return
        self._prune_viewpoint_cooldown()
        self.recent_viewpoints.append((int(self.step_idx), np.asarray(camera_xyz, dtype=np.float64).reshape(3).copy()))

    def _filter_viewpoint_cooldown(self, goals: list[ViewGoal]) -> list[ViewGoal]:
        if self.viewpoint_cooldown_steps <= 0 or self.viewpoint_cooldown_radius_m <= 0.0:
            return goals
        out = []
        rejected = 0
        for goal in goals:
            if self._viewpoint_on_cooldown(goal.camera_pose[:3, 3]):
                rejected += 1
                continue
            out.append(goal)
        if rejected:
            self.get_logger().info(
                f"step={self.step_idx} viewpoint cooldown rejected {rejected}/{len(goals)} candidates "
                f"radius_m={self.viewpoint_cooldown_radius_m:.2f} steps={self.viewpoint_cooldown_steps}"
            )
        if not out and goals:
            self.get_logger().info(
                f"step={self.step_idx} viewpoint cooldown covered all candidates; "
                "keeping them rejected to prevent two-view oscillation."
            )
            return []
        return out

    def _attempt_goal_execution(
        self,
        goals: list[ViewGoal],
        checker: UR5PointCloudCollisionChecker,
        joint_age: float,
        joint_state_reason: str,
        selection_mode_override: str | None = None,
    ) -> bool:
        if not goals:
            return False
        for cand_idx, goal in enumerate(goals, start=1):
            selection_mode = selection_mode_override or ("nbv" if goal.score > 0.0 else "fallback")
            self.latest_goal_meta = {
                "frontier_id": goal.frontier_id,
                "score": goal.score,
                "pose_kind": goal.pose_kind,
                "visibility_score": goal.visibility_score,
                "local_coverage": goal.local_coverage,
                "gain_score": goal.gain_score,
                "clearance": goal.clearance,
                "move_cost": goal.move_cost,
                "centroid": goal.centroid.tolist(),
                "mode": selection_mode,
                "camera_goal_xyz": goal.camera_pose[:3, 3].tolist(),
                "q_start": np.asarray(self.current_joints, dtype=np.float64).tolist(),
                "q_goal": goal.q_goal.tolist(),
            }
            self.get_logger().info(
                f"Selected frontier {goal.frontier_id} ({selection_mode}) candidate={cand_idx}/{len(goals)}: "
                f"score={goal.score:.4f}, visibility={goal.visibility_score:.4f}, "
                f"local_coverage={goal.local_coverage:.2f}, pose_kind={goal.pose_kind}, "
                f"centroid={np.round(goal.centroid, 3).tolist()}, "
                f"q_goal={np.round(goal.q_goal, 3).tolist()}"
            )
            self.get_logger().info(
                f"Planning from current_q={np.round(self.current_joints, 3).tolist()} "
                f"to q_goal={np.round(goal.q_goal, 3).tolist()} "
                f"camera_goal_xyz={np.round(goal.camera_pose[:3, 3], 3).tolist()} "
                f"joint_state_age_s={joint_age:.3f} joint_state_mode={joint_state_reason} "
                f"clearance_m={goal.clearance:.3f} move_cost={goal.move_cost:.3f}"
            )
            try:
                plan_t0 = time.perf_counter()
                if self.execution_planner == "field":
                    if self.collision_aware_field_rollout:
                        self.last_plan = self.planner.plan_collision_aware(
                            checker,
                            self.current_joints,
                            goal.q_goal,
                            self.step_size_q,
                            self.rollout_max_steps,
                            clearance_margin_m=self.trajectory_collision_margin_m,
                            max_local_candidates=self.field_local_rollout_candidates,
                        )
                        planner_name = "field_local"
                    else:
                        self.last_plan = self.planner.plan(
                            self.current_joints, goal.q_goal, self.step_size_q, self.rollout_max_steps
                        )
                        planner_name = "field"
                elif self.execution_planner in ("field_then_rrt", "field_rrt", "field_fallback"):
                    planner_name = "rrt_connect"
                    self.last_plan = None
                    if self.field_model.total_epochs_trained > 0:
                        field_plan = self.planner.plan(
                            self.current_joints, goal.q_goal, self.step_size_q, self.rollout_max_steps
                        )
                        field_ok, field_clearance = self._validate_plan_collision(checker, field_plan)
                        if field_ok:
                            self.last_plan = field_plan
                            planner_name = "field"
                        else:
                            self._append_failed_rollout_anchors(checker, field_plan)
                            self.get_logger().info(
                                f"Field plan rejected before RRT fallback: frontier={goal.frontier_id} "
                                f"min_clearance_m={field_clearance:.4f} "
                                f"required_margin_m={self.trajectory_collision_margin_m:.4f} "
                                f"hard_failed_anchor_buffer={len(self.hard_failed_anchor_qs)}"
                            )
                            if self.collision_aware_field_rollout:
                                local_plan = self.planner.plan_collision_aware(
                                    checker,
                                    self.current_joints,
                                    goal.q_goal,
                                    self.step_size_q,
                                    self.rollout_max_steps,
                                    clearance_margin_m=self.trajectory_collision_margin_m,
                                    max_local_candidates=self.field_local_rollout_candidates,
                                )
                                local_ok, local_clearance = self._validate_plan_collision(checker, local_plan)
                                if local_ok:
                                    self.last_plan = local_plan
                                    planner_name = "field_local"
                                elif len(local_plan) > 0:
                                    self._append_failed_rollout_anchors(checker, local_plan)
                                    self.get_logger().info(
                                        f"Local field plan rejected before RRT fallback: frontier={goal.frontier_id} "
                                        f"min_clearance_m={local_clearance:.4f} "
                                        f"required_margin_m={self.trajectory_collision_margin_m:.4f} "
                                        f"hard_failed_anchor_buffer={len(self.hard_failed_anchor_qs)}"
                                    )
                    if self.last_plan is None:
                        self.last_plan = self.exec_planner.plan(
                            checker,
                            self.current_joints,
                            goal.q_goal,
                            step_size_q=self.step_size_q,
                            max_iters=self.rrt_max_iters,
                            goal_bias=self.rrt_goal_bias,
                            clearance_margin_m=self.trajectory_collision_margin_m,
                        )
                        planner_name = "rrt_connect"
                else:
                    self.last_plan = self.exec_planner.plan(
                        checker,
                        self.current_joints,
                        goal.q_goal,
                        step_size_q=self.step_size_q,
                        max_iters=self.rrt_max_iters,
                        goal_bias=self.rrt_goal_bias,
                        clearance_margin_m=self.trajectory_collision_margin_m,
                    )
                    planner_name = "rrt_connect"
                plan_t1 = time.perf_counter()
            except Exception as exc:
                self.frontier_bank.mark_failed(goal.frontier_id)
                self.get_logger().warn(f"Planner failed for frontier {goal.frontier_id}: {exc}")
                self.last_plan = None
                continue
            if self.last_plan is None or len(self.last_plan) == 0:
                self.frontier_bank.mark_failed(goal.frontier_id)
                self.get_logger().warn(
                    f"Planner failed to find a path for frontier {goal.frontier_id} using {planner_name}. Trying next candidate."
                )
                self.last_plan = None
                continue
            self.frontier_bank.mark_selected(goal.frontier_id, self.step_idx)
            self.get_logger().info(
                f"Planner [{planner_name}] produced joint-space path with {len(self.last_plan)} waypoints "
                f"(step_size_q={self.step_size_q:.3f}, rollout_max_steps={self.rollout_max_steps}, "
                f"plan_ms={(plan_t1 - plan_t0) * 1e3:.1f})."
            )
            path_ok, min_clearance = self._validate_plan_collision(checker, self.last_plan)
            if not path_ok:
                self.frontier_bank.mark_failed(goal.frontier_id)
                self.get_logger().warn(
                    f"Rejected planned path for frontier {goal.frontier_id}: "
                    f"min_clearance_m={min_clearance:.4f} "
                    f"required_margin_m={self.trajectory_collision_margin_m:.4f}. "
                    f"Trying next candidate."
                )
                self.last_plan = None
                continue
            self.defer_training_until_motion = False
            self.defer_training_q = None
            self.defer_training_publish_time = 0.0
            exec_plan = self._prepare_execution_plan(
                self.last_plan,
                None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64),
            )
            exec_plan = self._truncate_execution_prefix(
                exec_plan,
                None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64),
            )
            exec_ok, exec_min_clearance = self._validate_plan_collision(checker, exec_plan)
            if not exec_ok:
                self.frontier_bank.mark_failed(goal.frontier_id)
                self._append_failed_rollout_anchors(checker, exec_plan)
                self.get_logger().warn(
                    f"Rejected post-processed execution path for frontier {goal.frontier_id}: "
                    f"min_clearance_m={exec_min_clearance:.4f} "
                    f"required_margin_m={self.trajectory_collision_margin_m:.4f}."
                )
                self.last_plan = None
                continue
            execution_reaches_view_goal = bool(
                len(exec_plan)
                and np.max(np.abs(np.asarray(exec_plan[-1], dtype=np.float64) - np.asarray(goal.q_goal, dtype=np.float64)))
                <= 1.0e-3
            )
            if execution_reaches_view_goal:
                self._record_viewpoint_cooldown(goal.camera_pose[:3, 3])
            else:
                self.get_logger().info(
                    "Execution prefix stops before the selected NBV pose; keeping that viewpoint eligible until its "
                    "post-IK camera pose is actually reached."
                )
            self._append_path_training_anchors(self.last_plan)
            self._publish_planned_path(exec_plan)
            if self.enable_trajectory_publish:
                self._publish_joint_trajectory(exec_plan, already_prepared=True)
                if self.strict_train_move_cycle and self.current_joints is not None:
                    self.defer_training_until_motion = True
                    self.defer_training_q = np.asarray(self.current_joints, dtype=np.float64).copy()
                    self.defer_training_publish_time = self.last_trajectory_publish_wall_time
                    self.training_frame_idx = 0
                    self.get_logger().info(
                        "Strict train/move cycle enabled: waiting for executed motion before collecting the next training batch."
                    )
            self._write_status()
            return True
        self.defer_training_until_motion = True
        self.defer_training_q = np.asarray(self.current_joints, dtype=np.float64).copy()
        self.defer_training_publish_time = self.last_trajectory_publish_wall_time
        self.latest_goal_meta = None
        return False

    def _append_path_training_anchors(self, plan: np.ndarray):
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            return
        stride = max(1, self.path_anchor_stride)
        anchors = pts[::stride].copy()
        if len(anchors) == 0 or np.max(np.abs(anchors[-1] - pts[-1])) > 1e-5:
            anchors = np.vstack((anchors, pts[-1])) if len(anchors) else pts[-1: ].copy()
        merged = anchors if len(self.path_training_anchor_qs) == 0 else np.vstack((self.path_training_anchor_qs, anchors))
        if len(merged) > self.path_anchor_buffer_limit:
            merged = merged[-self.path_anchor_buffer_limit :]
        self.path_training_anchor_qs = merged.astype(np.float64, copy=False)

    def _dense_plan_points(self, plan: np.ndarray) -> np.ndarray:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        dense = [pts[0].copy()]
        max_delta_step = max(0.02, 0.5 * self.step_size_q)
        for q in pts[1:]:
            prev = dense[-1]
            max_delta = float(np.max(np.abs(q - prev)))
            nseg = max(1, int(np.ceil(max_delta / max_delta_step)))
            for alpha in np.linspace(1.0 / nseg, 1.0, nseg):
                dense.append(prev + alpha * (q - prev))
        return np.asarray(dense, dtype=np.float64)

    def _append_failed_rollout_anchors(self, checker: UR5PointCloudCollisionChecker, plan: np.ndarray):
        dense = self._dense_plan_points(plan)
        if len(dense) == 0:
            return
        clearances = checker.clearance_batch(dense.astype(np.float32))
        if clearances.size == 0:
            return
        window = max(self.trajectory_collision_margin_m, self.hard_failed_clearance_window_m)
        candidate_idx = np.where(clearances <= window)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.argsort(clearances)[: min(8, len(clearances))]
        if len(candidate_idx) == 0:
            return
        order = candidate_idx[np.argsort(clearances[candidate_idx])]
        keep_count = min(64, len(order))
        anchors = dense[order[:keep_count]].copy()
        merged = anchors if len(self.hard_failed_anchor_qs) == 0 else np.vstack((self.hard_failed_anchor_qs, anchors))
        if len(merged) > self.hard_failed_anchor_buffer_limit:
            merged = merged[-self.hard_failed_anchor_buffer_limit :]
        self.hard_failed_anchor_qs = merged.astype(np.float64, copy=False)

    def _hard_failed_training_rows(
        self,
        checker: UR5PointCloudCollisionChecker,
        pair_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        target_pairs = self.hard_failed_pairs_per_step if pair_count is None else int(pair_count)
        if target_pairs <= 0 or len(self.hard_failed_anchor_qs) == 0:
            stats = self._empty_aux_sampler_stats("hard_failed_rollout")
            return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), stats
        # Preserve exact failed rollout states, including configurations on or
        # inside the obstacle boundary. The ordinary free-space sampler drops
        # d < clearance_offset states, which previously erased the most useful
        # evidence from a wall-crossing raw trajectory.
        barrier_count = min(max(1, target_pairs // 2), target_pairs)
        anchor_idx = self.rng.integers(0, len(self.hard_failed_anchor_qs), size=barrier_count)
        barrier_q = self.hard_failed_anchor_qs[anchor_idx]
        raw_barrier, rows_barrier, _barrier_stats = make_cspace_pair_rows_from_q_pairs(
            checker,
            self.kinematics,
            barrier_q,
            barrier_q,
            self.clearance_margin_m,
            0.0,
            clearance_label_floor=0.0,
            clearance_label_power=self.clearance_label_power,
            require_offset_clearance=False,
        )
        remaining = max(0, target_pairs - len(rows_barrier))
        raw_local, rows_local, stats = sample_path_centered_training_batch(
            checker,
            self.kinematics,
            self.hard_failed_anchor_qs,
            remaining,
            self.clearance_margin_m,
            self.clearance_offset_m,
            self.rng,
            clearance_label_floor=self.clearance_label_floor,
            clearance_label_power=self.clearance_label_power,
            proposal_batch_size=self.sampling_proposal_batch_size,
        )
        raw_parts = [part for part in (raw_barrier, raw_local) if len(part)]
        row_parts = [part for part in (rows_barrier, rows_local) if len(part)]
        raw = np.concatenate(raw_parts, axis=0).astype(np.float32, copy=False) if raw_parts else np.zeros((0, 12), dtype=np.float32)
        rows = np.concatenate(row_parts, axis=0).astype(np.float32, copy=False) if row_parts else np.zeros((0, 26), dtype=np.float32)
        stats["sampling_mode"] = "hard_failed_rollout:barrier+requeried"
        stats["accepted_pairs"] = float(len(rows))
        stats["barrier_pairs"] = float(len(rows_barrier))
        return raw, rows, stats

    def _empty_aux_sampler_stats(self, mode: str) -> dict[str, float]:
        return {
            "sampling_mode": mode,
            "attempts": 0.0,
            "ik_seed_tries": 0.0,
            "accepted_pairs": 0.0,
            "acceptance_rate": 0.0,
            "accepted_per_seed": 0.0,
            "samples_per_seed": 0.0,
            "refined_q0": 0.0,
            "refined_q1": 0.0,
            "anchor_seed_tries": 0.0,
            "anchor_seed_success": 0.0,
            "roi_seed_tries": 0.0,
            "workspace_seed_tries": 0.0,
            "roi_seed_success": 0.0,
            "workspace_seed_success": 0.0,
            "speed0_sat_frac": 0.0,
            "speed1_sat_frac": 0.0,
            "q0_clearance_mean": 0.0,
            "q0_clearance_min": 0.0,
            "q0_clearance_max": 0.0,
            "q0_near_margin_frac": 0.0,
            "q1_clearance_mean": 0.0,
            "q1_clearance_min": 0.0,
            "q1_clearance_max": 0.0,
            "q1_near_margin_frac": 0.0,
        }

    def _merge_sampler_stats(self, stats_list: list[dict[str, float]]) -> dict[str, float]:
        if not stats_list:
            return {}
        out: dict[str, float] = {}
        sum_keys = {
            "attempts", "ik_seed_tries", "accepted_pairs", "refined_q0", "refined_q1", "anchor_seed_tries", "anchor_seed_success",
            "roi_seed_tries", "workspace_seed_tries", "roi_seed_success", "workspace_seed_success",
        }
        mean_keys = {
            "speed0_sat_frac", "speed1_sat_frac", "speed0_low_frac", "speed1_low_frac",
            "speed0_critical_frac", "speed1_critical_frac", "q0_clearance_mean", "q0_clearance_min", "q0_clearance_max",
            "q0_near_margin_frac", "q1_clearance_mean", "q1_clearance_min", "q1_clearance_max",
            "q1_near_margin_frac", "q0_boundary_shell_frac", "q1_obstacle_side_frac",
        }
        total_attempts = sum(float(s.get("attempts", 0.0)) for s in stats_list)
        total_seeds = sum(float(s.get("ik_seed_tries", 0.0)) for s in stats_list)
        total_pairs = sum(float(s.get("accepted_pairs", 0.0)) for s in stats_list)
        for key in sum_keys:
            out[key] = sum(float(s.get(key, 0.0)) for s in stats_list)
        for key in mean_keys:
            weights = [float(s.get("accepted_pairs", 0.0)) for s in stats_list]
            vals = [float(s.get(key, 0.0)) for s in stats_list]
            denom = max(sum(weights), 1.0)
            out[key] = sum(v * w for v, w in zip(vals, weights)) / denom
        out["acceptance_rate"] = 0.0 if total_attempts <= 0 else total_pairs / total_attempts
        out["accepted_per_seed"] = 0.0 if total_seeds <= 0 else total_pairs / total_seeds
        out["samples_per_seed"] = sum(float(s.get("samples_per_seed", 0.0)) for s in stats_list)
        out["sampling_mode"] = "+".join(
            [str(s.get("sampling_mode", "")) for s in stats_list if str(s.get("sampling_mode", ""))]
        )
        return out

    def _publish_frontiers(self):
        arr = MarkerArray()
        marker = Marker()
        marker.header = Header(frame_id=self.visualization_frame)
        marker.ns = "frontiers"
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.85
        for rec in self.frontier_bank.active_records():
            p_vis = self._transform_point_for_visualization(rec.centroid)
            p = Point()
            p.x = float(p_vis[0])
            p.y = float(p_vis[1])
            p.z = float(p_vis[2])
            marker.points.append(p)
        arr.markers.append(marker)
        if self.frontier_roi_min is not None and self.frontier_roi_max is not None:
            roi = Marker()
            roi.header = Header(frame_id=self.visualization_frame)
            roi.ns = "frontier_roi"
            roi.id = 0
            roi.type = Marker.CUBE
            roi.action = Marker.ADD
            center = self._transform_point_for_visualization(0.5 * (self.frontier_roi_min + self.frontier_roi_max))
            size = self.frontier_roi_max - self.frontier_roi_min
            roi.pose.position.x = float(center[0])
            roi.pose.position.y = float(center[1])
            roi.pose.position.z = float(center[2])
            roi.pose.orientation.w = 1.0
            roi.scale.x = float(max(size[0], 1e-3))
            roi.scale.y = float(max(size[1], 1e-3))
            roi.scale.z = float(max(size[2], 1e-3))
            roi.color.r = 0.0
            roi.color.g = 0.8
            roi.color.b = 0.9
            roi.color.a = 0.08
            arr.markers.append(roi)
        if self.latest_goal_meta is not None:
            sel = self.frontier_bank.records.get(int(self.latest_goal_meta["frontier_id"]))
            if sel is not None:
                marker_sel = Marker()
                marker_sel.header = Header(frame_id=self.visualization_frame)
                marker_sel.ns = "selected_frontier"
                marker_sel.id = 2
                marker_sel.type = Marker.SPHERE
                marker_sel.action = Marker.ADD
                marker_sel.scale.x = 0.10
                marker_sel.scale.y = 0.10
                marker_sel.scale.z = 0.10
                marker_sel.color.r = 0.1
                marker_sel.color.g = 1.0
                marker_sel.color.b = 0.2
                marker_sel.color.a = 0.95
                sel_vis = self._transform_point_for_visualization(sel.centroid)
                marker_sel.pose.position.x = float(sel_vis[0])
                marker_sel.pose.position.y = float(sel_vis[1])
                marker_sel.pose.position.z = float(sel_vis[2])
                marker_sel.pose.orientation.w = 1.0
                arr.markers.append(marker_sel)
            if "camera_goal_xyz" in self.latest_goal_meta:
                marker_goal = Marker()
                marker_goal.header = Header(frame_id=self.visualization_frame)
                marker_goal.ns = "goal_camera"
                marker_goal.id = 3
                marker_goal.type = Marker.SPHERE
                marker_goal.action = Marker.ADD
                marker_goal.scale.x = 0.08
                marker_goal.scale.y = 0.08
                marker_goal.scale.z = 0.08
                marker_goal.color.r = 0.1
                marker_goal.color.g = 0.7
                marker_goal.color.b = 1.0
                marker_goal.color.a = 0.95
                goal_xyz = self._transform_point_for_visualization(
                    np.asarray(self.latest_goal_meta["camera_goal_xyz"], dtype=np.float64)
                )
                marker_goal.pose.position.x = float(goal_xyz[0])
                marker_goal.pose.position.y = float(goal_xyz[1])
                marker_goal.pose.position.z = float(goal_xyz[2])
                marker_goal.pose.orientation.w = 1.0
                arr.markers.append(marker_goal)
        self.frontier_pub.publish(arr)

    def _publish_debug_point_cloud(self, points_world: np.ndarray, stamp):
        msg = PointCloud2()
        points, frame_id = self._transform_points_for_visualization(points_world)
        msg.header = Header(frame_id=frame_id, stamp=self.get_clock().now().to_msg())
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        self.debug_points_pub.publish(msg)

    def _publish_joint_trajectory(self, plan: np.ndarray, *, already_prepared: bool = False):
        if plan is None or len(plan) == 0:
            return
        msg = JointTrajectory()
        msg.header = Header(frame_id=self.base_frame)
        msg.joint_names = JOINT_NAMES
        current_q = None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64)
        t_cur = 0.5
        exec_plan = np.asarray(plan, dtype=np.float64) if already_prepared else self._prepare_execution_plan(plan, current_q)
        if len(exec_plan) == 0:
            self.get_logger().warn("Skipping trajectory publish because the planned path had no finite execution waypoints.")
            return
        if not already_prepared:
            exec_plan = self._truncate_execution_prefix(exec_plan, current_q)
        if len(exec_plan) == 0:
            self.get_logger().warn("Skipping trajectory publish because no execution prefix remained after truncation.")
            return
        if current_q is not None:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in current_q]
            pt.time_from_start.sec = int(t_cur)
            pt.time_from_start.nanosec = int((t_cur - int(t_cur)) * 1e9)
            msg.points.append(pt)
        prev_q = current_q
        max_joint_speed = max(1e-3, self.trajectory_max_joint_speed)
        segment_dt_floor = 0.02
        for q in exec_plan:
            if prev_q is not None and np.max(np.abs(q - prev_q)) < 1e-4:
                continue
            if prev_q is None:
                dt = 0.5
            else:
                max_delta = float(np.max(np.abs(q - prev_q)))
                dt = max(segment_dt_floor, max_delta / max_joint_speed)
            t_cur += dt
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.time_from_start.sec = int(t_cur)
            pt.time_from_start.nanosec = int((t_cur - int(t_cur)) * 1e9)
            msg.points.append(pt)
            prev_q = q
        self.trajectory_pub.publish(msg)
        self.last_trajectory_publish_wall_time = time.monotonic()
        self.trajectory_busy_until = time.monotonic() + t_cur + self.trajectory_busy_margin_s
        self.get_logger().info(
            f"Published execution trajectory: exec_waypoints={len(msg.points)} "
            f"estimated_duration_s={t_cur:.1f} max_joint_speed={self.trajectory_max_joint_speed:.2f} "
            f"prefix_waypoints={len(exec_plan)}"
        )

    def _truncate_execution_prefix(self, exec_plan: np.ndarray, current_q: np.ndarray | None) -> np.ndarray:
        pts = np.asarray(exec_plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        max_points = int(self.execute_prefix_waypoints)
        if max_points <= 0:
            full_plan = pts.copy()
            if current_q is not None and len(full_plan):
                full_plan[0] = np.asarray(current_q, dtype=np.float64)
            return full_plan
        max_points = max(1, max_points)
        prefix = pts[:max_points].copy()
        if current_q is not None and len(prefix):
            prefix[0] = np.asarray(current_q, dtype=np.float64)
        if len(prefix) >= len(pts):
            return prefix
        min_duration = max(0.0, float(self.execute_prefix_min_duration_s))
        if min_duration <= 0.0:
            return prefix
        max_joint_speed = max(1e-3, self.trajectory_max_joint_speed)
        segment_dt_floor = 0.02
        accum = 0.0
        for idx in range(1, len(pts)):
            prev = pts[idx - 1]
            q = pts[idx]
            max_delta = float(np.max(np.abs(q - prev)))
            accum += max(segment_dt_floor, max_delta / max_joint_speed)
            if idx + 1 > len(prefix):
                prefix = pts[: idx + 1].copy()
            if accum >= min_duration:
                break
        if current_q is not None and len(prefix):
            prefix[0] = np.asarray(current_q, dtype=np.float64)
        return prefix

    def _prepare_execution_plan(self, plan: np.ndarray, current_q: np.ndarray | None) -> np.ndarray:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        finite_mask = np.all(np.isfinite(pts), axis=1)
        dropped = int(len(pts) - np.count_nonzero(finite_mask))
        if dropped:
            self.get_logger().warn(f"Execution plan contains {dropped} non-finite waypoint(s); dropping them before resampling.")
            pts = pts[finite_mask]
        if len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        if current_q is not None:
            current_q = np.asarray(current_q, dtype=np.float64)
            if np.all(np.isfinite(current_q)):
                pts[0] = current_q
            else:
                self.get_logger().warn("Current joint state is non-finite; using first finite plan waypoint as execution start.")
        sampled = [pts[0].copy()]
        max_delta_step = max(1e-3, self.trajectory_max_joint_speed * self.trajectory_min_segment_dt)
        for q in pts[1:]:
            prev = sampled[-1]
            if not np.all(np.isfinite(prev)) or not np.all(np.isfinite(q)):
                self.get_logger().warn("Skipping non-finite execution plan segment.")
                continue
            max_delta = float(np.max(np.abs(q - prev)))
            if not np.isfinite(max_delta):
                self.get_logger().warn("Skipping execution plan segment with non-finite joint delta.")
                continue
            nseg = max(1, int(np.ceil(max_delta / max_delta_step)))
            for alpha in np.linspace(1.0 / nseg, 1.0, nseg):
                qi = prev + alpha * (q - prev)
                if np.all(np.isfinite(qi)):
                    sampled.append(qi)
        sampled = np.asarray(sampled, dtype=np.float64)
        if sampled.ndim != 2 or len(sampled) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        stride = max(1, self.trajectory_waypoint_stride)
        sampled = sampled[::stride].copy()
        if not np.allclose(sampled[-1], pts[-1]):
            sampled = np.vstack((sampled, pts[-1]))
        window = max(1, self.trajectory_smoothing_window)
        if window > 1 and len(sampled) > 2:
            half = window // 2
            smoothed = sampled.copy()
            for idx in range(1, len(sampled) - 1):
                lo = max(0, idx - half)
                hi = min(len(sampled), idx + half + 1)
                smoothed[idx] = np.mean(sampled[lo:hi], axis=0)
            sampled = smoothed
        sampled = sampled[np.all(np.isfinite(sampled), axis=1)]
        if current_q is not None and len(sampled):
            sampled[0] = np.asarray(current_q, dtype=np.float64)
        return sampled

    def _validate_plan_collision(self, checker: UR5PointCloudCollisionChecker, plan: np.ndarray) -> tuple[bool, float]:
        dense_arr = self._dense_plan_points(plan).astype(np.float32)
        if len(dense_arr) == 0:
            return False, -1.0
        clearances = checker.clearance_batch(dense_arr)
        if clearances.size == 0:
            return False, -1.0
        min_clearance = float(np.min(clearances))
        return bool(min_clearance >= self.trajectory_collision_margin_m), min_clearance

    def _publish_planned_path(self, plan: np.ndarray):
        if plan is None or len(plan) == 0:
            return
        dense_plan = self._dense_plan_points(plan)
        if len(dense_plan) == 0:
            dense_plan = np.asarray(plan, dtype=np.float64)
        path = NavPath()
        path.header = Header(frame_id=self.visualization_frame if self.visualization_frame else self.base_frame)
        for q in np.asarray(dense_plan, dtype=np.float64):
            tool_pose = self.kinematics.fk(q)
            camera_pose = self.kinematics.tool_to_camera_pose(tool_pose, self.camera_in_tool)
            camera_xyz = self._transform_point_for_visualization(camera_pose[:3, 3])
            pose = PoseStamped()
            pose.header = Header(frame_id=self.visualization_frame)
            pose.pose.position.x = float(camera_xyz[0])
            pose.pose.position.y = float(camera_xyz[1])
            pose.pose.position.z = float(camera_xyz[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _save_step_artifacts(
        self,
        step_idx: int,
        depth_image_m: np.ndarray,
        camera_info: CameraInfo,
        camera_pose: np.ndarray,
        points_world: np.ndarray,
        raw_rows: np.ndarray,
        frame_rows: np.ndarray,
        occupied_points: np.ndarray,
        active_frontiers: list,
        loss: float | None,
    ):
        if step_idx % self.save_every_n_steps != 0:
            return
        np.savez(
            self.samples_dir / f"step_{step_idx:06d}.npz",
            raw_q_data=raw_rows,
            frame_data=frame_rows,
            occupied_points=occupied_points,
            current_q=self.current_joints.astype(np.float32),
            loss=np.array([-1.0 if loss is None else loss], dtype=np.float32),
            replay_size=np.array([self.field_model.replay_size], dtype=np.int32),
        )
        self._write_pcd(self.pcd_dir / f"step_{step_idx:06d}_depth_world.pcd", points_world)
        self._write_pcd(self.pcd_dir / f"step_{step_idx:06d}_occupied_world.pcd", occupied_points)
        self._save_debug_images(step_idx, depth_image_m, camera_info, camera_pose, active_frontiers)
        self.get_logger().info(
            f"Saved training view artifacts: step={step_idx} pairs={len(frame_rows)} "
            f"replay_size={self.field_model.replay_size} samples_dir={self.samples_dir}"
        )

    def _save_debug_images(
        self,
        step_idx: int,
        depth_image_m: np.ndarray,
        camera_info: CameraInfo,
        camera_pose: np.ndarray,
        active_frontiers: list,
    ):
        if self.latest_color is not None:
            color = self.latest_color.copy()
        else:
            norm = np.clip(depth_image_m, self.depth_min_m, self.depth_max_m)
            gray = ((norm - self.depth_min_m) / max(self.depth_max_m - self.depth_min_m, 1e-6) * 255.0).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        projected = self._project_frontiers_to_image(active_frontiers, camera_info, camera_pose)
        color_overlay = color.copy()
        depth_vis = np.clip(np.nan_to_num(depth_image_m, nan=0.0, posinf=0.0, neginf=0.0), self.depth_min_m, self.depth_max_m)
        depth_u8 = ((depth_vis - self.depth_min_m) / max(self.depth_max_m - self.depth_min_m, 1e-6) * 255.0).astype(np.uint8)
        depth_overlay = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        for px, py, frontier_id in projected:
            for image in (color_overlay, depth_overlay):
                cv2.circle(image, (px, py), 10, (0, 140, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    image,
                    str(frontier_id),
                    (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )
        cv2.imwrite(str(self.images_dir / f"step_{step_idx:06d}_color_frontiers.png"), color_overlay)
        cv2.imwrite(str(self.images_dir / f"step_{step_idx:06d}_depth_frontiers.png"), depth_overlay)

    def _project_frontiers_to_image(self, active_frontiers: list, camera_info: CameraInfo, camera_pose: np.ndarray) -> list[tuple[int, int, int]]:
        if not active_frontiers:
            return []
        cam_from_world = np.linalg.inv(camera_pose)
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        width, height = int(camera_info.width), int(camera_info.height)
        projected = []
        for rec in active_frontiers:
            p_world = np.ones((4,), dtype=np.float64)
            p_world[:3] = rec.centroid
            p_cam = cam_from_world @ p_world
            if p_cam[2] <= 1e-4:
                continue
            px = int(round(fx * (p_cam[0] / p_cam[2]) + cx))
            py = int(round(fy * (p_cam[1] / p_cam[2]) + cy))
            if 0 <= px < width and 0 <= py < height:
                projected.append((px, py, rec.frontier_id))
        return projected

    def _write_pcd(self, path: FsPath, points_xyz: np.ndarray):
        pts = np.asarray(points_xyz, dtype=np.float32)
        with path.open("wb") as f:
            header = (
                "# .PCD v0.7 - Point Cloud Data file format\n"
                "VERSION 0.7\n"
                "FIELDS x y z\n"
                "SIZE 4 4 4\n"
                "TYPE F F F\n"
                "COUNT 1 1 1\n"
                f"WIDTH {len(pts)}\n"
                "HEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\n"
                "DATA binary\n"
            )
            f.write(header.encode("ascii"))
            f.write(pts.tobytes())

    def _save_model_artifacts_if_needed(self):
        target_checkpoint_epoch = (
            self.field_model.total_epochs_trained // self.checkpoint_every_epochs
        ) * self.checkpoint_every_epochs
        if target_checkpoint_epoch <= self.field_model.last_checkpoint_epoch or target_checkpoint_epoch <= 0:
            return
        # Rendering a Matplotlib loss plot on every online collection cycle
        # adds CPU/GPU synchronization without changing the optimizer state.
        # Persist it alongside the checkpoint instead.
        self.field_model.save_loss_plot(self.model_artifacts_dir / "train_loss.png")
        checkpoint_path = self.model_artifacts_dir / f"weights_epoch_{target_checkpoint_epoch:06d}.pt"
        self.field_model.save_checkpoint(checkpoint_path)
        self.field_model.last_checkpoint_epoch = target_checkpoint_epoch
        self.get_logger().info(f"Saved model checkpoint: {checkpoint_path}")
        if self.enable_field_diagnostics:
            diag_dir = self.model_artifacts_dir / "field_diagnostics"
            try:
                saved = self.field_model.save_replay_diagnostic_plots(
                    diag_dir,
                    self.kinematics,
                    step_label=f"epoch_{target_checkpoint_epoch:06d}",
                    q_start=None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64),
                    max_rows=self.field_diagnostics_max_rows,
                    grid_size=self.field_diagnostics_grid_size,
                    include_joint_slices=self.field_diagnostics_save_joint_slices,
                )
                if saved:
                    self.get_logger().info(
                        f"Saved field diagnostic plots: dir={diag_dir} files={len(saved)}"
                    )
            except Exception as exc:
                self.get_logger().warn(f"Failed to save field diagnostic plots: {exc}")

    def _write_status(self):
        payload = {
            "step_idx": self.step_idx,
            "network_initialized": self.network_initialized,
            "training_finished": self.training_finished,
            "frontier_roi_source": self.frontier_roi_source,
            "roi_coverage": self.last_roi_coverage,
            "active_frontiers": len(self.frontier_bank.active_records()),
            "frontiers": {
                str(fid): {
                    "centroid": rec.centroid.tolist(),
                    "status": rec.status,
                    "voxel_count": rec.voxel_count,
                    "times_selected": rec.times_selected,
                    "times_failed": rec.times_failed,
                }
                for fid, rec in self.frontier_bank.records.items()
            },
            "latest_goal": self.latest_goal_meta,
            "last_plan_len": 0 if self.last_plan is None else int(len(self.last_plan)),
            "loss_history": self.field_model.loss_history[-20:],
            "total_epochs_trained": int(self.field_model.total_epochs_trained),
            "replay_size": int(self.field_model.replay_size),
            "replay_capacity": int(self.field_model.replay_capacity),
            "train_minibatch_size": int(self.field_model.minibatch_size),
            "effective_train_batch_size": int(self.field_model.effective_minibatch_size),
            "field_diagnostics_update_count": int(self.training_update_count),
            "field_diagnostics": dict(self.field_model.last_diagnostics),
            "field_eval": {
                "success_ratio": float(self.last_field_eval.get("success_ratio", 0.0)),
                "success_count": int(self.last_field_eval.get("success_count", 0)),
                "evaluated": int(self.last_field_eval.get("evaluated", 0)),
                "collision_aware_rollout": bool(self.field_eval_collision_aware_rollout),
            },
        }
        (self.output_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    rclpy.init()
    node = ArmMNTFieldsExplorer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
