from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import rclpy
import torch
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from ur_mntfields_arm.arm_field_model import ArmFieldModel
from ur_mntfields_arm.collision_checker import UR5PointCloudCollisionChecker, make_ur5_collision_checker
from ur_mntfields_arm.planner import ArmFieldPlanner
from ur_mntfields_arm.ur5_kinematics import JOINT_NAMES, UR5Kinematics, look_at_rotation


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


class FieldPathTest(Node):
    def __init__(self):
        super().__init__("field_path_test")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("visualization_frame", "world")
        self.declare_parameter("planned_path_topic", "/ur_mntfields_arm/test_planned_path")
        self.declare_parameter("trajectory_topic", "/ur_mntfields_arm/test_joint_trajectory")
        self.declare_parameter("goal_marker_topic", "/ur_mntfields_arm/test_goal_markers")
        self.declare_parameter("startup_positions", [0.12, -2.3, 1.9, -2.5, -1.57, 0.0])
        self.declare_parameter("camera_in_tool", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.10, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("voxel_size_m", 0.05)
        self.declare_parameter("raycast_stride_px", 8)
        self.declare_parameter("depth_min_m", 0.20)
        self.declare_parameter("depth_max_m", 2.00)
        self.declare_parameter("enable_robot_self_filter", True)
        self.declare_parameter("robot_self_filter_padding_m", 0.04)
        self.declare_parameter("robot_self_filter_tool_radius_m", 0.08)
        self.declare_parameter("robot_self_filter_mount_radius_m", 0.07)
        self.declare_parameter("clearance_backend", "original")
        self.declare_parameter("sdf_voxel_size_m", 0.04)
        self.declare_parameter("sdf_padding_m", 0.75)
        self.declare_parameter("sdf_max_cells", 4000000)
        self.declare_parameter("ur_type", "ur5")
        self.declare_parameter("checkpoint_path", "/tmp/ur_mntfields_arm/model/weights_final.pt")
        self.declare_parameter("model_dir", "/tmp/ur_mntfields_arm/model")
        self.declare_parameter("samples_dir", "/tmp/ur_mntfields_arm/samples")
        self.declare_parameter("step_size_q", 0.03)
        self.declare_parameter("rollout_max_steps", 120)
        self.declare_parameter("planner_mode", "bidirectional")
        self.declare_parameter("planner_direct_edge", True)
        self.declare_parameter("planner_shortcut", True)
        self.declare_parameter("collision_aware_field_rollout", True)
        self.declare_parameter("field_local_rollout_candidates", 32)
        self.declare_parameter("direct_joint_fallback_enabled", False)
        self.declare_parameter("publish_trajectory", True)
        self.declare_parameter("startup_pose_tolerance_rad", 0.05)
        self.declare_parameter("startup_pose_settle_s", 1.0)
        self.declare_parameter("min_goal_joint_delta_rad", 0.25)
        self.declare_parameter("trajectory_collision_margin_m", 0.01)
        self.declare_parameter("max_goal_candidates", 64)
        self.declare_parameter("goal_candidate_dedupe_rad", 0.035)
        self.declare_parameter("field_precheck_enabled", True)
        self.declare_parameter("field_precheck_min_speed", 1.0e-4)
        self.declare_parameter("field_precheck_neighborhood_samples", 16)
        self.declare_parameter("field_precheck_neighborhood_radius_norm", 0.015)
        self.declare_parameter("trajectory_max_joint_speed", 0.25)
        self.declare_parameter("trajectory_min_segment_dt", 0.35)
        self.declare_parameter("trajectory_waypoint_stride", 3)
        self.declare_parameter("trajectory_smoothing_window", 5)
        self.declare_parameter("interactive_goal_enabled", True)
        self.declare_parameter("interactive_goal_topic", "/clicked_point")
        self.declare_parameter("interactive_execute_on_click", True)
        self.declare_parameter("interactive_goal_mode", "tool_point")
        self.declare_parameter("fixed_goal_sequence_enabled", False)
        self.declare_parameter("fixed_goal_mode", "point")
        self.declare_parameter("fixed_goal_return_to_first", False)
        self.declare_parameter(
            "fixed_goal_joint_positions",
            [0.0] * 6,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY),
        )
        self.declare_parameter("fixed_goal_points_frame", "world")
        self.declare_parameter("fixed_goal_points_xyz", [0.60, 0.35, 0.68, 0.60, 0.35, 1.12, 0.78, 0.35, 0.68])
        self.declare_parameter("fixed_goal_reached_tolerance_rad", 0.05)
        self.declare_parameter("goal_view_standoff_m", [0.35, 0.45, 0.55])
        self.declare_parameter("goal_view_vertical_offsets_m", [0.0, 0.10, -0.10])
        self.declare_parameter("goal_view_lateral_offsets_m", [0.0, 0.10, -0.10])
        self.declare_parameter("goal_camera_alignment_min", 0.92)
        self.declare_parameter("goal_candidate_clearance_min_m", 0.01)
        self.declare_parameter("goal_tool_forward_alignment_min", 0.70)
        self.declare_parameter("path_shortcut_max_passes", 0)
        self.declare_parameter("path_shortcut_interp_step_rad", 0.04)
        self.declare_parameter("goal_tool_orientation_yaw_offsets_rad", [0.0, 1.5708, -1.5708, 3.1416])
        self.declare_parameter("goal_tool_orientation_pitch_offsets_rad", [0.0, 0.7854, -0.7854, 1.5708, -1.5708])
        self.declare_parameter("goal_tool_orientation_roll_offsets_rad", [0.0, 1.5708, -1.5708, 3.1416])
        self.declare_parameter("scene_boxes", [""])
        self.declare_parameter("scene_boxes_frame", "")

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.visualization_frame = str(self.get_parameter("visualization_frame").value)
        self.planned_path_topic = str(self.get_parameter("planned_path_topic").value)
        self.trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.startup_positions = np.asarray(self.get_parameter("startup_positions").value, dtype=np.float64).reshape(6)
        self.camera_in_tool = np.asarray(self.get_parameter("camera_in_tool").value, dtype=np.float64).reshape(4, 4)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
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
        self.sdf_voxel_size_m = float(max(1.0e-3, float(self.get_parameter("sdf_voxel_size_m").value)))
        self.sdf_padding_m = float(max(0.0, float(self.get_parameter("sdf_padding_m").value)))
        self.sdf_max_cells = max(10_000, int(self.get_parameter("sdf_max_cells").value))
        self.checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))
        self.model_dir = str(self.get_parameter("model_dir").value)
        self.samples_dir = Path(str(self.get_parameter("samples_dir").value))
        self.step_size_q = float(self.get_parameter("step_size_q").value)
        self.rollout_max_steps = int(self.get_parameter("rollout_max_steps").value)
        self.planner_mode = str(self.get_parameter("planner_mode").value).strip().lower()
        self.planner_direct_edge = bool(self.get_parameter("planner_direct_edge").value)
        self.planner_shortcut = bool(self.get_parameter("planner_shortcut").value)
        self.collision_aware_field_rollout = bool(self.get_parameter("collision_aware_field_rollout").value)
        self.field_local_rollout_candidates = max(4, int(self.get_parameter("field_local_rollout_candidates").value))
        self.direct_joint_fallback_enabled = bool(self.get_parameter("direct_joint_fallback_enabled").value)
        self.publish_trajectory = bool(self.get_parameter("publish_trajectory").value)
        self.startup_pose_tolerance_rad = float(self.get_parameter("startup_pose_tolerance_rad").value)
        self.startup_pose_settle_s = max(0.0, float(self.get_parameter("startup_pose_settle_s").value))
        self.min_goal_joint_delta_rad = float(self.get_parameter("min_goal_joint_delta_rad").value)
        self.trajectory_collision_margin_m = float(self.get_parameter("trajectory_collision_margin_m").value)
        self.max_goal_candidates = int(self.get_parameter("max_goal_candidates").value)
        self.goal_candidate_dedupe_rad = max(0.0, float(self.get_parameter("goal_candidate_dedupe_rad").value))
        self.field_precheck_enabled = bool(self.get_parameter("field_precheck_enabled").value)
        self.field_precheck_min_speed = max(0.0, float(self.get_parameter("field_precheck_min_speed").value))
        self.field_precheck_neighborhood_samples = max(0, int(self.get_parameter("field_precheck_neighborhood_samples").value))
        self.field_precheck_neighborhood_radius_norm = max(0.0, float(self.get_parameter("field_precheck_neighborhood_radius_norm").value))
        self.trajectory_max_joint_speed = float(self.get_parameter("trajectory_max_joint_speed").value)
        self.trajectory_min_segment_dt = float(self.get_parameter("trajectory_min_segment_dt").value)
        self.trajectory_waypoint_stride = int(self.get_parameter("trajectory_waypoint_stride").value)
        self.trajectory_smoothing_window = int(self.get_parameter("trajectory_smoothing_window").value)
        self.interactive_goal_enabled = bool(self.get_parameter("interactive_goal_enabled").value)
        self.interactive_goal_topic = str(self.get_parameter("interactive_goal_topic").value)
        self.interactive_execute_on_click = bool(self.get_parameter("interactive_execute_on_click").value)
        self.interactive_goal_mode = str(self.get_parameter("interactive_goal_mode").value).strip().lower()
        self.fixed_goal_sequence_enabled = bool(self.get_parameter("fixed_goal_sequence_enabled").value)
        self.fixed_goal_mode = str(self.get_parameter("fixed_goal_mode").value).strip().lower()
        self.fixed_goal_return_to_first = bool(self.get_parameter("fixed_goal_return_to_first").value)
        fixed_goal_joints = [float(v) for v in self.get_parameter("fixed_goal_joint_positions").value]
        if len(fixed_goal_joints) % 6 != 0:
            raise ValueError(
                f"fixed_goal_joint_positions must contain a multiple of 6 values, got {len(fixed_goal_joints)}"
            )
        self.fixed_goal_joint_positions = np.asarray(fixed_goal_joints, dtype=np.float64).reshape(-1, 6)
        self.fixed_goal_joint_sequence = self._goal_sequence_with_optional_return(self.fixed_goal_joint_positions)
        self.fixed_goal_points_frame = str(self.get_parameter("fixed_goal_points_frame").value)
        fixed_goal_xyz = [float(v) for v in self.get_parameter("fixed_goal_points_xyz").value]
        self.fixed_goal_points_xyz = np.asarray(fixed_goal_xyz, dtype=np.float64).reshape(-1, 3)
        self.fixed_goal_points_sequence = self._goal_sequence_with_optional_return(self.fixed_goal_points_xyz)
        self.fixed_goal_reached_tolerance_rad = float(self.get_parameter("fixed_goal_reached_tolerance_rad").value)
        self.goal_view_standoff_m = [float(v) for v in self.get_parameter("goal_view_standoff_m").value]
        self.goal_view_vertical_offsets_m = [float(v) for v in self.get_parameter("goal_view_vertical_offsets_m").value]
        self.goal_view_lateral_offsets_m = [float(v) for v in self.get_parameter("goal_view_lateral_offsets_m").value]
        self.goal_camera_alignment_min = float(self.get_parameter("goal_camera_alignment_min").value)
        self.goal_candidate_clearance_min_m = float(self.get_parameter("goal_candidate_clearance_min_m").value)
        self.goal_tool_forward_alignment_min = float(self.get_parameter("goal_tool_forward_alignment_min").value)
        self.path_shortcut_max_passes = int(self.get_parameter("path_shortcut_max_passes").value)
        self.path_shortcut_interp_step_rad = float(self.get_parameter("path_shortcut_interp_step_rad").value)
        self.goal_tool_orientation_yaw_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_yaw_offsets_rad").value
        ]
        self.goal_tool_orientation_pitch_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_pitch_offsets_rad").value
        ]
        self.goal_tool_orientation_roll_offsets_rad = [
            float(v) for v in self.get_parameter("goal_tool_orientation_roll_offsets_rad").value
        ]
        self.scene_boxes = self._parse_scene_boxes(self.get_parameter("scene_boxes").value)
        scene_boxes_frame = str(self.get_parameter("scene_boxes_frame").value)
        self.scene_boxes_frame = scene_boxes_frame if scene_boxes_frame else self.base_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.kinematics = UR5Kinematics(str(self.get_parameter("ur_type").value))
        self.field_model = ArmFieldModel(self.model_dir)
        self.field_model.load_checkpoint(self._resolve_checkpoint())
        self.planner = ArmFieldPlanner(self.field_model, self.kinematics)
        self.get_logger().info(
            "Field test planner config: "
            f"mode={self.planner_mode} clearance_backend={self.clearance_backend} "
            f"direct_edge={self.planner_direct_edge} shortcut={self.planner_shortcut} "
            f"step_size_q={self.step_size_q:.4f} rollout_max_steps={self.rollout_max_steps} "
            f"collision_margin_m={self.trajectory_collision_margin_m:.4f}"
        )
        self.robot_self_filter_checker = UR5PointCloudCollisionChecker(
            self.kinematics, np.zeros((0, 3), dtype=np.float32)
        )

        self.latest_info: CameraInfo | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_camera_pose: np.ndarray | None = None
        self.cached_collision_points: np.ndarray | None = None
        self.current_joints: np.ndarray | None = None
        self.startup_pose_reached = False
        self.startup_pose_reached_since: float | None = None
        self.last_startup_wait_log_time = 0.0
        self.last_camera_wait_log_time = 0.0
        self.last_goal_wait_log_time = 0.0
        self.last_fixed_wait_log_time = 0.0
        self.completed = False
        self.pending_clicked_goal_base: np.ndarray | None = None
        self.fixed_goal_index = 0
        self.active_goal_q: np.ndarray | None = None
        self.fixed_goal_markers: MarkerArray | None = None
        self.fixed_goal_markers_logged = False
        self.last_validation_min_idx = -1
        self.last_validation_min_q: np.ndarray | None = None
        self.active_goal_label = "interactive"
        self.rng = np.random.default_rng(23)

        self.path_pub = self.create_publisher(NavPath, self.planned_path_topic, 5)
        self.trajectory_pub = self.create_publisher(JointTrajectory, self.trajectory_topic, 5)
        marker_qos = QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_marker_pub = self.create_publisher(MarkerArray, self.goal_marker_topic, marker_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 20)
        if self.interactive_goal_enabled and not self.fixed_goal_sequence_enabled:
            self.create_subscription(PointStamped, self.interactive_goal_topic, self._clicked_goal_cb, 10)
        self.get_logger().info(f"Loaded field checkpoint: {self._resolve_checkpoint()}")
        self.get_logger().info(f"Field planner mode: {self.planner_mode}")
        if self.fixed_goal_sequence_enabled:
            if self.fixed_goal_mode == "joint":
                self.get_logger().info(
                    f"Fixed joint-space goal sequence enabled with {len(self.fixed_goal_joint_positions)} unique goals "
                    f"and {len(self.fixed_goal_joint_sequence)} trajectory legs: "
                    f"{np.round(self.fixed_goal_joint_positions, 3).tolist()}"
                )
            else:
                self.get_logger().info(
                    f"Fixed tool-point goal sequence enabled with {len(self.fixed_goal_points_xyz)} unique goals "
                    f"and {len(self.fixed_goal_points_sequence)} trajectory legs "
                    f"in frame={self.fixed_goal_points_frame}: {np.round(self.fixed_goal_points_xyz, 3).tolist()}"
                )
            self._publish_fixed_goal_point_markers()
            self.create_timer(1.0, self._republish_fixed_goal_point_markers)
            self.create_timer(0.5, self._fixed_goal_timer)
        elif self.interactive_goal_enabled:
            self.get_logger().info(
                f"Interactive 3D goal selection enabled on {self.interactive_goal_topic}. "
                f"Mode={self.interactive_goal_mode}. Use RViz 'Publish Point' to test the learned field "
                "against a chosen target."
            )

    def _resolve_checkpoint(self) -> Path:
        if self.checkpoint_path.exists():
            if self.checkpoint_path.name == "weights_final.pt":
                latest_epoch = self._latest_epoch_checkpoint(self.checkpoint_path.parent)
                if latest_epoch is not None:
                    final_epochs = self._checkpoint_epochs(self.checkpoint_path)
                    latest_epochs = self._checkpoint_epochs(latest_epoch)
                    if latest_epochs > final_epochs:
                        self.get_logger().warn(
                            f"Requested weights_final.pt has fewer trained epochs ({final_epochs}) than "
                            f"{latest_epoch.name} ({latest_epochs}); using {latest_epoch}."
                        )
                        return latest_epoch
            return self.checkpoint_path
        model_dir = Path(self.model_dir)
        latest_epoch = self._latest_epoch_checkpoint(model_dir)
        if latest_epoch is not None:
            return latest_epoch
        raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path} or in {model_dir}")

    def _latest_epoch_checkpoint(self, model_dir: Path) -> Path | None:
        candidates = sorted(model_dir.glob("weights_epoch_*.pt"))
        return candidates[-1] if candidates else None

    def _checkpoint_epochs(self, path: Path) -> int:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return -1
        try:
            return int(payload.get("total_epochs_trained", -1))
        except Exception:
            return -1

    def _camera_info_cb(self, msg: CameraInfo):
        self.latest_info = msg

    def _decode_depth_image(self, msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        if encoding in ("16uc1", "mono16"):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width).astype(np.float32) * 1e-3
            return depth
        if encoding in ("32fc1", "32sc1"):
            return np.frombuffer(msg.data, dtype=np.float32).reshape(height, width).astype(np.float32)
        # Fall back to inferring layout from row stride when encoding is blank or unexpected.
        step = int(msg.step)
        if step == width * 2:
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width).astype(np.float32) * 1e-3
        if step == width * 4:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(height, width).astype(np.float32)
        raise ValueError(
            f"Unsupported depth image encoding='{msg.encoding}' step={msg.step} "
            f"for shape=({height}, {width}) data_len={len(msg.data)}"
        )

    def _joint_state_cb(self, msg: JointState):
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        if not all(name in name_to_idx for name in JOINT_NAMES):
            return
        self.current_joints = np.array([msg.position[name_to_idx[name]] for name in JOINT_NAMES], dtype=np.float64)
        err = np.abs(self.current_joints - self.startup_positions)
        now = self.get_clock().now().nanoseconds * 1e-9
        within_tolerance = bool(np.all(err <= self.startup_pose_tolerance_rad))
        if within_tolerance:
            if self.startup_pose_reached_since is None:
                self.startup_pose_reached_since = now
            self.startup_pose_reached = bool((now - self.startup_pose_reached_since) >= self.startup_pose_settle_s)
        else:
            self.startup_pose_reached_since = None
            self.startup_pose_reached = False

    def _depth_cb(self, msg: Image):
        if self.completed and not self.interactive_goal_enabled:
            return
        if self.latest_info is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_camera_wait_log_time > 2.0:
                self.last_camera_wait_log_time = now
                self.get_logger().info("Waiting for camera_info before field path test.")
            return
        depth = self._decode_depth_image(msg)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed for test script: {exc}")
            return
        camera_pose = _transform_to_matrix(tf_msg)
        self.latest_depth = depth
        self.latest_camera_pose = camera_pose
        if self.interactive_goal_enabled and self.current_joints is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_startup_wait_log_time > 2.0:
                self.last_startup_wait_log_time = now
                self.get_logger().info("Waiting for joint_states before interactive field test.")
            return
        if self.fixed_goal_sequence_enabled and self.current_joints is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_startup_wait_log_time > 2.0:
                self.last_startup_wait_log_time = now
                self.get_logger().info("Waiting for joint_states before fixed goal sequence test.")
            return
        if ((not self.interactive_goal_enabled) or self.fixed_goal_sequence_enabled) and self.fixed_goal_index == 0 and (not self.startup_pose_reached):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_startup_wait_log_time > 2.0 and self.current_joints is not None:
                self.last_startup_wait_log_time = now
                err = np.abs(self.current_joints - self.startup_positions)
                max_idx = int(np.argmax(err))
                stable_s = 0.0 if self.startup_pose_reached_since is None else max(0.0, now - self.startup_pose_reached_since)
                self.get_logger().info(
                    f"Waiting for startup pose before field test: max_error_joint={JOINT_NAMES[max_idx]} "
                    f"error={float(err[max_idx]):.3f} target_q={np.round(self.startup_positions, 3).tolist()} "
                    f"current_q={np.round(self.current_joints, 3).tolist()} "
                    f"stable_s={stable_s:.2f}/{self.startup_pose_settle_s:.2f}"
                )
            return
        if self.fixed_goal_sequence_enabled:
            self._run_fixed_goal_sequence()
            return
        if self.interactive_goal_enabled:
            if self.pending_clicked_goal_base is not None:
                if self._run_to_target_point(self.pending_clicked_goal_base):
                    self.pending_clicked_goal_base = None
            else:
                now = self.get_clock().now().nanoseconds * 1e-9
                if now - self.last_goal_wait_log_time > 4.0:
                    self.last_goal_wait_log_time = now
                    self.get_logger().info(
                        "Waiting for a 3D goal point on /clicked_point. "
                        "In RViz, use the 'Publish Point' tool on the cabinet/interior target."
                    )
            return
        self._run_once(depth, self.latest_info, camera_pose)
        self.completed = True
        self.get_logger().info("Field path test complete.")
        self.create_timer(0.2, self._shutdown_once)

    def _shutdown_once(self):
        raise SystemExit(0)

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
            self.get_logger().warn(f"Goal point TF lookup failed ({source_frame} -> {target_frame}): {exc}")
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
            if len(values) == 6:
                boxes.append(np.asarray(values, dtype=np.float64))
        return boxes

    def _scene_boxes_in_base(self) -> np.ndarray:
        if not self.scene_boxes:
            return np.zeros((0, 6), dtype=np.float64)
        tf_m = np.eye(4, dtype=np.float64)
        if self.scene_boxes_frame != self.base_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, self.scene_boxes_frame, rclpy.time.Time())
                tf_m = _transform_to_matrix(tf)
            except TransformException as exc:
                self.get_logger().warn(
                    f"Scene box TF lookup failed ({self.scene_boxes_frame} -> {self.base_frame}): {exc}"
                )
                return np.zeros((0, 6), dtype=np.float64)
        out = []
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
            base_lo = np.min(corners_base, axis=0)
            base_hi = np.max(corners_base, axis=0)
            out.append(np.concatenate((0.5 * (base_lo + base_hi), np.maximum(base_hi - base_lo, 1e-6))))
        return np.asarray(out, dtype=np.float64)

    def _make_collision_checker(self, occupied_points: np.ndarray) -> UR5PointCloudCollisionChecker:
        return make_ur5_collision_checker(
            self.kinematics,
            occupied_points,
            box_obstacles=self._scene_boxes_in_base(),
            clearance_backend=self.clearance_backend,
            sdf_voxel_size_m=self.sdf_voxel_size_m,
            sdf_padding_m=self.sdf_padding_m,
            sdf_max_cells=self.sdf_max_cells,
        )

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

    def _filter_robot_self_points(self, points: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.enable_robot_self_filter:
            return points, 0
        return self.robot_self_filter_checker.filter_robot_self_points(
            points,
            np.asarray(q, dtype=np.float64),
            padding_m=self.robot_self_filter_padding_m,
            extra_spheres=self._tool_camera_self_filter_spheres(q),
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

    def _current_collision_points(self, q_start: np.ndarray, context: str) -> tuple[np.ndarray | None, int, int, bool]:
        if self.latest_depth is None or self.latest_info is None or self.latest_camera_pose is None:
            return None, 0, 0, False
        points_world = self._depth_to_world(self.latest_depth, self.latest_info, self.latest_camera_pose)
        raw_point_count = int(len(points_world))
        if raw_point_count > 0:
            points_world, self_removed = self._filter_robot_self_points(points_world, q_start)
            if len(points_world) > 0:
                self.cached_collision_points = np.asarray(points_world, dtype=np.float32).copy()
                return points_world, raw_point_count, self_removed, False
            self.get_logger().warn(
                f"No depth points available after robot self-filtering for {context}: "
                f"raw_points={raw_point_count} self_removed={self_removed}"
            )
        elif self.cached_collision_points is None:
            self.get_logger().warn(f"No depth points available for {context}.")
        if self.cached_collision_points is None or len(self.cached_collision_points) == 0:
            return None, raw_point_count, 0, False
        cached = np.asarray(self.cached_collision_points, dtype=np.float32)
        self.get_logger().warn(
            f"No current depth points available for {context}; reusing cached collision cloud "
            f"with {len(cached)} points."
        )
        return cached, raw_point_count, 0, True

    def _resolve_sample_file(self) -> Path:
        candidates = sorted(self.samples_dir.glob("step_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No training sample files found in {self.samples_dir}")
        preferred = [p for p in reversed(candidates) if "samples" in str(p.parent)]
        return preferred[0] if preferred else candidates[-1]

    def _load_goal_states(self) -> np.ndarray:
        path = self._resolve_sample_file()
        payload = np.load(path)
        if "frame_data" not in payload:
            raise KeyError(f"{path} does not contain frame_data")
        frame_data = np.asarray(payload["frame_data"], dtype=np.float32)
        if frame_data.ndim != 2 or len(frame_data) == 0:
            raise ValueError(f"{path} frame_data is empty")
        if frame_data.shape[1] == 13:
            qn = frame_data[:, :6]
        elif frame_data.shape[1] == 26:
            qn0 = frame_data[:, :6]
            qn1 = frame_data[:, 6:12]
            qn = np.vstack((qn0, qn1))
        else:
            raise ValueError(f"Unsupported frame_data width in {path}: {frame_data.shape}")
        goals = np.asarray([self.kinematics.denormalize(row) for row in qn], dtype=np.float32)
        self.get_logger().info(f"Loaded {len(goals)} goal states from {path}")
        return goals

    def _goal_sequence_with_optional_return(self, goals: np.ndarray) -> np.ndarray:
        arr = np.asarray(goals, dtype=np.float64)
        if not self.fixed_goal_return_to_first or len(arr) == 0:
            return arr.copy()
        return np.concatenate((arr, arr[:1].copy()), axis=0)

    def _clicked_goal_cb(self, msg: PointStamped):
        xyz = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        frame = msg.header.frame_id if msg.header.frame_id else self.visualization_frame
        xyz_base = self._transform_point_between_frames(xyz, frame, self.base_frame)
        if xyz_base is None:
            return
        self.pending_clicked_goal_base = xyz_base
        self.active_goal_label = "interactive"
        self.get_logger().info(
            f"Received interactive 3D goal point: frame={frame} base_xyz={np.round(xyz_base, 3).tolist()}"
        )

    def _publish_fixed_goal_point_markers(self):
        if not self.fixed_goal_sequence_enabled:
            return
        arr = MarkerArray()
        frame_id = self.base_frame
        palette = (
            (1.0, 0.55, 0.05),
            (1.0, 0.10, 0.10),
            (0.95, 0.75, 0.10),
            (0.85, 0.20, 0.70),
        )
        if self.fixed_goal_mode == "joint":
            goal_points_base = [
                np.asarray(self.kinematics.fk(q)[:3, 3], dtype=np.float64)
                for q in self.fixed_goal_joint_positions
            ]
        else:
            goal_points_base = []
            for xyz_src in self.fixed_goal_points_xyz:
                xyz_base = self._transform_point_between_frames(xyz_src, self.fixed_goal_points_frame, self.base_frame)
                if xyz_base is not None:
                    goal_points_base.append(xyz_base)
        for idx, xyz_base in enumerate(goal_points_base):
            marker = Marker()
            marker.header = Header(frame_id=frame_id)
            marker.ns = "field_test_fixed_goal_points"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.07
            marker.scale.y = 0.07
            marker.scale.z = 0.07
            color = palette[idx % len(palette)]
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = 0.95
            marker.pose.position.x = float(xyz_base[0])
            marker.pose.position.y = float(xyz_base[1])
            marker.pose.position.z = float(xyz_base[2])
            marker.pose.orientation.w = 1.0
            arr.markers.append(marker)
        if arr.markers:
            self.fixed_goal_markers = arr
            self.goal_marker_pub.publish(arr)
            if not self.fixed_goal_markers_logged:
                self.fixed_goal_markers_logged = True
                self.get_logger().info(
                    f"Published {len(arr.markers)} fixed goal markers on {self.goal_marker_topic} "
                    f"(ns=field_test_fixed_goal_points)."
                )

    def _republish_fixed_goal_point_markers(self):
        if self.fixed_goal_markers is None:
            self._publish_fixed_goal_point_markers()
        if self.fixed_goal_markers is not None:
            self.goal_marker_pub.publish(self.fixed_goal_markers)

    def _fixed_goal_timer(self):
        if not self.fixed_goal_sequence_enabled or self.completed:
            return
        if self.latest_info is None or self.latest_depth is None or self.latest_camera_pose is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_fixed_wait_log_time > 2.0:
                self.last_fixed_wait_log_time = now
                self.get_logger().info(
                    "Waiting for camera_info/depth/TF before fixed goal sequence: "
                    f"have_camera_info={self.latest_info is not None} "
                    f"have_depth={self.latest_depth is not None} "
                    f"have_camera_pose={self.latest_camera_pose is not None}"
                )
            return
        if self.current_joints is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_fixed_wait_log_time > 2.0:
                self.last_fixed_wait_log_time = now
                self.get_logger().info("Waiting for joint_states before fixed goal sequence.")
            return
        if self.fixed_goal_index == 0 and not self.startup_pose_reached:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self.last_fixed_wait_log_time > 2.0:
                self.last_fixed_wait_log_time = now
                err = np.abs(np.asarray(self.current_joints, dtype=np.float64) - self.startup_positions)
                max_idx = int(np.argmax(err))
                stable_s = 0.0 if self.startup_pose_reached_since is None else max(0.0, now - self.startup_pose_reached_since)
                self.get_logger().info(
                    f"Waiting for startup pose before fixed goal sequence: max_error_joint={JOINT_NAMES[max_idx]} "
                    f"error={float(err[max_idx]):.3f} target_q={np.round(self.startup_positions, 3).tolist()} "
                    f"current_q={np.round(self.current_joints, 3).tolist()} "
                    f"stable_s={stable_s:.2f}/{self.startup_pose_settle_s:.2f}"
                )
            return
        self._run_fixed_goal_sequence()

    def _run_fixed_goal_sequence(self):
        if self.current_joints is None:
            return
        goals = self.fixed_goal_joint_sequence if self.fixed_goal_mode == "joint" else self.fixed_goal_points_sequence
        if self.active_goal_q is not None:
            err = np.max(np.abs(np.asarray(self.current_joints, dtype=np.float64) - self.active_goal_q))
            if err <= self.fixed_goal_reached_tolerance_rad:
                self.get_logger().info(
                    f"Reached fixed goal {self.fixed_goal_index}/{len(goals)} within "
                    f"{self.fixed_goal_reached_tolerance_rad:.3f} rad."
                )
                self.active_goal_q = None
            else:
                return
        if self.fixed_goal_index >= len(goals):
            if not self.completed:
                self.completed = True
                self.get_logger().info("Fixed field-goal sequence complete.")
            return
        if self.fixed_goal_mode == "joint":
            target_q = np.asarray(self.fixed_goal_joint_sequence[self.fixed_goal_index], dtype=np.float64)
            self.get_logger().info(
                f"Planning fixed joint goal {self.fixed_goal_index + 1}/{len(goals)} "
                f"goal_q={np.round(target_q, 3).tolist()}"
            )
            self.active_goal_label = f"fixed joint goal {self.fixed_goal_index + 1}/{len(goals)}"
            goal_q = self._run_to_joint_goal(target_q)
            if goal_q is None:
                self.get_logger().warn(f"Failed to plan fixed joint goal {self.fixed_goal_index + 1}/{len(goals)}.")
                self.completed = True
                return
            self.active_goal_q = goal_q
            self.fixed_goal_index += 1
            return
        target_src = self.fixed_goal_points_sequence[self.fixed_goal_index]
        target_base = self._transform_point_between_frames(target_src, self.fixed_goal_points_frame, self.base_frame)
        if target_base is None:
            return
        self.get_logger().info(
            f"Planning fixed goal {self.fixed_goal_index + 1}/{len(goals)} "
            f"src_frame={self.fixed_goal_points_frame} src_xyz={np.round(target_src, 3).tolist()} "
            f"base_xyz={np.round(target_base, 3).tolist()}"
        )
        self.active_goal_label = f"fixed goal {self.fixed_goal_index + 1}/{len(goals)}"
        goal_q = self._run_to_target_point(target_base)
        if goal_q is None:
            self.get_logger().warn(
                f"Failed to plan fixed goal {self.fixed_goal_index + 1}/{len(goals)}."
            )
            self.completed = True
            return
        self.active_goal_q = goal_q
        self.fixed_goal_index += 1

    def _segment_collision_free(
        self, checker: UR5PointCloudCollisionChecker, qa: np.ndarray, qb: np.ndarray, margin_m: float | None = None
    ) -> tuple[bool, float]:
        qa = np.asarray(qa, dtype=np.float64)
        qb = np.asarray(qb, dtype=np.float64)
        margin = self.trajectory_collision_margin_m if margin_m is None else float(margin_m)
        max_step = max(0.01, self.path_shortcut_interp_step_rad)
        max_delta = float(np.max(np.abs(qb - qa)))
        nseg = max(1, int(np.ceil(max_delta / max_step)))
        pts = np.asarray([qa + alpha * (qb - qa) for alpha in np.linspace(0.0, 1.0, nseg + 1)], dtype=np.float32)
        clearances = checker.clearance_batch(pts)
        if clearances.size == 0:
            return False, -1.0
        min_clearance = float(np.min(clearances))
        return bool(min_clearance >= margin), min_clearance

    def _shortcut_plan(self, checker: UR5PointCloudCollisionChecker, plan: np.ndarray) -> np.ndarray:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) <= 2:
            return pts.astype(np.float32)
        if self.path_shortcut_max_passes <= 0:
            return pts.astype(np.float32)
        cur = pts.copy()
        for _ in range(max(1, self.path_shortcut_max_passes)):
            shortcut = [cur[0].copy()]
            idx = 0
            changed = False
            while idx < len(cur) - 1:
                next_idx = idx + 1
                for cand in range(len(cur) - 1, idx, -1):
                    ok, _ = self._segment_collision_free(checker, cur[idx], cur[cand])
                    if ok:
                        next_idx = cand
                        break
                if next_idx > idx + 1:
                    changed = True
                shortcut.append(cur[next_idx].copy())
                idx = next_idx
            cur = np.asarray(shortcut, dtype=np.float64)
            if not changed:
                break
        return cur.astype(np.float32)

    def _plan_length(self, plan: np.ndarray) -> float:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _direct_joint_plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)
        max_delta = float(np.max(np.abs(q_goal - q_start)))
        step = max(0.01, 0.5 * self.step_size_q)
        nseg = max(1, int(np.ceil(max_delta / step)))
        return np.asarray(
            [q_start + alpha * (q_goal - q_start) for alpha in np.linspace(0.0, 1.0, nseg + 1)],
            dtype=np.float32,
        )

    def _candidate_camera_goal_states(
        self, checker: UR5PointCloudCollisionChecker, q_start: np.ndarray, target_base: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        target = np.asarray(target_base, dtype=np.float64).reshape(3)
        current_tool = self.kinematics.fk(q_start)
        current_cam = self.kinematics.tool_to_camera_pose(current_tool, self.camera_in_tool)
        current_cam_xyz = current_cam[:3, 3]
        nominal = current_cam_xyz - target
        if np.linalg.norm(nominal) < 1e-6:
            nominal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        nominal /= np.linalg.norm(nominal)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = np.cross(world_up, nominal)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(nominal, lateral)
        vertical /= np.linalg.norm(vertical)

        candidates: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for standoff in self.goal_view_standoff_m:
            for dz in self.goal_view_vertical_offsets_m:
                for dy in self.goal_view_lateral_offsets_m:
                    cam_xyz = target + nominal * float(standoff) + vertical * float(dz) + lateral * float(dy)
                    cam_pose = np.eye(4, dtype=np.float64)
                    cam_pose[:3, :3] = look_at_rotation(cam_xyz, target, world_up)
                    cam_pose[:3, 3] = cam_xyz
                    desired_tool = self.kinematics.camera_to_tool_pose(cam_pose, self.camera_in_tool)
                    q_goal = self.kinematics.solve_ik_full(desired_tool, q_start)
                    if q_goal is None:
                        continue
                    clearance = float(checker.clearance_batch(np.asarray([q_goal], dtype=np.float32))[0])
                    if not np.isfinite(clearance) or clearance < self.goal_candidate_clearance_min_m:
                        continue
                    actual_tool = self.kinematics.fk(q_goal)
                    actual_cam = self.kinematics.tool_to_camera_pose(actual_tool, self.camera_in_tool)
                    optical = actual_cam[:3, 2]
                    to_target = target - actual_cam[:3, 3]
                    n = np.linalg.norm(to_target)
                    if n < 1e-6:
                        continue
                    alignment = float(np.dot(optical, to_target / n))
                    if alignment < self.goal_camera_alignment_min:
                        continue
                    move_cost = float(np.linalg.norm(q_goal - q_start))
                    score = 2.0 * alignment + 0.5 * clearance - 0.35 * move_cost
                    candidates.append((q_goal.astype(np.float64), actual_cam, score, clearance))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def _candidate_tool_point_goal_states(
        self, checker: UR5PointCloudCollisionChecker, q_start: np.ndarray, target_base: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        target = np.asarray(target_base, dtype=np.float64).reshape(3)
        current_tool = self.kinematics.fk(q_start)
        base_rot = current_tool[:3, :3].copy()
        target_from_base = target / max(np.linalg.norm(target), 1e-8)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        approach_x = look_at_rotation(target - 0.20 * target_from_base, target, world_up)
        approach_neg_x = look_at_rotation(target + 0.20 * target_from_base, target, world_up)
        orientation_bases = [
            ("current", base_rot),
            ("look_at_from_base", approach_x @ _rot_y(-np.pi / 2.0)),
            ("look_at_from_base_flip", approach_x @ _rot_y(np.pi / 2.0)),
            ("look_at_outward", approach_neg_x @ _rot_y(-np.pi / 2.0)),
            ("look_at_outward_flip", approach_neg_x @ _rot_y(np.pi / 2.0)),
        ]
        candidates: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        stats = {"attempted": 0, "ik_fail": 0, "clearance_fail": 0, "alignment_fail": 0, "accepted": 0}
        seen = set()
        for _label, orient_base in orientation_bases:
            for yaw in self.goal_tool_orientation_yaw_offsets_rad:
                for pitch in self.goal_tool_orientation_pitch_offsets_rad:
                    for roll in self.goal_tool_orientation_roll_offsets_rad:
                        stats["attempted"] += 1
                        desired_tool = np.eye(4, dtype=np.float64)
                        desired_tool[:3, :3] = orient_base @ _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
                        desired_tool[:3, 3] = target
                        q_goal = self.kinematics.solve_ik_full(desired_tool, q_start)
                        if q_goal is None:
                            stats["ik_fail"] += 1
                            continue
                        q_key = tuple(np.round(q_goal, 4).tolist())
                        if q_key in seen:
                            continue
                        seen.add(q_key)
                        clearance = float(checker.clearance_batch(np.asarray([q_goal], dtype=np.float32))[0])
                        if not np.isfinite(clearance) or clearance < self.goal_candidate_clearance_min_m:
                            stats["clearance_fail"] += 1
                            continue
                        actual_tool = self.kinematics.fk(q_goal)
                        pos_err = float(np.linalg.norm(actual_tool[:3, 3] - target))
                        tool_forward = actual_tool[:3, 0]
                        alignment = float(np.dot(tool_forward, target_from_base))
                        if alignment < self.goal_tool_forward_alignment_min:
                            stats["alignment_fail"] += 1
                            continue
                        move_cost = float(np.linalg.norm(q_goal - q_start))
                        score = 2.0 * alignment + 8.0 * clearance - 0.25 * move_cost - 4.0 * pos_err
                        candidates.append((q_goal.astype(np.float64), actual_tool, score, clearance))
                        stats["accepted"] += 1
        candidates.sort(key=lambda item: item[2], reverse=True)
        self.get_logger().info(
            "Tool-point goal search: "
            f"attempted={stats['attempted']} ik_fail={stats['ik_fail']} "
            f"clearance_fail={stats['clearance_fail']} alignment_fail={stats['alignment_fail']} "
            f"accepted={stats['accepted']}"
        )
        return candidates

    def _dedupe_goal_candidates(self, candidates: list[tuple[np.ndarray, np.ndarray, float, float]]) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        if self.goal_candidate_dedupe_rad <= 0.0 or len(candidates) <= 1:
            return candidates
        out: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for cand in candidates:
            q = np.asarray(cand[0], dtype=np.float64)
            if any(float(np.max(np.abs(q - np.asarray(prev[0], dtype=np.float64)))) <= self.goal_candidate_dedupe_rad for prev in out):
                continue
            out.append(cand)
        if len(out) < len(candidates):
            self.get_logger().info(
                f"Deduped goal candidates: kept={len(out)}/{len(candidates)} "
                f"threshold_rad={self.goal_candidate_dedupe_rad:.3f}"
            )
        return out

    def _field_candidate_precheck(self, q_start: np.ndarray, q_goal: np.ndarray) -> dict[str, float | bool]:
        if not self.field_precheck_enabled:
            return {"ok": True, "field_speed_start": -1.0, "field_speed_goal": -1.0, "field_goal_nbhd_mean": -1.0, "field_goal_nbhd_min": -1.0}
        q0n = self.kinematics.normalize(np.asarray(q_start, dtype=np.float64)).astype(np.float32)
        q1n = self.kinematics.normalize(np.asarray(q_goal, dtype=np.float64)).astype(np.float32)
        pred0, pred1 = self.field_model.predict_normalized_pair_speeds(q0n, q1n)
        speed0 = float(pred0[0]) if len(pred0) else float("nan")
        speed1 = float(pred1[0]) if len(pred1) else float("nan")
        nbhd_mean = speed1
        nbhd_min = speed1
        if self.field_precheck_neighborhood_samples > 0 and self.field_precheck_neighborhood_radius_norm > 0.0:
            samples = [q1n]
            for _ in range(self.field_precheck_neighborhood_samples):
                direction = self.rng.normal(size=6)
                norm = float(np.linalg.norm(direction))
                if norm > 1.0e-8:
                    direction = direction / norm
                radius = float(self.rng.uniform(0.0, self.field_precheck_neighborhood_radius_norm))
                samples.append(np.clip(q1n + direction.astype(np.float32) * radius, -0.5, 0.5))
            q1_samples = np.asarray(samples, dtype=np.float32)
            q0_samples = np.repeat(q0n[None, :], len(q1_samples), axis=0)
            _p0, p1 = self.field_model.predict_normalized_pair_speeds(q0_samples, q1_samples)
            finite = p1[np.isfinite(p1)]
            if len(finite):
                nbhd_mean = float(np.mean(finite))
                nbhd_min = float(np.min(finite))
        ok = (
            np.isfinite(speed0)
            and np.isfinite(speed1)
            and np.isfinite(nbhd_mean)
            and np.isfinite(nbhd_min)
            and max(speed0, speed1, nbhd_mean) >= self.field_precheck_min_speed
        )
        return {
            "ok": bool(ok),
            "field_speed_start": speed0,
            "field_speed_goal": speed1,
            "field_goal_nbhd_mean": nbhd_mean,
            "field_goal_nbhd_min": nbhd_min,
        }

    def _field_speed_at_waypoint(self, q_waypoint: np.ndarray | None, q_goal: np.ndarray) -> float:
        if q_waypoint is None:
            return -1.0
        q0n = self.kinematics.normalize(np.asarray(q_waypoint, dtype=np.float64)).astype(np.float32)
        q1n = self.kinematics.normalize(np.asarray(q_goal, dtype=np.float64)).astype(np.float32)
        pred0, _pred1 = self.field_model.predict_normalized_pair_speeds(q0n, q1n)
        return float(pred0[0]) if len(pred0) and np.isfinite(pred0[0]) else -1.0

    def _run_to_target_point(self, target: np.ndarray) -> np.ndarray | None:
        if self.latest_depth is None or self.latest_info is None:
            return None
        if self.current_joints is None:
            self.get_logger().warn("Cannot plan to clicked goal because no joint state has been received yet.")
            return None
        if self.latest_camera_pose is None:
            self.get_logger().warn("Cannot plan to clicked goal because no camera pose is available yet.")
            return None
        target = np.asarray(target, dtype=np.float64).copy()
        points_world = self._depth_to_world(self.latest_depth, self.latest_info, self.latest_camera_pose)
        if len(points_world) == 0:
            self.get_logger().warn("No depth points available for interactive 3D goal planning.")
            return None
        q_start = np.asarray(self.current_joints, dtype=np.float64).copy()
        raw_point_count = int(len(points_world))
        points_world, self_removed = self._filter_robot_self_points(points_world, q_start)
        if len(points_world) == 0:
            self.get_logger().warn(
                f"No depth points available after robot self-filtering: raw_points={raw_point_count} "
                f"self_removed={self_removed}"
            )
            return None
        self.cached_collision_points = np.asarray(points_world, dtype=np.float32).copy()
        checker = self._make_collision_checker(points_world)
        candidate_t0 = time.perf_counter()
        if self.interactive_goal_mode == "tool_point":
            candidates = self._candidate_tool_point_goal_states(checker, q_start, target)
        else:
            candidates = self._candidate_camera_goal_states(checker, q_start, target)
        candidates = self._dedupe_goal_candidates(candidates)
        candidate_ms = (time.perf_counter() - candidate_t0) * 1e3
        if not candidates:
            self.get_logger().warn(
                f"No feasible {self.interactive_goal_mode} goals found for {self.active_goal_label} "
                f"base_xyz={np.round(target, 3).tolist()}"
            )
            return None
        self.get_logger().info(
            f"{self.active_goal_label} candidates: mode={self.interactive_goal_mode} "
            f"target={np.round(target, 3).tolist()} feasible={len(candidates)} candidate_ms={candidate_ms:.1f}"
        )
        best_plan = None
        best_goal = None
        best_clearance = -1.0
        best_goal_pose = None
        found_valid_path = False
        for idx, (q_goal, actual_pose, _, goal_clearance) in enumerate(candidates[: self.max_goal_candidates], start=1):
            field_diag = self._field_candidate_precheck(q_start, q_goal)
            if not bool(field_diag.get("ok", True)):
                self.get_logger().info(
                    f"{self.active_goal_label} candidate={idx}/{min(len(candidates), self.max_goal_candidates)} "
                    f"rejected by field precheck: goal_q={np.round(q_goal, 3).tolist()} "
                    f"field_speed_start={float(field_diag.get('field_speed_start', -1.0)):.4f} "
                    f"field_speed_goal={float(field_diag.get('field_speed_goal', -1.0)):.4f} "
                    f"field_goal_nbhd_mean={float(field_diag.get('field_goal_nbhd_mean', -1.0)):.4f} "
                    f"field_goal_nbhd_min={float(field_diag.get('field_goal_nbhd_min', -1.0)):.4f}"
                )
                continue
            plan_t0 = time.perf_counter()
            if self.collision_aware_field_rollout:
                raw_plan = self.planner.plan_collision_aware(
                    checker,
                    q_start,
                    q_goal,
                    self.step_size_q,
                    self.rollout_max_steps,
                    clearance_margin_m=self.trajectory_collision_margin_m,
                    max_local_candidates=self.field_local_rollout_candidates,
                    allow_direct_edge=self.planner_direct_edge,
                    shortcut_path=self.planner_shortcut,
                )
            else:
                raw_plan = self.planner.plan(
                    q_start, q_goal, self.step_size_q, self.rollout_max_steps, mode=self.planner_mode
                )
            plan_ms = (time.perf_counter() - plan_t0) * 1e3
            validate_t0 = time.perf_counter()
            shortcut_plan = self._shortcut_plan(checker, raw_plan)
            path_ok, min_clearance = self._validate_plan_collision(checker, shortcut_plan)
            min_q_field_speed = self._field_speed_at_waypoint(self.last_validation_min_q, q_goal)
            validate_ms = (time.perf_counter() - validate_t0) * 1e3
            raw_len = self._plan_length(raw_plan)
            short_len = self._plan_length(shortcut_plan)
            self.get_logger().info(
                f"{self.active_goal_label} candidate={idx}/{min(len(candidates), self.max_goal_candidates)} "
                f"goal_q={np.round(q_goal, 3).tolist()} raw_waypoints={len(raw_plan)} short_waypoints={len(shortcut_plan)} "
                f"raw_len={raw_len:.3f} short_len={short_len:.3f} goal_clearance_m={goal_clearance:.4f} "
                f"field_speed_start={float(field_diag.get('field_speed_start', -1.0)):.4f} "
                f"field_speed_goal={float(field_diag.get('field_speed_goal', -1.0)):.4f} "
                f"field_goal_nbhd_mean={float(field_diag.get('field_goal_nbhd_mean', -1.0)):.4f} "
                f"path_clearance_m={min_clearance:.4f} path_ok={path_ok} "
                f"min_q_field_speed={min_q_field_speed:.4f} "
                f"min_clearance_idx={self.last_validation_min_idx} "
                f"min_clearance_q={np.round(self.last_validation_min_q, 3).tolist() if self.last_validation_min_q is not None else None} "
                f"plan_ms={plan_ms:.1f} validate_ms={validate_ms:.1f}"
            )
            if path_ok:
                best_plan = shortcut_plan
                best_goal = q_goal
                best_clearance = min_clearance
                best_goal_pose = actual_pose
                found_valid_path = True
                break
            if min_clearance > best_clearance:
                best_plan = shortcut_plan
                best_goal = q_goal
                best_clearance = min_clearance
                best_goal_pose = actual_pose
        if best_plan is None or best_goal is None or best_goal_pose is None:
            self.get_logger().warn("Interactive 3D goal planning could not produce any plan.")
            return None
        if not found_valid_path:
            self.get_logger().warn(
                f"All candidate field rollouts failed collision validation for {self.active_goal_label} "
                f"target={np.round(target, 3).tolist()}. "
                f"Best_min_clearance_m={best_clearance:.4f}. Refusing to execute invalid trajectory."
            )
            self._publish_goal_markers(q_start, best_goal, target)
            self._publish_planned_path(best_plan)
            return None
        reached_pose = (
            self.kinematics.tool_to_camera_pose(best_goal_pose, self.camera_in_tool)
            if self.interactive_goal_mode != "tool_point"
            else best_goal_pose
        )
        self.get_logger().info(
            f"Selected {self.active_goal_label}: target={np.round(target, 3).tolist()} "
            f"mode={self.interactive_goal_mode} goal_q={np.round(best_goal, 3).tolist()} "
            f"reached_xyz={np.round(reached_pose[:3, 3], 3).tolist()} plan_waypoints={len(best_plan)} "
            f"min_clearance_m={best_clearance:.4f}"
        )
        self._publish_goal_markers(q_start, best_goal, target)
        self._publish_planned_path(best_plan)
        if self.publish_trajectory and self.interactive_execute_on_click:
            self._publish_joint_trajectory(best_plan)
        return np.asarray(best_goal, dtype=np.float64)

    def _run_to_joint_goal(self, q_goal: np.ndarray) -> np.ndarray | None:
        if self.latest_depth is None or self.latest_info is None:
            return None
        if self.current_joints is None:
            self.get_logger().warn("Cannot plan to fixed joint goal because no joint state has been received yet.")
            return None
        if self.latest_camera_pose is None:
            self.get_logger().warn("Cannot plan to fixed joint goal because no camera pose is available yet.")
            return None
        q_start = np.asarray(self.current_joints, dtype=np.float64).copy()
        q_goal = self.kinematics.clamp(np.asarray(q_goal, dtype=np.float64).reshape(6))
        points_world, raw_point_count, self_removed, used_cached_cloud = self._current_collision_points(
            q_start, "fixed joint goal planning"
        )
        if points_world is None:
            return None
        checker = self._make_collision_checker(points_world)
        goal_clearance = float(checker.clearance(q_goal))
        field_diag = self._field_candidate_precheck(q_start, q_goal)
        if not bool(field_diag.get("ok", True)):
            self.get_logger().warn(
                f"{self.active_goal_label} rejected by field precheck: goal_q={np.round(q_goal, 3).tolist()} "
                f"field_speed_start={float(field_diag.get('field_speed_start', -1.0)):.4f} "
                f"field_speed_goal={float(field_diag.get('field_speed_goal', -1.0)):.4f} "
                f"field_goal_nbhd_mean={float(field_diag.get('field_goal_nbhd_mean', -1.0)):.4f} "
                f"field_goal_nbhd_min={float(field_diag.get('field_goal_nbhd_min', -1.0)):.4f}"
            )
            return None
        plan_t0 = time.perf_counter()
        if self.collision_aware_field_rollout:
            raw_plan = self.planner.plan_collision_aware(
                checker,
                q_start,
                q_goal,
                self.step_size_q,
                self.rollout_max_steps,
                clearance_margin_m=self.trajectory_collision_margin_m,
                max_local_candidates=self.field_local_rollout_candidates,
                allow_direct_edge=self.planner_direct_edge,
                shortcut_path=self.planner_shortcut,
            )
        else:
            raw_plan = self.planner.plan(q_start, q_goal, self.step_size_q, self.rollout_max_steps, mode=self.planner_mode)
        plan_ms = (time.perf_counter() - plan_t0) * 1e3
        planner_debug = dict(getattr(self.planner, "last_debug", {}))
        used_direct_fallback = False
        if np.asarray(raw_plan).ndim != 2 or len(raw_plan) == 0:
            if not self.direct_joint_fallback_enabled:
                self.get_logger().warn(
                    f"{self.active_goal_label}: field planner returned empty path. "
                    f"Refusing direct joint fallback because direct_joint_fallback_enabled=false. "
                    f"planner_debug={planner_debug}"
                )
                return None
            used_direct_fallback = True
            self.get_logger().warn(
                f"{self.active_goal_label}: field planner returned empty path; "
                f"trying direct joint-space interpolation fallback. planner_debug={planner_debug}"
            )
            raw_plan = self._direct_joint_plan(q_start, q_goal)
        validate_t0 = time.perf_counter()
        plan = self._shortcut_plan(checker, raw_plan)
        path_ok, min_clearance = self._validate_plan_collision(checker, plan)
        min_q_field_speed = self._field_speed_at_waypoint(self.last_validation_min_q, q_goal)
        validate_ms = (time.perf_counter() - validate_t0) * 1e3
        self.get_logger().info(
            f"{self.active_goal_label}: start_q={np.round(q_start, 3).tolist()} "
            f"goal_q={np.round(q_goal, 3).tolist()} raw_waypoints={len(raw_plan)} plan_waypoints={len(plan)} "
            f"raw_len={self._plan_length(raw_plan):.3f} plan_len={self._plan_length(plan):.3f} "
            f"goal_clearance_m={goal_clearance:.4f} path_clearance_m={min_clearance:.4f} path_ok={path_ok} "
            f"field_speed_start={float(field_diag.get('field_speed_start', -1.0)):.4f} "
            f"field_speed_goal={float(field_diag.get('field_speed_goal', -1.0)):.4f} "
            f"field_goal_nbhd_mean={float(field_diag.get('field_goal_nbhd_mean', -1.0)):.4f} "
            f"min_q_field_speed={min_q_field_speed:.4f} "
            f"min_clearance_idx={self.last_validation_min_idx} "
            f"min_clearance_q={np.round(self.last_validation_min_q, 3).tolist() if self.last_validation_min_q is not None else None} "
            f"raw_points={raw_point_count} self_removed={self_removed} "
            f"used_cached_cloud={used_cached_cloud} collision_points={len(points_world)} "
            f"used_direct_fallback={used_direct_fallback} planner_debug={planner_debug} "
            f"plan_ms={plan_ms:.1f} validate_ms={validate_ms:.1f}"
        )
        self._publish_goal_markers(q_start, q_goal, self.kinematics.fk(q_goal)[:3, 3])
        self._publish_planned_path(plan)
        if not path_ok:
            self.get_logger().warn(
                f"Collision validation failed for {self.active_goal_label}. Refusing to execute invalid trajectory."
            )
            return None
        if self.publish_trajectory and self.interactive_execute_on_click:
            self._publish_joint_trajectory(plan)
        return q_goal

    def _validate_plan_collision(self, checker: UR5PointCloudCollisionChecker, plan: np.ndarray) -> tuple[bool, float]:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            self.last_validation_min_idx = -1
            self.last_validation_min_q = None
            return False, -1.0
        dense = [pts[0].copy()]
        max_delta_step = max(0.02, 0.5 * self.step_size_q)
        for q in pts[1:]:
            prev = dense[-1]
            max_delta = float(np.max(np.abs(q - prev)))
            nseg = max(1, int(np.ceil(max_delta / max_delta_step)))
            for alpha in np.linspace(1.0 / nseg, 1.0, nseg):
                dense.append(prev + alpha * (q - prev))
        clearances = checker.clearance_batch(np.asarray(dense, dtype=np.float32))
        if clearances.size == 0:
            self.last_validation_min_idx = -1
            self.last_validation_min_q = None
            return False, -1.0
        min_idx = int(np.argmin(clearances))
        min_clearance = float(clearances[min_idx])
        self.last_validation_min_idx = min_idx
        self.last_validation_min_q = np.asarray(dense[min_idx], dtype=np.float64).copy()
        return bool(min_clearance >= self.trajectory_collision_margin_m), min_clearance

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

    def _run_once(self, depth: np.ndarray, info: CameraInfo, camera_pose: np.ndarray):
        points_world = self._depth_to_world(depth, info, camera_pose)
        if len(points_world) == 0:
            self.get_logger().warn("No depth points available for collision validation in field test.")
            return
        q_start = self.current_joints.copy() if self.current_joints is not None else self.startup_positions.copy()
        raw_point_count = int(len(points_world))
        points_world, self_removed = self._filter_robot_self_points(points_world, q_start)
        if len(points_world) == 0:
            self.get_logger().warn(
                f"No depth points available after robot self-filtering in field test: "
                f"raw_points={raw_point_count} self_removed={self_removed}"
            )
            return
        self.cached_collision_points = np.asarray(points_world, dtype=np.float32).copy()
        occupied_points = points_world
        checker = self._make_collision_checker(occupied_points)
        goals = self._load_goal_states()
        deltas = np.max(np.abs(goals - q_start[None, :]), axis=1)
        clearances = checker.clearance_batch(goals)
        valid = (deltas >= self.min_goal_joint_delta_rad) & np.isfinite(clearances) & (clearances > 0.0)
        valid_goals = goals[valid]
        if len(valid_goals) == 0:
            self.get_logger().warn(
                f"No valid goal states found in saved samples for current_q={np.round(q_start, 3).tolist()}"
            )
            return
        order = np.argsort(-deltas[valid])
        valid_goals = valid_goals[order][: self.max_goal_candidates]
        self.get_logger().info(
            f"Field path test candidate goals: loaded={len(goals)} valid={len(valid_goals)} "
            f"collision_points={len(occupied_points)} raw_points={raw_point_count} self_removed={self_removed}"
        )
        best_plan = None
        best_goal = None
        best_clearance = -1.0
        for idx, q_goal in enumerate(valid_goals, start=1):
            if self.collision_aware_field_rollout:
                raw_plan = self.planner.plan_collision_aware(
                    checker,
                    q_start,
                    q_goal,
                    self.step_size_q,
                    self.rollout_max_steps,
                    clearance_margin_m=self.trajectory_collision_margin_m,
                    max_local_candidates=self.field_local_rollout_candidates,
                    allow_direct_edge=self.planner_direct_edge,
                    shortcut_path=self.planner_shortcut,
                )
            else:
                raw_plan = self.planner.plan(
                    q_start, q_goal, self.step_size_q, self.rollout_max_steps, mode=self.planner_mode
                )
            plan = self._shortcut_plan(checker, raw_plan)
            path_ok, min_clearance = self._validate_plan_collision(checker, plan)
            self.get_logger().info(
                f"Field path test candidate={idx}/{len(valid_goals)} "
                f"goal_q={np.round(q_goal, 3).tolist()} raw_waypoints={len(raw_plan)} plan_waypoints={len(plan)} "
                f"raw_len={self._plan_length(raw_plan):.3f} plan_len={self._plan_length(plan):.3f} "
                f"min_clearance_m={min_clearance:.4f} path_ok={path_ok} "
                f"min_clearance_idx={self.last_validation_min_idx} "
                f"min_clearance_q={np.round(self.last_validation_min_q, 3).tolist() if self.last_validation_min_q is not None else None}"
            )
            if path_ok:
                best_plan = plan
                best_goal = q_goal
                best_clearance = min_clearance
                break
            if min_clearance > best_clearance:
                best_plan = plan
                best_goal = q_goal
                best_clearance = min_clearance
        if best_plan is None or best_goal is None:
            self.get_logger().warn("Field path test could not produce any plan.")
            return
        self.get_logger().info(
            f"Field path test selected goal_q={np.round(best_goal, 3).tolist()} "
            f"start_q={np.round(q_start, 3).tolist()} plan_waypoints={len(best_plan)} "
            f"min_clearance_m={best_clearance:.4f}"
        )
        self._publish_goal_markers(q_start, best_goal, None)
        self._publish_planned_path(best_plan)
        if self.publish_trajectory:
            self._publish_joint_trajectory(best_plan)

    def _wrap_plan_near_start(self, plan: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        pts = np.asarray(plan, dtype=np.float64).copy()
        q_ref = np.asarray(q_start, dtype=np.float64).copy()
        if pts.ndim != 2 or len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        for idx in range(len(pts)):
            for j in range(6):
                delta = pts[idx, j] - q_ref[j]
                pts[idx, j] = q_ref[j] + np.arctan2(np.sin(delta), np.cos(delta))
            q_ref = pts[idx]
        return pts

    def _prepare_execution_plan(self, plan: np.ndarray, current_q: np.ndarray | None) -> np.ndarray:
        pts = np.asarray(plan, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]
        if len(pts) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        if current_q is not None:
            current_q = np.asarray(current_q, dtype=np.float64)
            pts = self._wrap_plan_near_start(pts, current_q)
            if np.all(np.isfinite(current_q)):
                pts[0] = current_q
        sampled = [pts[0].copy()]
        max_delta_step = max(1e-3, self.trajectory_max_joint_speed * self.trajectory_min_segment_dt)
        for q in pts[1:]:
            prev = sampled[-1]
            max_delta = float(np.max(np.abs(q - prev)))
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

    def _publish_goal_markers(self, q_start: np.ndarray, q_goal: np.ndarray, clicked_target_base: np.ndarray | None):
        arr = MarkerArray()
        for idx, (q, ns, color) in enumerate(
            (
                (q_start, "field_test_start", (0.1, 1.0, 0.2)),
                (q_goal, "field_test_goal", (0.1, 0.7, 1.0)),
            )
        ):
            marker = Marker()
            marker.header = Header(frame_id=self.visualization_frame if self.visualization_frame else self.base_frame)
            marker.ns = ns
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = 0.95
            tool_pose = self.kinematics.fk(np.asarray(q, dtype=np.float64))
            camera_pose = self.kinematics.tool_to_camera_pose(tool_pose, self.camera_in_tool)
            vis = self._transform_point_for_visualization(camera_pose[:3, 3])
            marker.pose.position.x = float(vis[0])
            marker.pose.position.y = float(vis[1])
            marker.pose.position.z = float(vis[2])
            marker.pose.orientation.w = 1.0
            arr.markers.append(marker)
        if clicked_target_base is not None:
            marker = Marker()
            marker.header = Header(frame_id=self.visualization_frame if self.visualization_frame else self.base_frame)
            marker.ns = "field_test_clicked_target"
            marker.id = len(arr.markers)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.07
            marker.scale.y = 0.07
            marker.scale.z = 0.07
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.05
            marker.color.a = 0.95
            vis = self._transform_point_for_visualization(clicked_target_base)
            marker.pose.position.x = float(vis[0])
            marker.pose.position.y = float(vis[1])
            marker.pose.position.z = float(vis[2])
            marker.pose.orientation.w = 1.0
            arr.markers.append(marker)
        self.goal_marker_pub.publish(arr)

    def _publish_planned_path(self, plan: np.ndarray):
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
            pose.header = Header(frame_id=self.visualization_frame if self.visualization_frame else self.base_frame)
            pose.pose.position.x = float(camera_xyz[0])
            pose.pose.position.y = float(camera_xyz[1])
            pose.pose.position.z = float(camera_xyz[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_joint_trajectory(self, plan: np.ndarray):
        current_q = None if self.current_joints is None else np.asarray(self.current_joints, dtype=np.float64)
        exec_plan = self._prepare_execution_plan(plan, current_q)
        if len(exec_plan) == 0:
            self.get_logger().warn("Skipping field test trajectory publish because execution plan is empty.")
            return
        msg = JointTrajectory()
        msg.header = Header(frame_id=self.base_frame)
        msg.joint_names = JOINT_NAMES
        t_cur = 0.5
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
        self.get_logger().info(
            f"Publishing field test trajectory: exec_waypoints={len(msg.points)} estimated_duration_s={t_cur:.1f}"
        )
        self.trajectory_pub.publish(msg)


def main():
    rclpy.init()
    node = FieldPathTest()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
