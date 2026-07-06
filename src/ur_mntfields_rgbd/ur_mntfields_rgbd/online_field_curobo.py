from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from ur_mntfields_arm.tb_core.model_base import Model

from .sampling import (
    calculate_speed_and_normal,
    compute_normalization_bound,
    depth_to_meters,
    ray_dirs_camera,
    rotation_distance_deg,
    sample_training_frame,
    transform_to_matrix,
    translation_distance,
)


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def look_at_quaternion(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    forward = _normalize(target_xyz - camera_xyz)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _normalize(np.cross(world_up, forward))
    up = _normalize(np.cross(forward, right))
    rot = np.stack((forward, -right, -up), axis=1)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    return quat


def parse_box_specs(specs: list[str]) -> np.ndarray:
    boxes = []
    for raw in specs:
        text = str(raw).strip()
        if not text:
            continue
        vals = [float(v.strip()) for v in text.split(",")]
        if len(vals) != 6:
            continue
        boxes.append(vals)
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def boxes_to_bounds(boxes_xyzwhd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * boxes_xyzwhd[:, 3:6]
    mins = boxes_xyzwhd[:, 0:3] - half
    maxs = boxes_xyzwhd[:, 0:3] + half
    return mins.min(axis=0).astype(np.float32), maxs.max(axis=0).astype(np.float32)


class CuroboPlanner:
    def __init__(
        self,
        logger,
        robot_config: str,
        scene_config: str,
        tool_frame: str,
        device: str,
        warmup_iterations: int,
    ) -> None:
        self._logger = logger
        self._robot_config = robot_config.strip()
        self._scene_config = scene_config.strip()
        self._tool_frame = tool_frame.strip()
        self._device = device
        self._warmup_iterations = warmup_iterations
        self._planner = None
        self._joint_names = None
        self._tool_frames = None
        self._load_error = None

    @property
    def ready(self) -> bool:
        return self._planner is not None

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names or JOINT_NAMES)

    def initialize(self) -> bool:
        if self._planner is not None:
            return True
        if not self._robot_config:
            self._load_error = "robot_config parameter is empty"
            return False
        _ensure_curobo_importable()
        try:
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to import curobo: {exc}"
            return False

        try:
            scene_model = self._scene_config or None
            cfg = MotionPlannerCfg.create(
                robot=self._robot_config,
                scene_model=scene_model,
            )
            planner = MotionPlanner(cfg)
            planner.warmup(enable_graph=True, num_warmup_iterations=self._warmup_iterations)
            self._planner = planner
            self._joint_names = list(planner.joint_names)
            self._tool_frames = list(planner.tool_frames)
            if not self._tool_frame:
                self._tool_frame = self._tool_frames[0]
            self._logger.info(
                f"cuRobo planner ready: tool_frame={self._tool_frame} joints={self._joint_names}"
            )
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to initialize curobo planner: {exc}"
            self._planner = None
            return False

    def plan_pose(self, current_q: np.ndarray, goal_pose_msg: PoseStamped) -> Optional[JointTrajectory]:
        if not self.initialize():
            return None
        try:
            from curobo.types import GoalToolPose, JointState as CuroboJointState, Pose
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to import curobo types: {exc}"
            return None

        try:
            position = torch.tensor(
                [[
                    float(goal_pose_msg.pose.position.x),
                    float(goal_pose_msg.pose.position.y),
                    float(goal_pose_msg.pose.position.z),
                ]],
                device=self._device,
                dtype=torch.float32,
            )
            quaternion = torch.tensor(
                [[
                    float(goal_pose_msg.pose.orientation.w),
                    float(goal_pose_msg.pose.orientation.x),
                    float(goal_pose_msg.pose.orientation.y),
                    float(goal_pose_msg.pose.orientation.z),
                ]],
                device=self._device,
                dtype=torch.float32,
            )
            goal_pose = Pose(position=position, quaternion=quaternion)
            goal_tool_poses = GoalToolPose.from_poses({self._tool_frame: goal_pose}, num_goalset=1)
            current_state = CuroboJointState.from_position(
                torch.tensor(current_q, device=self._device, dtype=torch.float32).unsqueeze(0),
                joint_names=self._joint_names,
            )
            result = self._planner.plan_pose(goal_tool_poses, current_state)
            if result is None or not bool(result.success.any().item()):
                self._load_error = f"planning failed with status={getattr(result, 'status', 'unknown')}"
                return None

            interpolated = result.get_interpolated_plan()
            positions = np.asarray(interpolated.position.detach().cpu().numpy(), dtype=np.float32)
            while positions.ndim > 2:
                positions = positions[0]
            if positions.ndim == 1:
                positions = positions[None, :]
            dt_val = getattr(interpolated, "dt", None)
            if torch.is_tensor(dt_val):
                dt_sec = float(dt_val.detach().cpu().reshape(-1)[0].item())
            elif dt_val is None:
                dt_sec = 0.02
            else:
                dt_sec = float(np.asarray(dt_val).reshape(-1)[0])
            return build_joint_trajectory(self._joint_names, positions, dt_sec)
        except Exception as exc:  # pragma: no cover
            self._load_error = f"planning exception: {exc}\n{traceback.format_exc(limit=5)}"
            return None

    def last_error(self) -> str:
        return self._load_error or "unknown cuRobo error"


def normalize_pairs(pair_world: np.ndarray, bound: np.ndarray) -> np.ndarray:
    repeats = pair_world.shape[1] // 3
    bound_len = np.tile(bound[1] - bound[0], repeats)
    bound_start = np.tile(bound[0], repeats)
    return ((pair_world - bound_start) / bound_len - 0.5).astype(np.float32)


def rollout_path(
    function,
    device: str,
    bound: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    step_size: float,
    alpha: float,
    max_steps: int,
    goal_tolerance: float,
) -> tuple[np.ndarray, bool]:
    cur = start_xyz.astype(np.float32).copy()
    goal = np.clip(goal_xyz.astype(np.float32), bound[0], bound[1])
    path = [cur.copy()]

    for _ in range(max_steps):
        delta = goal - cur
        dist = float(np.linalg.norm(delta))
        if dist <= goal_tolerance:
            path.append(goal.copy())
            return np.stack(path, axis=0), True

        geom_dir = delta / max(dist, 1e-8)
        pair = np.concatenate((cur, goal), axis=0)[None, :]
        pair_n = normalize_pairs(pair, bound)
        pair_t = torch.from_numpy(pair_n).float().to(device).requires_grad_(True)
        tau, _w, coords = function.network.out(pair_t)
        grad = torch.autograd.grad(
            outputs=tau.sum(),
            inputs=coords,
            retain_graph=False,
            create_graph=False,
        )[0]
        grad_xyz_norm = grad[0, :3].detach().cpu().numpy().astype(np.float32)
        bound_len = (bound[1] - bound[0]).astype(np.float32)
        grad_xyz = grad_xyz_norm / (bound_len + 1e-8)
        grad_norm = float(np.linalg.norm(grad_xyz))
        if grad_norm < 1e-6:
            direction = geom_dir
        else:
            field_dir = -grad_xyz / grad_norm
            direction = alpha * field_dir + (1.0 - alpha) * geom_dir
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        step = min(step_size, dist)
        nxt = cur + direction.astype(np.float32) * step
        nxt = np.minimum(np.maximum(nxt, bound[0]), bound[1])
        if float(np.linalg.norm(nxt - cur)) < 1e-6:
            break
        cur = nxt.astype(np.float32)
        path.append(cur.copy())

    return np.stack(path, axis=0), False


def build_joint_trajectory(joint_names: list[str], positions: np.ndarray, dt_sec: float) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = list(joint_names)
    dt_sec = max(float(dt_sec), 1e-3)
    for idx, q in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        total = (idx + 1) * dt_sec
        sec = int(total)
        nsec = int((total - sec) * 1.0e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        traj.points.append(pt)
    return traj


def _ensure_curobo_importable() -> None:
    candidate_roots = []
    env_root = os.environ.get("CUROBO_ROOT", "").strip()
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    candidate_roots.append(Path("/home/mayank/curobo"))

    for root in candidate_roots:
        if not root.exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        venv_site = root / ".venv" / "lib"
        if venv_site.exists():
            for py_dir in sorted(venv_site.glob("python*/site-packages")):
                py_dir_str = str(py_dir)
                if py_dir_str not in sys.path:
                    sys.path.insert(0, py_dir_str)


def subsample_anchor_points(path_xyz: np.ndarray, stride: int) -> np.ndarray:
    if len(path_xyz) <= 2:
        return path_xyz.astype(np.float32)
    stride = max(1, int(stride))
    anchors = [path_xyz[0]]
    for idx in range(stride, len(path_xyz) - 1, stride):
        anchors.append(path_xyz[idx])
    anchors.append(path_xyz[-1])
    return np.asarray(anchors, dtype=np.float32)


def merge_joint_trajectories(segments: list[JointTrajectory]) -> Optional[JointTrajectory]:
    if not segments:
        return None
    merged = JointTrajectory()
    merged.joint_names = list(segments[0].joint_names)
    time_offset = 0.0
    for seg_idx, seg in enumerate(segments):
        for pt_idx, pt in enumerate(seg.points):
            if seg_idx > 0 and pt_idx == 0:
                continue
            new_pt = JointTrajectoryPoint()
            new_pt.positions = list(pt.positions)
            if pt.velocities:
                new_pt.velocities = list(pt.velocities)
            if pt.accelerations:
                new_pt.accelerations = list(pt.accelerations)
            cur_t = float(pt.time_from_start.sec) + 1.0e-9 * float(pt.time_from_start.nanosec)
            total = time_offset + cur_t
            sec = int(total)
            nsec = int(round((total - sec) * 1.0e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            new_pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            merged.points.append(new_pt)
        if seg.points:
            last = seg.points[-1].time_from_start
            time_offset += float(last.sec) + 1.0e-9 * float(last.nanosec)
    return merged


class OnlineRGBDFieldCurobo(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_field_curobo_pipeline")

        self._declare_sampler_params()
        self._declare_training_params()
        self._declare_execution_params()

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame_override = self.get_parameter("camera_frame").value
        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.n_rays = int(self.get_parameter("n_rays").value)
        self.n_strat_samples = int(self.get_parameter("n_strat_samples").value)
        self.dist_behind_surf = float(self.get_parameter("dist_behind_surf").value)
        self.sample_min = float(self.get_parameter("sample_min").value)
        self.sample_max = float(self.get_parameter("sample_max").value)
        self.num_pairs = int(self.get_parameter("num_pairs").value)
        self.scale_factor = float(self.get_parameter("scale_factor").value)
        self.bound_padding_xy = float(self.get_parameter("bound_padding_xy").value)
        self.bound_padding_z = float(self.get_parameter("bound_padding_z").value)
        self.capture_stride = max(1, int(self.get_parameter("capture_stride").value))
        self.min_translation_m = float(self.get_parameter("min_translation_m").value)
        self.min_rotation_deg = float(self.get_parameter("min_rotation_deg").value)
        self.save_color = bool(self.get_parameter("save_color").value)
        self.save_depth_npy = bool(self.get_parameter("save_depth_npy").value)
        self.train_every_n_captures = max(1, int(self.get_parameter("train_every_n_captures").value))
        self.train_epochs = max(1, int(self.get_parameter("train_epochs").value))
        self.max_raw_rows = max(1, int(self.get_parameter("max_raw_rows").value))
        self.max_train_rows = max(1, int(self.get_parameter("max_train_rows").value))
        self.model_dir = Path(self.get_parameter("model_dir").value).expanduser().resolve()
        self.model_device = str(self.get_parameter("model_device").value)
        self.model_lr = float(self.get_parameter("model_lr").value)
        self.rollout_step_size = float(self.get_parameter("rollout_step_size").value)
        self.rollout_alpha = float(self.get_parameter("rollout_alpha").value)
        self.rollout_max_steps = int(self.get_parameter("rollout_max_steps").value)
        self.rollout_goal_tolerance = float(self.get_parameter("rollout_goal_tolerance").value)
        self.enable_curobo = bool(self.get_parameter("enable_curobo").value)
        self.curobo_follow_mode = str(self.get_parameter("curobo_follow_mode").value)
        self.curobo_anchor_stride = max(1, int(self.get_parameter("curobo_anchor_stride").value))
        self.trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("color", "depth", "pose", "samples"):
            (self.output_dir / name).mkdir(exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_pose = None
        self.frame_index = 0
        self.capture_count = 0
        self.dirs_c = None
        self.camera_intrinsics = None
        self.current_joint_positions = None
        self.current_camera_pose = None
        self.pending_goal = None
        self.active_goal = None
        self.last_command_time = 0.0
        self.raw_history = []
        self.camera_pose_history = []
        self.bound = None
        self.model = None
        self.last_train_loss = None
        self.last_workspace_path = None
        self.scene_boxes = parse_box_specs(
            [str(v) for v in self.get_parameter("scene_boxes").value]
        )

        self.path_pub = self.create_publisher(PoseArray, "~/workspace_path", 5)
        self.nav_path_pub = self.create_publisher(NavPath, "/ur_mntfields_arm/planned_path", 5)
        self.frontier_pub = self.create_publisher(MarkerArray, "/ur_mntfields_arm/frontiers", 5)
        self.traj_pub = self.create_publisher(JointTrajectory, self.trajectory_topic, 5)
        self.goal_sub = self.create_subscription(PoseStamped, "~/goal_pose", self._goal_cb, 5)
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)

        queue_size = int(self.get_parameter("queue_size").value)
        sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=queue_size,
            slop=sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.curobo = CuroboPlanner(
            logger=self.get_logger(),
            robot_config=str(self.get_parameter("curobo_robot_config").value),
            scene_config=str(self.get_parameter("curobo_scene_config").value),
            tool_frame=str(self.get_parameter("curobo_tool_frame").value),
            device=str(self.get_parameter("curobo_device").value),
            warmup_iterations=int(self.get_parameter("curobo_warmup_iterations").value),
        )
        if self.enable_curobo:
            if not self.curobo.initialize():
                self.get_logger().warning(f"cuRobo disabled at startup: {self.curobo.last_error()}")
                self.enable_curobo = False

        self.get_logger().info(
            f"Online 3D field pipeline ready: output_dir={self.output_dir} model_dir={self.model_dir} "
            f"pairs_per_capture={self.num_pairs} train_every_n_captures={self.train_every_n_captures} "
            f"enable_curobo={self.enable_curobo} curobo_follow_mode={self.curobo_follow_mode}"
        )

    def _declare_sampler_params(self) -> None:
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("output_dir", "/tmp/mntfields_rgbd_output")
        self.declare_parameter("save_color", True)
        self.declare_parameter("save_depth_npy", True)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth", 0.15)
        self.declare_parameter("max_depth", 2.5)
        self.declare_parameter("n_rays", 5000)
        self.declare_parameter("n_strat_samples", 8)
        self.declare_parameter("dist_behind_surf", 0.4)
        self.declare_parameter("sample_min", 0.07)
        self.declare_parameter("sample_max", 0.3)
        self.declare_parameter("num_pairs", 10000)
        self.declare_parameter("scale_factor", 1.0)
        self.declare_parameter("bound_padding_xy", 0.05)
        self.declare_parameter("bound_padding_z", 0.05)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("capture_stride", 1)
        self.declare_parameter("min_translation_m", 0.0)
        self.declare_parameter("min_rotation_deg", 0.0)
        self.declare_parameter("random_seed", 0)

    def _declare_training_params(self) -> None:
        self.declare_parameter("train_every_n_captures", 4)
        self.declare_parameter("train_epochs", 20)
        self.declare_parameter("max_raw_rows", 120000)
        self.declare_parameter("max_train_rows", 30000)
        self.declare_parameter("model_dir", "/tmp/mntfields_rgbd_model")
        self.declare_parameter("model_device", "cuda:0")
        self.declare_parameter("model_lr", 1.0e-4)
        self.declare_parameter("save_training_dumps", True)

    def _declare_execution_params(self) -> None:
        self.declare_parameter("rollout_step_size", 0.05)
        self.declare_parameter("rollout_alpha", 0.8)
        self.declare_parameter("rollout_max_steps", 120)
        self.declare_parameter("rollout_goal_tolerance", 0.05)
        self.declare_parameter("enable_curobo", False)
        self.declare_parameter("curobo_follow_mode", "final_pose")
        self.declare_parameter("curobo_anchor_stride", 8)
        self.declare_parameter("trajectory_topic", "/ur_mntfields_arm/joint_trajectory")
        self.declare_parameter("curobo_robot_config", "")
        self.declare_parameter("curobo_scene_config", "")
        self.declare_parameter("curobo_tool_frame", "")
        self.declare_parameter("curobo_device", "cuda:0")
        self.declare_parameter("curobo_warmup_iterations", 5)
        self.declare_parameter("enable_nbv", False)
        self.declare_parameter("nbv_roi_center", [0.55, 0.0, 0.75])
        self.declare_parameter("scene_boxes", [""])
        self.declare_parameter("scene_boxes_frame", "world")
        self.declare_parameter("nbv_bbox_padding_xyz", [0.02, 0.02, 0.02])
        self.declare_parameter("nbv_preferred_standoff_m", 0.30)
        self.declare_parameter("nbv_lateral_offsets", [-0.20, 0.0, 0.20])
        self.declare_parameter("nbv_height_offsets", [-0.18, 0.0, 0.18])
        self.declare_parameter("nbv_num_azimuth_samples", 12)
        self.declare_parameter("nbv_min_view_change_m", 0.15)
        self.declare_parameter("nbv_goal_reached_tolerance_m", 0.08)
        self.declare_parameter("nbv_command_cooldown_s", 3.0)

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        if not all(name in name_to_idx for name in JOINT_NAMES):
            return
        self.current_joint_positions = np.array(
            [float(msg.position[name_to_idx[name]]) for name in JOINT_NAMES],
            dtype=np.float32,
        )

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.pending_goal = msg
        self.active_goal = None
        self._publish_goal_marker(msg)
        self.get_logger().info(
            f"Received workspace goal: xyz=[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]"
        )
        self._try_plan_and_execute()

    def synced_callback(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        self.frame_index += 1
        if self.frame_index % self.capture_stride != 0:
            return

        frame_id = self.camera_frame_override or depth_msg.header.frame_id or color_msg.header.frame_id
        if not frame_id:
            return

        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, frame_id, depth_msg.header.stamp)
        except TransformException as exc:
            err = str(exc)
            if "extrapolation" in err.lower() or "future" in err.lower():
                try:
                    transform = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
                    self.get_logger().info(
                        f"Using latest TF for {self.base_frame} -> {frame_id} after timestamp miss."
                    )
                except TransformException as latest_exc:
                    self.get_logger().warning(
                        f"TF lookup failed for {self.base_frame} -> {frame_id}: {latest_exc}"
                    )
                    return
            else:
                self.get_logger().warning(f"TF lookup failed for {self.base_frame} -> {frame_id}: {exc}")
                return

        t_world_camera = transform_to_matrix(transform.transform.translation, transform.transform.rotation)
        self.current_camera_pose = t_world_camera
        self._check_active_goal_reached()

        if self.last_pose is not None:
            moved = translation_distance(t_world_camera, self.last_pose)
            rotated = rotation_distance_deg(t_world_camera, self.last_pose)
            if moved < self.min_translation_m and rotated < self.min_rotation_deg:
                return

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        height = int(info_msg.height)
        width = int(info_msg.width)
        fx = float(info_msg.k[0])
        fy = float(info_msg.k[4])
        cx = float(info_msg.k[2])
        cy = float(info_msg.k[5])

        if self.dirs_c is None or self.camera_intrinsics != (height, width, fx, fy, cx, cy):
            self.dirs_c = ray_dirs_camera(height, width, fx, fy, cx, cy)
            self.camera_intrinsics = (height, width, fx, fy, cx, cy)

        depth_m = depth_to_meters(depth, self.depth_scale, self.max_depth)
        try:
            samples = sample_training_frame(
                depth_m=depth_m,
                t_world_camera=t_world_camera,
                dirs_c=self.dirs_c,
                n_rays=self.n_rays,
                n_strat_samples=self.n_strat_samples,
                dist_behind_surf=self.dist_behind_surf,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                sample_min=self.sample_min,
                sample_max=self.sample_max,
                num_pairs=self.num_pairs,
                scale_factor=self.scale_factor,
                bound_padding_xy=self.bound_padding_xy,
                bound_padding_z=self.bound_padding_z,
                rng=self.rng,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Skipping frame: {exc}")
            return

        if samples["frame_data"].shape[0] == 0:
            self.get_logger().warning("Skipping frame: raw data converted to empty Nx14 training data.")
            return

        self.raw_history.append(samples["raw_frame_data"].astype(np.float32))
        self.camera_pose_history.append(t_world_camera.astype(np.float32))

        stem = f"{self.capture_count:06d}"
        np.save(self.output_dir / "pose" / f"{stem}.npy", t_world_camera)
        np.savez_compressed(
            self.output_dir / "samples" / f"{stem}.npz",
            frame_data=samples["frame_data"],
            raw_frame_data=samples["raw_frame_data"],
            normalization_bound=samples["normalization_bound"],
            camera_pose_world=samples["camera_pose_world"],
            camera_intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
        )
        if self.save_color:
            cv2.imwrite(str(self.output_dir / "color" / f"{stem}.png"), color)
        if self.save_depth_npy:
            np.save(self.output_dir / "depth" / f"{stem}.npy", depth_m)

        frame_data = samples["frame_data"]
        q0_clearance = frame_data[:, 6]
        q1_clearance = frame_data[:, 7]
        self.get_logger().info(
            f"Capture {stem}: train_rows={len(frame_data)} q0_clearance_mean={float(np.mean(q0_clearance)):.4f} "
            f"q1_clearance_mean={float(np.mean(q1_clearance)):.4f} speed0_sat_frac={float(np.mean(q0_clearance >= 0.999)):.3f} "
            f"speed1_sat_frac={float(np.mean(q1_clearance >= 0.999)):.3f}"
        )

        self.last_pose = t_world_camera
        self.capture_count += 1

        if self.capture_count % self.train_every_n_captures == 0:
            self._train_field()
            self._maybe_schedule_nbv_goal()
            self._try_plan_and_execute()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self.model = Model(
            folder=str(self.model_dir),
            dim=3,
            B_scale=10,
            device=self.model_device,
            init_network=True,
            eval=False,
            lr=self.model_lr,
        )

    def _train_field(self) -> None:
        if not self.raw_history:
            return
        raw = np.concatenate(self.raw_history, axis=0)
        if len(raw) > self.max_raw_rows:
            idx = self.rng.choice(len(raw), size=self.max_raw_rows, replace=False)
            raw = raw[idx]
        self.bound = compute_normalization_bound(raw.reshape(-1, 3), self.bound_padding_xy, self.bound_padding_z)
        frame_data = calculate_speed_and_normal(raw, self.bound, self.sample_min, self.sample_max)
        if len(frame_data) == 0:
            self.get_logger().warning("Training skipped: no valid Nx14 rows after global normalization.")
            return
        if len(frame_data) > self.max_train_rows:
            idx = self.rng.choice(len(frame_data), size=self.max_train_rows, replace=False)
            frame_data = frame_data[idx]
        self._ensure_model()
        frame_t = torch.from_numpy(frame_data).float()
        loss, _ = self.model.train_core(self.train_epochs, frame_data=frame_t, is_one_frame=True)
        self.last_train_loss = None if loss is None else float(loss)
        self._save_model_artifacts(raw, frame_data)
        self.get_logger().info(
            f"3D field training complete: captures={self.capture_count} raw_rows={len(raw)} train_rows={len(frame_data)} loss={self.last_train_loss}"
        )

    def _save_model_artifacts(self, raw: np.ndarray, frame_data: np.ndarray) -> None:
        if self.model is None or self.bound is None:
            return
        torch.save(self.model.B, self.model_dir / "B.pt")
        torch.save(self.model.network.state_dict(), self.model_dir / "network.pt")
        source_xyz = self._current_start_xyz()
        np.savez_compressed(
            self.model_dir / "field_metadata.npz",
            normalization_bound=self.bound.astype(np.float32),
            source_xyz=source_xyz.astype(np.float32),
            raw_rows=int(len(raw)),
            train_rows=int(len(frame_data)),
            last_loss=np.array([-1.0 if self.last_train_loss is None else self.last_train_loss], dtype=np.float32),
        )

    def _current_start_xyz(self) -> np.ndarray:
        if self.current_camera_pose is not None:
            return self.current_camera_pose[:3, 3].astype(np.float32)
        if self.camera_pose_history:
            return self.camera_pose_history[-1][:3, 3].astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _publish_workspace_path(self, path_xyz: np.ndarray, goal_pose: PoseStamped) -> None:
        msg = PoseArray()
        msg.header = goal_pose.header
        nav = NavPath()
        nav.header = goal_pose.header
        for xyz in path_xyz:
            pose = Pose()
            pose.position.x = float(xyz[0])
            pose.position.y = float(xyz[1])
            pose.position.z = float(xyz[2])
            pose.orientation = goal_pose.pose.orientation
            msg.poses.append(pose)
            ps = PoseStamped()
            ps.header = goal_pose.header
            ps.pose = pose
            nav.poses.append(ps)
        self.path_pub.publish(msg)
        self.nav_path_pub.publish(nav)

    def _publish_goal_marker(self, goal: Optional[PoseStamped]) -> None:
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = "frontiers"
        marker.id = 0
        if goal is None:
            marker.action = Marker.DELETE
            arr.markers.append(marker)
            arrow = Marker()
            arrow.header.frame_id = self.base_frame
            arrow.header.stamp = stamp
            arrow.ns = "frontiers"
            arrow.id = 1
            arrow.action = Marker.DELETE
            arr.markers.append(arrow)
            self.frontier_pub.publish(arr)
            return
        marker.action = Marker.ADD
        marker.type = Marker.SPHERE
        marker.pose = goal.pose
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.a = 0.95
        marker.color.r = 0.10
        marker.color.g = 0.80
        marker.color.b = 0.20
        arr.markers.append(marker)

        roi = None
        bbox = self._scene_bbox_in_base()
        if bbox is not None:
            roi = 0.5 * (bbox[0] + bbox[1])
        else:
            roi = np.asarray(self.get_parameter("nbv_roi_center").value, dtype=np.float32)
        arrow = Marker()
        arrow.header.frame_id = self.base_frame
        arrow.header.stamp = stamp
        arrow.ns = "frontiers"
        arrow.id = 1
        arrow.action = Marker.ADD
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.02
        arrow.scale.y = 0.04
        arrow.scale.z = 0.06
        arrow.color.a = 0.95
        arrow.color.r = 0.95
        arrow.color.g = 0.20
        arrow.color.b = 0.10
        p0 = Pose().position
        p0.x = goal.pose.position.x
        p0.y = goal.pose.position.y
        p0.z = goal.pose.position.z
        p1 = Pose().position
        p1.x = float(roi[0])
        p1.y = float(roi[1])
        p1.z = float(roi[2])
        arrow.points = [p0, p1]
        arr.markers.append(arrow)
        self.frontier_pub.publish(arr)

    def _check_active_goal_reached(self) -> None:
        if self.active_goal is None or self.current_camera_pose is None:
            return
        goal_xyz = np.array(
            [
                float(self.active_goal.pose.position.x),
                float(self.active_goal.pose.position.y),
                float(self.active_goal.pose.position.z),
            ],
            dtype=np.float32,
        )
        cur_xyz = self.current_camera_pose[:3, 3].astype(np.float32)
        tol = float(self.get_parameter("nbv_goal_reached_tolerance_m").value)
        if float(np.linalg.norm(cur_xyz - goal_xyz)) <= tol:
            self.get_logger().info(
                f"Reached active goal within {tol:.3f} m at xyz={goal_xyz.tolist()}."
            )
            self.active_goal = None
            self._publish_goal_marker(None)
            self._maybe_schedule_nbv_goal()
            self._try_plan_and_execute()

    def _make_pose_stamped(self, xyz: np.ndarray, quat_wxyz: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        return msg

    def _lookup_frame_transform(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException:
            return None
        return transform_to_matrix(tf_msg.transform.translation, tf_msg.transform.rotation)

    def _scene_bbox_in_base(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.scene_boxes.size == 0:
            return None
        boxes = self.scene_boxes.copy()
        scene_frame = str(self.get_parameter("scene_boxes_frame").value).strip()
        if scene_frame and scene_frame != self.base_frame:
            tf_mat = self._lookup_frame_transform(self.base_frame, scene_frame)
            if tf_mat is None:
                return None
            centers_h = np.concatenate(
                (boxes[:, 0:3], np.ones((len(boxes), 1), dtype=np.float32)), axis=1
            )
            centers_base = (tf_mat @ centers_h.T).T[:, 0:3]
            boxes[:, 0:3] = centers_base.astype(np.float32)
        bmin, bmax = boxes_to_bounds(boxes)
        pad = np.asarray(self.get_parameter("nbv_bbox_padding_xyz").value, dtype=np.float32)
        return bmin - pad, bmax + pad

    def _select_nbv_goal(self) -> Optional[PoseStamped]:
        if not bool(self.get_parameter("enable_nbv").value):
            return None
        if self.current_camera_pose is None:
            return None
        bbox = self._scene_bbox_in_base()
        if bbox is not None:
            bmin, bmax = bbox
            roi = 0.5 * (bmin + bmax)
            standoff = float(self.get_parameter("nbv_preferred_standoff_m").value)
            front_x = bmin[0] - standoff
            lat_offsets = [float(v) for v in self.get_parameter("nbv_lateral_offsets").value]
            z_offsets = [float(v) for v in self.get_parameter("nbv_height_offsets").value]
            current_xyz = self.current_camera_pose[:3, 3].astype(np.float32)
            min_change = float(self.get_parameter("nbv_min_view_change_m").value)
            pose_hist = [p[:3, 3].astype(np.float32) for p in self.camera_pose_history[-64:]]

            best_score = -1.0e9
            best_pose = None
            for lat in lat_offsets:
                for z_off in z_offsets:
                    xyz = np.array(
                        [front_x, roi[1] + lat, roi[2] + z_off],
                        dtype=np.float32,
                    )
                    if float(np.linalg.norm(xyz - current_xyz)) < min_change:
                        continue
                    novelty = 0.0
                    if pose_hist:
                        novelty = min(float(np.linalg.norm(xyz - p)) for p in pose_hist)
                    quat = look_at_quaternion(xyz, roi)
                    lateral_penalty = abs(float(lat))
                    vertical_penalty = abs(float(z_off))
                    score = 1.5 * novelty - 0.6 * lateral_penalty - 0.4 * vertical_penalty
                    if score > best_score:
                        best_score = score
                        best_pose = self._make_pose_stamped(xyz, quat)
            if best_pose is not None:
                return best_pose

        roi = np.asarray(self.get_parameter("nbv_roi_center").value, dtype=np.float32)
        radius = float(self.get_parameter("nbv_preferred_standoff_m").value) + 0.25
        z_offsets = [float(v) for v in self.get_parameter("nbv_height_offsets").value]
        n_az = max(4, int(self.get_parameter("nbv_num_azimuth_samples").value))
        current_xyz = self.current_camera_pose[:3, 3].astype(np.float32)
        min_change = float(self.get_parameter("nbv_min_view_change_m").value)
        pose_hist = [p[:3, 3].astype(np.float32) for p in self.camera_pose_history[-64:]]

        best_score = -1.0e9
        best_pose = None
        for z_off in z_offsets:
            for k in range(n_az):
                az = 2.0 * np.pi * float(k) / float(n_az)
                xyz = roi + np.array(
                    [radius * np.cos(az), radius * np.sin(az), z_off],
                    dtype=np.float32,
                )
                if xyz[2] < 0.15:
                    continue
                if float(np.linalg.norm(xyz - current_xyz)) < min_change:
                    continue
                novelty = 0.0
                if pose_hist:
                    novelty = min(float(np.linalg.norm(xyz - p)) for p in pose_hist)
                face_quat = look_at_quaternion(xyz, roi)
                height_penalty = abs(float(xyz[2] - roi[2]))
                score = 1.5 * novelty - 0.2 * height_penalty
                if score > best_score:
                    best_score = score
                    best_pose = self._make_pose_stamped(xyz, face_quat)
        return best_pose

    def _maybe_schedule_nbv_goal(self) -> None:
        if not bool(self.get_parameter("enable_nbv").value):
            return
        if self.pending_goal is not None or self.active_goal is not None:
            return
        now = time.monotonic()
        if now - self.last_command_time < float(self.get_parameter("nbv_command_cooldown_s").value):
            return
        goal = self._select_nbv_goal()
        if goal is None:
            return
        self.pending_goal = goal
        self._publish_goal_marker(goal)
        self.get_logger().info(
            f"Selected NBV goal: xyz=[{goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, {goal.pose.position.z:.3f}]"
        )

    def _try_plan_and_execute(self) -> None:
        if self.pending_goal is None:
            return
        if self.active_goal is not None:
            return
        if self.current_camera_pose is None:
            self.get_logger().info("Goal planning waiting for current camera pose.")
            return

        start_xyz = self._current_start_xyz()
        goal_xyz = np.array(
            [
                float(self.pending_goal.pose.position.x),
                float(self.pending_goal.pose.position.y),
                float(self.pending_goal.pose.position.z),
            ],
            dtype=np.float32,
        )

        if self.model is not None and self.bound is not None:
            path_xyz, success = rollout_path(
                function=self.model.function,
                device=self.model.device,
                bound=self.bound,
                start_xyz=start_xyz,
                goal_xyz=goal_xyz,
                step_size=self.rollout_step_size,
                alpha=self.rollout_alpha,
                max_steps=self.rollout_max_steps,
                goal_tolerance=self.rollout_goal_tolerance,
            )
            self.last_workspace_path = path_xyz
            self._publish_workspace_path(path_xyz, self.pending_goal)
            self.get_logger().info(
                f"Workspace field rollout: waypoints={len(path_xyz)} success={success} start={start_xyz.tolist()} goal={goal_xyz.tolist()}"
            )
        else:
            path_xyz = np.stack((start_xyz, goal_xyz), axis=0)
            self.last_workspace_path = path_xyz
            self._publish_workspace_path(path_xyz, self.pending_goal)
            self.get_logger().info("Field not trained yet; published straight workspace segment only.")

        if not self.enable_curobo:
            return
        if self.current_joint_positions is None:
            self.get_logger().info("cuRobo planning waiting for joint states.")
            return
        traj = self._plan_curobo_trajectory(path_xyz, self.pending_goal)
        if traj is None:
            self.get_logger().warning(f"cuRobo planning failed: {self.curobo.last_error()}")
            return
        self.traj_pub.publish(traj)
        self.active_goal = self.pending_goal
        self.pending_goal = None
        self.last_command_time = time.monotonic()
        self.get_logger().info(
            f"Published cuRobo trajectory with {len(traj.points)} waypoints to {self.trajectory_topic}."
        )

    def _plan_curobo_trajectory(
        self,
        path_xyz: np.ndarray,
        goal_pose: PoseStamped,
    ) -> Optional[JointTrajectory]:
        mode = self.curobo_follow_mode.strip().lower()
        if mode == "final_pose":
            return self.curobo.plan_pose(self.current_joint_positions, goal_pose)
        if mode != "anchor_chain":
            self.get_logger().warning(
                f"Unknown cuRobo follow mode '{self.curobo_follow_mode}', falling back to final_pose."
            )
            return self.curobo.plan_pose(self.current_joint_positions, goal_pose)

        anchors = subsample_anchor_points(path_xyz, self.curobo_anchor_stride)
        if len(anchors) < 2:
            return self.curobo.plan_pose(self.current_joint_positions, goal_pose)

        current_q = self.current_joint_positions.copy()
        segments: list[JointTrajectory] = []
        self.get_logger().info(
            f"cuRobo anchor-chain planning: anchors={len(anchors)} stride={self.curobo_anchor_stride}"
        )
        for idx, xyz in enumerate(anchors[1:], start=1):
            anchor_goal = PoseStamped()
            anchor_goal.header = goal_pose.header
            anchor_goal.pose.position.x = float(xyz[0])
            anchor_goal.pose.position.y = float(xyz[1])
            anchor_goal.pose.position.z = float(xyz[2])
            anchor_goal.pose.orientation = goal_pose.pose.orientation
            seg = self.curobo.plan_pose(current_q, anchor_goal)
            if seg is None or not seg.points:
                self.get_logger().warning(f"cuRobo failed at anchor {idx}/{len(anchors)-1}.")
                return None
            segments.append(seg)
            current_q = np.asarray(seg.points[-1].positions, dtype=np.float32)
        return merge_joint_trajectories(segments)


def main() -> None:
    rclpy.init()
    node = OnlineRGBDFieldCurobo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
