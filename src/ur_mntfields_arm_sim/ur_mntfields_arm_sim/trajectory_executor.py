from __future__ import annotations

import time
from math import fabs

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class TrajectoryExecutor(Node):
    def __init__(self):
        super().__init__("trajectory_executor")
        self.declare_parameter("enable_startup_wiggle", False)
        self.declare_parameter("startup_wiggle_delay_s", 12.0)
        self.declare_parameter("startup_wiggle_delta", [0.12, -0.10, 0.10, 0.08, 0.0, 0.08])
        self.declare_parameter("enable_startup_home", True)
        self.declare_parameter("enable_startup_inspection", True)
        self.declare_parameter("home_delay_s", 2.0)
        self.declare_parameter("inspection_delay_s", 8.0)
        self.declare_parameter("startup_timer_period_s", 1.0)
        self.declare_parameter("pose_reached_tolerance_rad", 0.05)
        self.declare_parameter("pose_report_interval_s", 2.0)
        self.declare_parameter("home_positions", [0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        self.declare_parameter("inspection_positions", [0.35, -1.10, 1.10, -1.55, -1.57, 0.0])
        self.declare_parameter(
            "trajectory_action_name",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter("direct_publish_topic", "")
        self.action = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action_name").value),
        )
        self.direct_publish_topic = str(self.get_parameter("direct_publish_topic").value)
        self.direct_pub = (
            self.create_publisher(JointTrajectory, self.direct_publish_topic, 5)
            if self.direct_publish_topic
            else None
        )
        self.busy = False
        self.current_positions = None
        self.pending_target_name = None
        self.pending_target_positions = None
        self.queued_trajectory: JointTrajectory | None = None
        self.last_target_report_time = 0.0
        self.sent_startup_wiggle = False
        self.sent_startup_home = False
        self.sent_startup_inspection = False
        self.start_time = time.monotonic()
        self.enable_startup_wiggle = bool(self.get_parameter("enable_startup_wiggle").value)
        self.create_subscription(JointTrajectory, "/ur_mntfields_arm/joint_trajectory", self._traj_cb, 5)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)
        self.startup_wiggle_timer = self.create_timer(
            float(self.get_parameter("startup_timer_period_s").value), self._maybe_send_startup_wiggle
        )

    def _joint_state_cb(self, msg: JointState):
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        if not all(name in name_to_idx for name in JOINT_NAMES):
            return
        self.current_positions = [float(msg.position[name_to_idx[name]]) for name in JOINT_NAMES]
        self._check_pending_target()

    def _traj_cb(self, msg: JointTrajectory):
        if not msg.points:
            return
        if self.busy:
            self.queued_trajectory = msg
            self.get_logger().info("Queued trajectory while previous goal is executing.")
            return
        self.sent_startup_wiggle = True
        self._send_goal(msg)

    def _maybe_send_startup_wiggle(self):
        if self.busy:
            return
        elapsed = time.monotonic() - self.start_time
        home_positions = [float(v) for v in self.get_parameter("home_positions").value]
        inspection_positions = [float(v) for v in self.get_parameter("inspection_positions").value]
        if bool(self.get_parameter("enable_startup_home").value) and not self.sent_startup_home and elapsed >= float(self.get_parameter("home_delay_s").value):
            msg = JointTrajectory()
            msg.joint_names = JOINT_NAMES
            home = JointTrajectoryPoint()
            home.positions = home_positions
            home.time_from_start.sec = 4
            msg.points = [home]
            self.sent_startup_home = True
            self.get_logger().info(f"Sending startup home trajectory to {self._format_positions(home_positions)}.")
            self._send_goal(msg, target_name="startup_home", target_positions=home_positions)
            return
        home_completed = self.sent_startup_home or not bool(self.get_parameter("enable_startup_home").value)
        if bool(self.get_parameter("enable_startup_inspection").value) and home_completed and not self.sent_startup_inspection and elapsed >= float(self.get_parameter("inspection_delay_s").value):
            msg = JointTrajectory()
            msg.joint_names = JOINT_NAMES
            inspect = JointTrajectoryPoint()
            inspect.positions = inspection_positions
            inspect.time_from_start.sec = 5
            msg.points = [inspect]
            self.sent_startup_inspection = True
            self.get_logger().info(f"Sending startup inspection trajectory to {self._format_positions(inspection_positions)}.")
            self._send_goal(msg, target_name="startup_inspection", target_positions=inspection_positions)
            return
        if not self.enable_startup_wiggle or self.sent_startup_wiggle:
            return
        if self.current_positions is None:
            return
        if elapsed < float(self.get_parameter("startup_wiggle_delay_s").value):
            return
        delta = [float(v) for v in self.get_parameter("startup_wiggle_delta").value]
        q0 = inspection_positions if self.sent_startup_inspection else list(self.current_positions)
        q1 = [q + dq for q, dq in zip(q0, delta)]
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        start = JointTrajectoryPoint()
        start.positions = q0
        start.time_from_start.sec = 0
        goal = JointTrajectoryPoint()
        goal.positions = q1
        goal.time_from_start.sec = 2
        settle = JointTrajectoryPoint()
        settle.positions = q0
        settle.time_from_start.sec = 4
        msg.points = [start, goal, settle]
        self.sent_startup_wiggle = True
        self.get_logger().info("Sending startup wiggle trajectory.")
        self._send_goal(msg, target_name="startup_wiggle_return", target_positions=q0)

    def _send_goal(self, msg: JointTrajectory, target_name: str | None = None, target_positions: list[float] | None = None):
        self.pending_target_name = target_name
        self.pending_target_positions = list(target_positions) if target_positions is not None else None
        self.last_target_report_time = time.monotonic()
        if self.direct_pub is not None:
            self.busy = True
            self.direct_pub.publish(msg)
            self.get_logger().info(f"Published trajectory directly to {self.direct_publish_topic}.")
            self.busy = False
            return
        if not self.action.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Trajectory action server unavailable.")
            self.busy = False
            if len(msg.points) == 2 and msg.joint_names == JOINT_NAMES:
                if not self.sent_startup_home:
                    self.sent_startup_home = False
                else:
                    self.sent_startup_inspection = False
            if len(msg.points) == 3 and msg.joint_names == JOINT_NAMES:
                self.sent_startup_wiggle = False
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = msg
        self.busy = True
        future = self.action.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.busy = False
            self.get_logger().warn("Trajectory goal rejected.")
            self._send_queued_trajectory_if_idle()
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self.busy = False
        result = future.result()
        if result is None:
            self.get_logger().warn("Trajectory execution returned no result.")
            return
        self.get_logger().info(f"Trajectory finished with error_code={result.result.error_code}")
        self.pending_target_name = None
        self.pending_target_positions = None
        self._send_queued_trajectory_if_idle()

    def _send_queued_trajectory_if_idle(self):
        if self.busy or self.queued_trajectory is None:
            return
        queued = self.queued_trajectory
        self.queued_trajectory = None
        self.sent_startup_wiggle = True
        self.get_logger().info("Sending queued trajectory.")
        self._send_goal(queued)

    def _check_pending_target(self):
        if self.current_positions is None or self.pending_target_positions is None:
            return
        tolerance = float(self.get_parameter("pose_reached_tolerance_rad").value)
        errors = [fabs(q - q_des) for q, q_des in zip(self.current_positions, self.pending_target_positions)]
        if all(err <= tolerance for err in errors):
            self.get_logger().info(
                f"Reached {self.pending_target_name} within {tolerance:.3f} rad. Current pose: {self._format_positions(self.current_positions)}."
            )
            self.pending_target_name = None
            self.pending_target_positions = None
            self.last_target_report_time = 0.0
            return
        report_interval = float(self.get_parameter("pose_report_interval_s").value)
        now = time.monotonic()
        if now - self.last_target_report_time < report_interval:
            return
        self.last_target_report_time = now
        max_idx = max(range(len(errors)), key=lambda idx: errors[idx])
        self.get_logger().info(
            f"{self.pending_target_name} not reached yet. Max error: {JOINT_NAMES[max_idx]}={errors[max_idx]:.3f} rad. Current pose: {self._format_positions(self.current_positions)}."
        )

    def _format_positions(self, positions: list[float]) -> str:
        pairs = ", ".join(f"{name}={value:.3f}" for name, value in zip(JOINT_NAMES, positions))
        return f"[{pairs}]"


def main():
    rclpy.init()
    node = TrajectoryExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
