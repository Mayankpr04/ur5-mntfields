from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ur_mntfields_arm.ur5_kinematics import JOINT_NAMES, UR5Kinematics


def _rotation_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


class JointTeleop(Node):
    def __init__(self, step_rad: float, duration_s: float, trajectory_topic: str, ur_type: str):
        super().__init__("ur5_joint_teleop")
        self.step_rad = float(step_rad)
        self.duration_s = float(duration_s)
        self.kinematics = UR5Kinematics(ur_type=ur_type)
        self.q = 0.5 * (self.kinematics.joint_min + self.kinematics.joint_max)
        self.selected_joint_idx = 0
        self.have_joint_state = False
        self.pub = self.create_publisher(JointTrajectory, trajectory_topic, 5)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)
        self.get_logger().info(f"Publishing teleop trajectories to {trajectory_topic}")
        self._print_help()

    def _joint_state_cb(self, msg: JointState):
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        if all(name in name_to_pos for name in JOINT_NAMES):
            self.q = np.asarray([name_to_pos[name] for name in JOINT_NAMES], dtype=np.float64)
            self.have_joint_state = True

    def _print_help(self):
        print(
            "\nUR5 joint teleop keys:\n"
            "  1-6 select joint\n"
            "  w increase selected joint, s decrease selected joint\n"
            "  p print FK pose, h help, x exit\n",
            flush=True,
        )

    def _publish(self):
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = self.q.astype(float).tolist()
        pt.time_from_start.sec = int(self.duration_s)
        pt.time_from_start.nanosec = int((self.duration_s - int(self.duration_s)) * 1.0e9)
        msg.points.append(pt)
        self.pub.publish(msg)

    def _joint_state_text(self) -> str:
        pairs = []
        for idx, (name, value) in enumerate(zip(JOINT_NAMES, self.q.tolist())):
            marker = "*" if idx == self.selected_joint_idx else " "
            pairs.append(f"{marker}{idx + 1}:{name}={value:+.5f}rad/{math.degrees(value):+.2f}deg")
        return "  ".join(pairs)

    def _print_state_and_pose(self):
        pose = self.kinematics.fk(self.q)
        roll, pitch, yaw = _rotation_to_rpy(pose[:3, :3])
        print("joint_states: " + self._joint_state_text(), flush=True)
        print(
            "tool0_xyz_rpy="
            + " ".join(
                f"{v:.5f}"
                for v in [
                    pose[0, 3],
                    pose[1, 3],
                    pose[2, 3],
                    roll,
                    pitch,
                    yaw,
                ]
            ),
            flush=True,
        )

    def handle_key(self, key: str) -> bool:
        if key == "x":
            return False
        if key == "h":
            self._print_help()
            return True
        if key == "p":
            self._print_state_and_pose()
            return True
        if key in {"1", "2", "3", "4", "5", "6"}:
            self.selected_joint_idx = int(key) - 1
            print(f"selected joint {key}: {JOINT_NAMES[self.selected_joint_idx]}", flush=True)
            self._print_state_and_pose()
            return True
        if key not in {"w", "s"}:
            return True
        sign = 1.0 if key == "w" else -1.0
        self.q[self.selected_joint_idx] += sign * self.step_rad
        self.q = self.kinematics.clamp(self.q)
        self._publish()
        self._print_state_and_pose()
        return True


def _read_key(timeout_s: float = 0.05) -> str | None:
    ready, _w, _e = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    return sys.stdin.read(1)


def main():
    parser = argparse.ArgumentParser(description="Keyboard teleop for UR5 joints with FK pose printout.")
    parser.add_argument("--step-rad", type=float, default=0.05, help="Joint increment per keypress in radians.")
    parser.add_argument("--duration-s", type=float, default=0.6, help="Trajectory duration per keypress.")
    parser.add_argument(
        "--trajectory-topic",
        default="/ur_mntfields_arm/joint_trajectory",
        help="JointTrajectory topic consumed by trajectory_executor.",
    )
    parser.add_argument("--ur-type", default="ur5", help="UR description type.")
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = JointTeleop(args.step_rad, args.duration_s, args.trajectory_topic, args.ur_type)
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.02)
            key = _read_key(0.02)
            if key is not None:
                running = node.handle_key(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
