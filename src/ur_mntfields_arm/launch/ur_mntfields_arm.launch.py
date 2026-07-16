from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="ur_mntfields_arm",
                executable="arm_mntfields_explorer",
                name="arm_mntfields_explorer",
                on_exit=Shutdown(reason="online field training process exited"),
                output="screen",
                parameters=["/home/mayank/ur_ws/src/ur_mntfields_arm/config/arm_explorer.yaml"],
            )
        ]
    )
