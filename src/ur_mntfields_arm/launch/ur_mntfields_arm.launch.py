from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="ur_mntfields_arm",
                executable="arm_mntfields_explorer",
                name="arm_mntfields_explorer",
                output="screen",
                parameters=["/home/mayank/ur_ws/src/ur_mntfields_arm/config/arm_explorer.yaml"],
            )
        ]
    )
