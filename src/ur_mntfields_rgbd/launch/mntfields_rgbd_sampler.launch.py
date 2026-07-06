from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir")
    base_frame = LaunchConfiguration("base_frame")
    camera_frame = LaunchConfiguration("camera_frame")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("output_dir", default_value="/home/mayank/ur_ws/src/mntfields_rgbd_output"),
            DeclareLaunchArgument("base_frame", default_value="camera_link"),
            DeclareLaunchArgument("camera_frame", default_value="camera_color_optical_frame"),
            DeclareLaunchArgument("color_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/aligned_depth_to_color/camera_info"),
            Node(
                package="ur_mntfields_rgbd",
                executable="rgbd_mntfields_sampler",
                name="rgbd_mntfields_sampler",
                output="screen",
                parameters=[
                    PathJoinSubstitution([FindPackageShare("ur_mntfields_rgbd"), "config", "sampler.yaml"]),
                    {
                        "output_dir": output_dir,
                        "base_frame": base_frame,
                        "camera_frame": camera_frame,
                        "color_topic": color_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,
                    },
                ],
            ),
        ]
    )
