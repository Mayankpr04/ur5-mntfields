from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_pkg = FindPackageShare("ur_mntfields_arm_sim")
    real_cfg = PathJoinSubstitution([sim_pkg, "config", "real_scene.yaml"])
    scene_boxes_cfg = PathJoinSubstitution([sim_pkg, "config", "real_scene_boxes.yaml"])
    rviz_cfg = PathJoinSubstitution([sim_pkg, "rviz", "rviz_ur5_sim.rviz"])

    launch_rviz = LaunchConfiguration("launch_rviz")
    enable_trajectory_publish = LaunchConfiguration("enable_trajectory_publish")
    start_trajectory_executor = LaunchConfiguration("start_trajectory_executor")
    start_scene_boxes_publisher = LaunchConfiguration("start_scene_boxes_publisher")
    clearance_backend = LaunchConfiguration("clearance_backend")
    scene_boxes = LaunchConfiguration("scene_boxes")
    scene_boxes_frame = LaunchConfiguration("scene_boxes_frame")

    publish_camera_tf = LaunchConfiguration("publish_camera_tf")
    camera_tf_parent = LaunchConfiguration("camera_tf_parent")
    camera_tf_child = LaunchConfiguration("camera_tf_child")
    camera_tf_x = LaunchConfiguration("camera_tf_x")
    camera_tf_y = LaunchConfiguration("camera_tf_y")
    camera_tf_z = LaunchConfiguration("camera_tf_z")
    camera_tf_yaw = LaunchConfiguration("camera_tf_yaw")
    camera_tf_pitch = LaunchConfiguration("camera_tf_pitch")
    camera_tf_roll = LaunchConfiguration("camera_tf_roll")

    # PLACEHOLDER: replace these static TF args after hand-eye calibration.
    # Keep them consistent with real_scene.yaml: arm_mntfields_explorer.camera_in_tool.
    camera_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tool_to_realsense_static_tf",
        output="screen",
        condition=IfCondition(publish_camera_tf),
        arguments=[
            camera_tf_x,
            camera_tf_y,
            camera_tf_z,
            camera_tf_yaw,
            camera_tf_pitch,
            camera_tf_roll,
            camera_tf_parent,
            camera_tf_child,
        ],
    )

    explorer = Node(
        package="ur_mntfields_arm",
        executable="arm_mntfields_explorer",
        name="arm_mntfields_explorer",
        output="screen",
        parameters=[
            real_cfg,
            {
                "use_sim_time": False,
                "clearance_backend": clearance_backend,
                "enable_trajectory_publish": enable_trajectory_publish,
                "scene_boxes": [scene_boxes],
                "scene_boxes_frame": scene_boxes_frame,
            },
        ],
    )

    scene_boxes_publisher = Node(
        package="ur_mntfields_arm_sim",
        executable="scene_boxes_publisher",
        name="scene_boxes_publisher",
        output="screen",
        condition=IfCondition(start_scene_boxes_publisher),
        parameters=[scene_boxes_cfg, {"use_sim_time": False}],
    )

    trajectory_executor = Node(
        package="ur_mntfields_arm_sim",
        executable="trajectory_executor",
        name="trajectory_executor",
        output="screen",
        condition=IfCondition(start_trajectory_executor),
        parameters=[real_cfg, {"use_sim_time": False}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_ur5_real",
        output="screen",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(launch_rviz),
        parameters=[{"use_sim_time": False}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument(
                "enable_trajectory_publish",
                default_value="false",
                description="Safety default is false. Set true only after TF/ROI/depth validation.",
            ),
            DeclareLaunchArgument(
                "start_trajectory_executor",
                default_value="false",
                description="Start bridge to FollowJointTrajectory action. Keep false for mapping/training-only runs.",
            ),
            DeclareLaunchArgument(
                "start_scene_boxes_publisher",
                default_value="true",
                description="Continuously publish saved scene boxes from real_scene_boxes.yaml on /scene_boxes.",
            ),
            DeclareLaunchArgument(
                "clearance_backend",
                default_value="sdf",
                description="Collision checker backend: original or sdf.",
            ),
            DeclareLaunchArgument(
                "scene_boxes",
                default_value="",
                description="Optional one-shot scene box string: x,y,z,sx,sy,sz;... Runtime updates use /scene_boxes.",
            ),
            DeclareLaunchArgument("scene_boxes_frame", default_value="base_link"),
            DeclareLaunchArgument(
                "publish_camera_tf",
                default_value="true",
                description="Publish placeholder camera static TF. Disable if your calibrated TF is already published.",
            ),
            DeclareLaunchArgument("camera_tf_parent", default_value="tool0"),
            DeclareLaunchArgument("camera_tf_child", default_value="camera_color_optical_frame"),
            DeclareLaunchArgument("camera_tf_x", default_value="0.10"),
            DeclareLaunchArgument("camera_tf_y", default_value="0.0"),
            DeclareLaunchArgument("camera_tf_z", default_value="0.0"),
            DeclareLaunchArgument("camera_tf_yaw", default_value="0.0"),
            DeclareLaunchArgument("camera_tf_pitch", default_value="0.0"),
            DeclareLaunchArgument("camera_tf_roll", default_value="0.0"),
            camera_static_tf,
            scene_boxes_publisher,
            explorer,
            trajectory_executor,
            rviz,
        ]
    )
