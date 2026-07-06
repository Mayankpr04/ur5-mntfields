from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_pkg = FindPackageShare("ur_mntfields_arm_sim")
    arm_pkg = FindPackageShare("ur_mntfields_arm")
    scene_cfg = PathJoinSubstitution([sim_pkg, "config", "sim_scene.yaml"])
    startup_cfg = PathJoinSubstitution([sim_pkg, "config", "startup_poses.yaml"])
    world = PathJoinSubstitution([sim_pkg, "worlds", "ur5_cabinet_world.sdf"])
    rviz_cfg = PathJoinSubstitution([sim_pkg, "rviz", "rviz_ur5_sim.rviz"])
    xacro_file = PathJoinSubstitution([sim_pkg, "urdf", "ur_with_wrist_camera.urdf.xacro"])
    initial_positions = PathJoinSubstitution([sim_pkg, "config", "initial_positions.yaml"])
    controllers_cfg = PathJoinSubstitution([sim_pkg, "config", "gz_controllers.yaml"])

    sim_launch_rviz = LaunchConfiguration("sim_launch_rviz")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    planner_mode = LaunchConfiguration("planner_mode")
    clearance_backend = LaunchConfiguration("clearance_backend")
    planner_direct_edge = LaunchConfiguration("planner_direct_edge")
    planner_shortcut = LaunchConfiguration("planner_shortcut")

    robot_description = Command(
        [
            "xacro ",
            xacro_file,
            " ur_type:=ur5",
            " base_x:=0.15",
            " base_y:=0.35",
            " base_z:=0.50",
            " sim_ignition:=true",
            " use_fake_hardware:=false",
            " fake_sensor_commands:=false",
            " simulation_controllers:=",
            controllers_cfg,
            " initial_positions_file:=",
            initial_positions,
        ]
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": ["-r ", world], "on_exit_shutdown": "true"}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "ur5_mntfields", "-topic", "/robot_description"],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/camera/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            ("/camera/camera/image", "/camera/camera/color/image_raw"),
            ("/camera/camera/camera_info", "/camera/camera/aligned_depth_to_color/camera_info"),
            ("/camera/camera/depth_image", "/camera/camera/aligned_depth_to_color/image_raw"),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "180"],
        output="screen",
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "180"],
        output="screen",
    )

    cabinet_markers = Node(
        package="ur_mntfields_arm_sim",
        executable="cabinet_marker_publisher",
        name="cabinet_marker_publisher",
        output="screen",
        parameters=[scene_cfg, {"use_sim_time": True}],
    )

    executor = Node(
        package="ur_mntfields_arm_sim",
        executable="trajectory_executor",
        name="trajectory_executor",
        output="screen",
        parameters=[
            startup_cfg,
            {
                "direct_publish_topic": "",
                "trajectory_action_name": "/joint_trajectory_controller/follow_joint_trajectory",
                "enable_startup_wiggle": False,
                "pose_reached_tolerance_rad": 0.04,
                "use_sim_time": True,
            }
        ],
    )

    field_test = Node(
        package="ur_mntfields_arm",
        executable="test_trained_field",
        name="test_trained_field",
        output="screen",
        parameters=[
            scene_cfg,
            startup_cfg,
            {
                "checkpoint_path": checkpoint_path,
                "planner_mode": planner_mode,
                "clearance_backend": clearance_backend,
                "planner_direct_edge": planner_direct_edge,
                "planner_shortcut": planner_shortcut,
                "planned_path_topic": "/ur_mntfields_arm/planned_path",
                "trajectory_topic": "/ur_mntfields_arm/joint_trajectory",
                "goal_marker_topic": "/ur_mntfields_arm/test_goal_markers",
                "startup_positions": [0.0, -2.74850, 1.50004, -1.71994, -1.57000, 0.03334],
                "startup_pose_tolerance_rad": 0.04,
                "startup_pose_settle_s": 1.0,
                "rollout_max_steps": 120,
                "interactive_goal_enabled": False,
                "fixed_goal_sequence_enabled": True,
                "fixed_goal_mode": "joint",
                "fixed_goal_return_to_first": True,
                "fixed_goal_joint_positions": [
                    0.0, -1.20804, 1.09161, -2.81853, -1.57000, 0.03334,
                    0.0, -0.83967, 1.61948, -4.01894, -1.57000, 0.03334,
                    -0.77152, -0.68967, 1.43704, -3.81446, -1.57000, 0.03334,
                ],
                "fixed_goal_reached_tolerance_rad": 0.05,
                "goal_candidate_clearance_min_m": 0.03,
                "goal_tool_forward_alignment_min": 0.70,
                "max_goal_candidates": 10,
                "goal_candidate_dedupe_rad": 0.035,
                "field_precheck_enabled": True,
                "field_precheck_min_speed": 0.02,
                "field_precheck_neighborhood_samples": 8,
                "field_precheck_neighborhood_radius_norm": 0.015,
                "field_local_rollout_candidates": 32,
                "direct_joint_fallback_enabled": False,
                "trajectory_max_joint_speed": 0.10,
                "trajectory_min_segment_dt": 0.65,
                "trajectory_waypoint_stride": 1,
                "trajectory_smoothing_window": 1,
                "use_sim_time": True,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_ur5_sim",
        output="screen",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(sim_launch_rviz),
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_launch_rviz", default_value="true"),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="/home/mayank/ur_ws/src/ur5_sim_training_urdf_fk/model/weights_final.pt",
            ),
            DeclareLaunchArgument("planner_mode", default_value="bidirectional"),
            DeclareLaunchArgument(
                "planner_direct_edge",
                default_value="true",
                description="Allow direct start-goal edge before field rollout in fixed-goal field test.",
            ),
            DeclareLaunchArgument(
                "planner_shortcut",
                default_value="true",
                description="Allow post-search shortcutting inside the field planner.",
            ),
            DeclareLaunchArgument(
                "clearance_backend",
                default_value="original",
                description="Clearance backend for test planning: original or sdf",
            ),
            gz_sim,
            robot_state_publisher,
            spawn,
            bridge,
            TimerAction(period=5.0, actions=[joint_state_broadcaster_spawner]),
            TimerAction(period=7.0, actions=[joint_trajectory_controller_spawner]),
            cabinet_markers,
            executor,
            field_test,
            rviz,
        ]
    )
