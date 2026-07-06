from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_pkg = FindPackageShare("ur_mntfields_arm_sim")
    scene_cfg = PathJoinSubstitution([sim_pkg, "config", "sim_scene.yaml"])
    startup_cfg = PathJoinSubstitution([sim_pkg, "config", "startup_poses.yaml"])
    world = PathJoinSubstitution([sim_pkg, "worlds", "ur5_cabinet_world.sdf"])
    rviz_cfg = PathJoinSubstitution([sim_pkg, "rviz", "rviz_ur5_sim.rviz"])
    xacro_file = PathJoinSubstitution([sim_pkg, "urdf", "ur_with_wrist_camera.urdf.xacro"])
    initial_positions = PathJoinSubstitution([sim_pkg, "config", "initial_positions.yaml"])
    controllers_cfg = PathJoinSubstitution([sim_pkg, "config", "gz_controllers.yaml"])
    sim_launch_rviz = LaunchConfiguration("sim_launch_rviz")

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
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "180",
        ],
        output="screen",
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "180",
        ],
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
                "enable_startup_home": False,
                "enable_startup_inspection": False,
                "enable_startup_wiggle": False,
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
            gz_sim,
            robot_state_publisher,
            spawn,
            bridge,
            TimerAction(period=5.0, actions=[joint_state_broadcaster_spawner]),
            TimerAction(period=7.0, actions=[joint_trajectory_controller_spawner]),
            cabinet_markers,
            executor,
            rviz,
        ]
    )
