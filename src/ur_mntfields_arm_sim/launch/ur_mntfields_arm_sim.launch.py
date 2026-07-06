from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("ur_mntfields_arm_sim"), "config", "sim_scene.yaml"])
    startup_cfg = PathJoinSubstitution([FindPackageShare("ur_mntfields_arm_sim"), "config", "startup_poses.yaml"])
    rviz_cfg = PathJoinSubstitution([FindPackageShare("ur_mntfields_arm_sim"), "rviz", "rviz_ur5_sim.rviz"])
    sim_launch_rviz = LaunchConfiguration("sim_launch_rviz")
    output_dir = LaunchConfiguration("output_dir")
    clearance_backend = LaunchConfiguration("clearance_backend")
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_robot_driver"), "/launch/ur_control.launch.py"]
        ),
        launch_arguments={
            "ur_type": "ur5",
            "robot_ip": "0.0.0.0",
            "use_fake_hardware": "true",
            "fake_sensor_commands": "true",
            "launch_rviz": "false",
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
            "description_package": "ur_mntfields_arm_sim",
            "description_file": "ur_with_wrist_camera.urdf.xacro",
        }.items(),
    )

    depth_sim = Node(
        package="ur_mntfields_arm_sim",
        executable="synthetic_depth_camera",
        name="synthetic_depth_camera",
        output="screen",
        parameters=[cfg],
    )

    explorer = Node(
        package="ur_mntfields_arm",
        executable="arm_mntfields_explorer",
        name="arm_mntfields_explorer",
        output="screen",
        parameters=[cfg, {"output_dir": output_dir, "clearance_backend": clearance_backend}],
    )

    executor = Node(
        package="ur_mntfields_arm_sim",
        executable="trajectory_executor",
        name="trajectory_executor",
        output="screen",
        parameters=[startup_cfg],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_ur5_sim",
        output="screen",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(sim_launch_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_launch_rviz", default_value="true"),
            DeclareLaunchArgument("output_dir", default_value="/home/mayank/ur_ws/src/ur5_sim_training_urdf_fk"),
            DeclareLaunchArgument(
                "clearance_backend",
                default_value="original",
                description="Clearance backend for sampling/planning: original or sdf",
            ),
            ur_control,
            depth_sim,
            explorer,
            executor,
            rviz,
        ]
    )
