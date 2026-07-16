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
    allow_uncertified_anchor_checkpoint = LaunchConfiguration(
        "allow_uncertified_anchor_checkpoint"
    )
    planner_type = LaunchConfiguration("planner_type")
    clearance_backend = LaunchConfiguration("clearance_backend")
    planner_direct_edge = LaunchConfiguration("planner_direct_edge")
    planner_shortcut = LaunchConfiguration("planner_shortcut")
    budgeted_anchor_count = LaunchConfiguration("budgeted_anchor_count")
    budgeted_anchor_sample_path = LaunchConfiguration("budgeted_anchor_sample_path")
    budgeted_anchor_force_first_goal = LaunchConfiguration("budgeted_anchor_force_first_goal")
    path_shortcut_max_passes = LaunchConfiguration("path_shortcut_max_passes")
    rrt_step_size_q = LaunchConfiguration("rrt_step_size_q")
    rrt_max_iters = LaunchConfiguration("rrt_max_iters")
    rrt_goal_bias = LaunchConfiguration("rrt_goal_bias")
    rrt_edge_check_step_rad = LaunchConfiguration("rrt_edge_check_step_rad")
    collision_cloud_path = LaunchConfiguration("collision_cloud_path")
    collision_aware_field_rollout = LaunchConfiguration("collision_aware_field_rollout")
    trajectory_collision_validation_enabled = LaunchConfiguration("trajectory_collision_validation_enabled")
    field_precheck_enabled = LaunchConfiguration("field_precheck_enabled")
    learned_speed_search_min_speed = LaunchConfiguration("learned_speed_search_min_speed")
    field_path_joint_edge_weight = LaunchConfiguration("field_path_joint_edge_weight")
    field_path_turn_weight = LaunchConfiguration("field_path_turn_weight")
    field_path_tool_edge_weight = LaunchConfiguration("field_path_tool_edge_weight")
    field_path_tool_goal_weight = LaunchConfiguration("field_path_tool_goal_weight")
    field_path_clearance_penalty_weight = LaunchConfiguration("field_path_clearance_penalty_weight")
    field_path_clearance_soft_margin_m = LaunchConfiguration("field_path_clearance_soft_margin_m")
    field_path_return_first_goal = LaunchConfiguration("field_path_return_first_goal")
    field_path_forward_probe_fraction = LaunchConfiguration("field_path_forward_probe_fraction")
    field_cartesian_candidate_count = LaunchConfiguration("field_cartesian_candidate_count")
    field_cartesian_candidate_step_m = LaunchConfiguration("field_cartesian_candidate_step_m")
    field_cartesian_candidate_damping = LaunchConfiguration("field_cartesian_candidate_damping")
    cartesian_graph_tool_edge_weight = LaunchConfiguration("cartesian_graph_tool_edge_weight")
    cartesian_graph_tool_goal_weight = LaunchConfiguration("cartesian_graph_tool_goal_weight")
    cartesian_graph_joint_edge_weight = LaunchConfiguration("cartesian_graph_joint_edge_weight")
    cartesian_graph_joint_goal_weight = LaunchConfiguration("cartesian_graph_joint_goal_weight")
    cartesian_graph_tau_weight = LaunchConfiguration("cartesian_graph_tau_weight")
    cartesian_graph_clearance_penalty_weight = LaunchConfiguration("cartesian_graph_clearance_penalty_weight")
    cartesian_graph_clearance_soft_margin_m = LaunchConfiguration("cartesian_graph_clearance_soft_margin_m")
    cartesian_shortcut_enabled = LaunchConfiguration("cartesian_shortcut_enabled")
    cartesian_shortcut_tool_weight = LaunchConfiguration("cartesian_shortcut_tool_weight")
    cartesian_shortcut_joint_weight = LaunchConfiguration("cartesian_shortcut_joint_weight")
    cartesian_shortcut_smoothness_weight = LaunchConfiguration("cartesian_shortcut_smoothness_weight")
    cartesian_shortcut_min_improvement = LaunchConfiguration("cartesian_shortcut_min_improvement")
    cartesian_shortcut_max_skip = LaunchConfiguration("cartesian_shortcut_max_skip")
    cartesian_shortcut_try_reverse = LaunchConfiguration("cartesian_shortcut_try_reverse")
    sampled_rollout_samples = LaunchConfiguration("sampled_rollout_samples")
    sampled_rollout_horizon = LaunchConfiguration("sampled_rollout_horizon")
    sampled_rollout_iterations = LaunchConfiguration("sampled_rollout_iterations")
    sampled_rollout_noise_scale = LaunchConfiguration("sampled_rollout_noise_scale")
    sampled_rollout_temperature = LaunchConfiguration("sampled_rollout_temperature")
    sampled_rollout_tau_weight = LaunchConfiguration("sampled_rollout_tau_weight")
    sampled_rollout_goal_dist_weight = LaunchConfiguration("sampled_rollout_goal_dist_weight")
    sampled_rollout_joint_path_weight = LaunchConfiguration("sampled_rollout_joint_path_weight")
    sampled_rollout_tool_path_weight = LaunchConfiguration("sampled_rollout_tool_path_weight")
    sampled_rollout_clearance_penalty_weight = LaunchConfiguration("sampled_rollout_clearance_penalty_weight")
    sampled_rollout_topk_edge_checks = LaunchConfiguration("sampled_rollout_topk_edge_checks")
    interactive_goal_enabled = LaunchConfiguration("interactive_goal_enabled")
    interactive_goal_mode = LaunchConfiguration("interactive_goal_mode")
    fixed_goal_sequence_enabled = LaunchConfiguration("fixed_goal_sequence_enabled")
    fixed_goal_mode = LaunchConfiguration("fixed_goal_mode")
    fixed_goal_return_to_first = LaunchConfiguration("fixed_goal_return_to_first")
    fixed_goal_anchor_routing_enabled = LaunchConfiguration("fixed_goal_anchor_routing_enabled")
    fixed_goal_joint_positions_csv = LaunchConfiguration("fixed_goal_joint_positions_csv")
    fixed_goal_point_x = LaunchConfiguration("fixed_goal_point_x")
    fixed_goal_point_y = LaunchConfiguration("fixed_goal_point_y")
    fixed_goal_point_z = LaunchConfiguration("fixed_goal_point_z")
    max_goal_candidates = LaunchConfiguration("max_goal_candidates")
    goal_candidate_dedupe_rad = LaunchConfiguration("goal_candidate_dedupe_rad")
    goal_candidate_sweep_enabled = LaunchConfiguration("goal_candidate_sweep_enabled")
    goal_candidate_sweep_execute_best = LaunchConfiguration("goal_candidate_sweep_execute_best")
    goal_candidate_sweep_save_path = LaunchConfiguration("goal_candidate_sweep_save_path")
    goal_candidate_execute_indices_csv = LaunchConfiguration("goal_candidate_execute_indices_csv")
    goal_candidate_execute_return_startup = LaunchConfiguration("goal_candidate_execute_return_startup")

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
                "planner_type": planner_type,
                "allow_uncertified_anchor_checkpoint": allow_uncertified_anchor_checkpoint,
                "clearance_backend": clearance_backend,
                "planner_direct_edge": planner_direct_edge,
                "planner_shortcut": planner_shortcut,
                "budgeted_anchor_count": budgeted_anchor_count,
                "budgeted_anchor_sample_path": budgeted_anchor_sample_path,
                "budgeted_anchor_force_first_goal": budgeted_anchor_force_first_goal,
                "path_shortcut_max_passes": path_shortcut_max_passes,
                "rrt_step_size_q": rrt_step_size_q,
                "rrt_max_iters": rrt_max_iters,
                "rrt_goal_bias": rrt_goal_bias,
                "rrt_edge_check_step_rad": rrt_edge_check_step_rad,
                "collision_cloud_path": collision_cloud_path,
                "collision_aware_field_rollout": collision_aware_field_rollout,
                "trajectory_collision_validation_enabled": trajectory_collision_validation_enabled,
                "learned_speed_search_min_speed": learned_speed_search_min_speed,
                "field_path_joint_edge_weight": field_path_joint_edge_weight,
                "field_path_turn_weight": field_path_turn_weight,
                "field_path_tool_edge_weight": field_path_tool_edge_weight,
                "field_path_tool_goal_weight": field_path_tool_goal_weight,
                "field_path_clearance_penalty_weight": field_path_clearance_penalty_weight,
                "field_path_clearance_soft_margin_m": field_path_clearance_soft_margin_m,
                "field_path_return_first_goal": field_path_return_first_goal,
                "field_path_forward_probe_fraction": field_path_forward_probe_fraction,
                "field_cartesian_candidate_count": field_cartesian_candidate_count,
                "field_cartesian_candidate_step_m": field_cartesian_candidate_step_m,
                "field_cartesian_candidate_damping": field_cartesian_candidate_damping,
                "cartesian_graph_tool_edge_weight": cartesian_graph_tool_edge_weight,
                "cartesian_graph_tool_goal_weight": cartesian_graph_tool_goal_weight,
                "cartesian_graph_joint_edge_weight": cartesian_graph_joint_edge_weight,
                "cartesian_graph_joint_goal_weight": cartesian_graph_joint_goal_weight,
                "cartesian_graph_tau_weight": cartesian_graph_tau_weight,
                "cartesian_graph_clearance_penalty_weight": cartesian_graph_clearance_penalty_weight,
                "cartesian_graph_clearance_soft_margin_m": cartesian_graph_clearance_soft_margin_m,
                "cartesian_shortcut_enabled": cartesian_shortcut_enabled,
                "cartesian_shortcut_tool_weight": cartesian_shortcut_tool_weight,
                "cartesian_shortcut_joint_weight": cartesian_shortcut_joint_weight,
                "cartesian_shortcut_smoothness_weight": cartesian_shortcut_smoothness_weight,
                "cartesian_shortcut_min_improvement": cartesian_shortcut_min_improvement,
                "cartesian_shortcut_max_skip": cartesian_shortcut_max_skip,
                "cartesian_shortcut_try_reverse": cartesian_shortcut_try_reverse,
                "sampled_rollout_samples": sampled_rollout_samples,
                "sampled_rollout_horizon": sampled_rollout_horizon,
                "sampled_rollout_iterations": sampled_rollout_iterations,
                "sampled_rollout_noise_scale": sampled_rollout_noise_scale,
                "sampled_rollout_temperature": sampled_rollout_temperature,
                "sampled_rollout_tau_weight": sampled_rollout_tau_weight,
                "sampled_rollout_goal_dist_weight": sampled_rollout_goal_dist_weight,
                "sampled_rollout_joint_path_weight": sampled_rollout_joint_path_weight,
                "sampled_rollout_tool_path_weight": sampled_rollout_tool_path_weight,
                "sampled_rollout_clearance_penalty_weight": sampled_rollout_clearance_penalty_weight,
                "sampled_rollout_topk_edge_checks": sampled_rollout_topk_edge_checks,
                "planned_path_topic": "/ur_mntfields_arm/planned_path",
                "trajectory_topic": "/ur_mntfields_arm/joint_trajectory",
                "goal_marker_topic": "/ur_mntfields_arm/test_goal_markers",
                "startup_positions": [0.0, -2.74850, 1.50004, -1.71994, -1.57000, 0.03334],
                "startup_pose_tolerance_rad": 0.04,
                "startup_pose_settle_s": 1.0,
                "rollout_max_steps": 120,
                "interactive_goal_enabled": interactive_goal_enabled,
                "interactive_goal_mode": interactive_goal_mode,
                "fixed_goal_sequence_enabled": fixed_goal_sequence_enabled,
                "fixed_goal_mode": fixed_goal_mode,
                "fixed_goal_return_to_first": fixed_goal_return_to_first,
                "fixed_goal_anchor_routing_enabled": fixed_goal_anchor_routing_enabled,
                "fixed_goal_joint_positions_csv": fixed_goal_joint_positions_csv,
                # Camera-facing transition pose at world xyz ~= [0.346, 0.350, 0.800],
                # centered on the cabinet and 0.35 m in front of its opening.
                "fixed_goal_anchor_joint_position": [-0.0834, -2.0102, 2.7639, -3.8953, -1.4873, 0.0],
                "fixed_goal_anchor_route_indices": [1, 0, 2, 3, 0, 1],
                "fixed_goal_joint_positions": [
                    0.0, -1.20804, 1.09161, -2.81853, -1.57000, 0.03334,
                    0.0, -0.83967, 1.61948, -4.01894, -1.57000, 0.03334,
                    -0.77152, -0.68967, 1.43704, -3.81446, -1.57000, 0.03334,
                ],
                "fixed_goal_points_xyz_csv": [fixed_goal_point_x, ",", fixed_goal_point_y, ",", fixed_goal_point_z],
                "fixed_goal_reached_tolerance_rad": 0.05,
                "goal_candidate_clearance_min_m": 0.03,
                "goal_tool_forward_alignment_min": 0.70,
                "max_goal_candidates": max_goal_candidates,
                "goal_candidate_dedupe_rad": goal_candidate_dedupe_rad,
                "goal_candidate_sweep_enabled": goal_candidate_sweep_enabled,
                "goal_candidate_sweep_execute_best": goal_candidate_sweep_execute_best,
                "goal_candidate_sweep_save_path": goal_candidate_sweep_save_path,
                "goal_candidate_execute_indices_csv": goal_candidate_execute_indices_csv,
                "goal_candidate_execute_return_startup": goal_candidate_execute_return_startup,
                "field_precheck_enabled": field_precheck_enabled,
                "field_precheck_min_speed": 0.02,
                "field_precheck_neighborhood_samples": 8,
                "field_precheck_neighborhood_radius_norm": 0.015,
                "field_local_rollout_candidates": 32,
                "direct_joint_fallback_enabled": False,
                "trajectory_max_joint_speed": 0.10,
                "trajectory_max_joint_acceleration": 0.25,
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
                default_value="/home/mayank/ur_ws/src/ur5_sim_training_factorized_v2/model/weights_final.pt",
            ),
            DeclareLaunchArgument("planner_mode", default_value="bidirectional"),
            DeclareLaunchArgument(
                "allow_uncertified_anchor_checkpoint",
                default_value="false",
                description="Diagnostic-only override allowing field_anchor to load weights_partial.pt.",
            ),
            DeclareLaunchArgument(
                "planner_type",
                default_value="field_search",
                description="Planner implementation: field_anchor, field, field_search, learned_speed_search, or rrt_connect.",
            ),
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
                "budgeted_anchor_count",
                default_value="1",
                description="Hard anchor budget for field_anchor; valid values are one or two.",
            ),
            DeclareLaunchArgument(
                "budgeted_anchor_sample_path",
                default_value="",
                description="Optional training step NPZ used to derive environment-specific free configuration probes.",
            ),
            DeclareLaunchArgument(
                "budgeted_anchor_force_first_goal",
                default_value="false",
                description="Route the first fixed goal through the selected opening anchor.",
            ),
            DeclareLaunchArgument(
                "path_shortcut_max_passes",
                default_value="0",
                description="Dense collision-checked post-processing passes; time is reported in validate_ms.",
            ),
            DeclareLaunchArgument(
                "rrt_step_size_q",
                default_value="0.20",
                description="RRTConnect extension length in physical joint-space radians.",
            ),
            DeclareLaunchArgument("rrt_max_iters", default_value="4000"),
            DeclareLaunchArgument("rrt_goal_bias", default_value="0.20"),
            DeclareLaunchArgument(
                "rrt_edge_check_step_rad",
                default_value="0.04",
                description="Dense physical-joint interpolation resolution for every RRT edge check.",
            ),
            DeclareLaunchArgument(
                "collision_cloud_path",
                default_value="",
                description="Optional .npz (occupied_points) or .npy cloud used by geometric planners and validation.",
            ),
            DeclareLaunchArgument(
                "collision_aware_field_rollout",
                default_value="true",
                description="Use collision-aware planner edge checks. Set false for field-only timing tests.",
            ),
            DeclareLaunchArgument(
                "trajectory_collision_validation_enabled",
                default_value="true",
                description="Validate planned trajectory collision before execution. Set false only for field-only timing tests.",
            ),
            DeclareLaunchArgument(
                "field_precheck_enabled",
                default_value="true",
                description="Reject endpoints with a failed learned state/coverage oracle before planning.",
            ),
            DeclareLaunchArgument(
                "learned_speed_search_min_speed",
                default_value="0.20",
                description="Minimum neural predicted speed accepted by network-only field_search edges.",
            ),
            DeclareLaunchArgument(
                "clearance_backend",
                default_value="original",
                description="Clearance backend for test planning: original or sdf",
            ),
            DeclareLaunchArgument(
                "field_path_joint_edge_weight",
                default_value="0.15",
                description="Accumulated joint-edge path cost weight for deterministic field_search.",
            ),
            DeclareLaunchArgument(
                "field_path_turn_weight",
                default_value="0.25",
                description="Penalty for sharp changes in joint-space search direction.",
            ),
            DeclareLaunchArgument(
                "field_path_tool_edge_weight",
                default_value="0.0",
                description="Accumulated Cartesian tool-edge path cost weight for deterministic field_search.",
            ),
            DeclareLaunchArgument(
                "field_path_tool_goal_weight",
                default_value="-1.0",
                description="Cartesian tool distance-to-goal heuristic weight. Negative preserves legacy behavior by matching field_path_tool_edge_weight.",
            ),
            DeclareLaunchArgument(
                "field_path_clearance_penalty_weight",
                default_value="20.0",
                description="Accumulated soft clearance penalty weight for deterministic field_search.",
            ),
            DeclareLaunchArgument(
                "field_path_clearance_soft_margin_m",
                default_value="0.04",
                description="Extra clearance band used by the accumulated soft clearance penalty.",
            ),
            DeclareLaunchArgument(
                "field_path_return_first_goal",
                default_value="true",
                description="Return the first valid goal edge found under path-cost ordering instead of proving a cheaper queued goal.",
            ),
            DeclareLaunchArgument(
                "field_path_forward_probe_fraction",
                default_value="0.15",
                description="Fraction of rollout_max_steps spent on the forward probe before reverse search in path-cost bidirectional mode.",
            ),
            DeclareLaunchArgument(
                "field_cartesian_candidate_count",
                default_value="0",
                description="Extra FK-Jacobian tool-progress candidates per field_search expansion. Zero preserves legacy behavior.",
            ),
            DeclareLaunchArgument(
                "field_cartesian_candidate_step_m",
                default_value="0.05",
                description="Desired Cartesian tool step length for FK-Jacobian candidate generation.",
            ),
            DeclareLaunchArgument(
                "field_cartesian_candidate_damping",
                default_value="0.001",
                description="Damping for FK-Jacobian least-squares Cartesian candidate generation.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_tool_edge_weight",
                default_value="1.0",
                description="Cartesian graph planner accumulated tool path length weight.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_tool_goal_weight",
                default_value="2.0",
                description="Cartesian graph planner tool distance-to-goal heuristic weight.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_joint_edge_weight",
                default_value="0.05",
                description="Cartesian graph planner accumulated normalized joint path weight.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_joint_goal_weight",
                default_value="0.25",
                description="Cartesian graph planner normalized joint distance-to-goal heuristic weight.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_tau_weight",
                default_value="0.0",
                description="Optional learned travel-time heuristic weight for cartesian_graph.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_clearance_penalty_weight",
                default_value="0.0",
                description="Optional soft clearance penalty for cartesian_graph edge costs.",
            ),
            DeclareLaunchArgument(
                "cartesian_graph_clearance_soft_margin_m",
                default_value="0.04",
                description="Extra clearance band used by cartesian_graph soft clearance penalty.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_enabled",
                default_value="false",
                description="Enable opt-in Cartesian-aware valid-path shortcut selection after field_search.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_tool_weight",
                default_value="1.0",
                description="Tool Cartesian path length weight for the opt-in post-search shortcut selector.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_joint_weight",
                default_value="0.10",
                description="Normalized joint path length weight for the opt-in post-search shortcut selector.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_smoothness_weight",
                default_value="0.0",
                description="Final path smoothness weight for accepting the opt-in Cartesian shortcut candidate.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_min_improvement",
                default_value="0.01",
                description="Minimum relative Cartesian path-cost improvement required before replacing the baseline shortcut.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_max_skip",
                default_value="48",
                description="Maximum waypoint lookahead for each batched Cartesian shortcut step.",
            ),
            DeclareLaunchArgument(
                "cartesian_shortcut_try_reverse",
                default_value="false",
                description="When Cartesian shortcutting is enabled, also try reverse field_search and choose the lower Cartesian-cost valid path.",
            ),
            DeclareLaunchArgument("sampled_rollout_samples", default_value="256"),
            DeclareLaunchArgument("sampled_rollout_horizon", default_value="8"),
            DeclareLaunchArgument("sampled_rollout_iterations", default_value="2"),
            DeclareLaunchArgument("sampled_rollout_noise_scale", default_value="1.0"),
            DeclareLaunchArgument("sampled_rollout_temperature", default_value="35.0"),
            DeclareLaunchArgument("sampled_rollout_tau_weight", default_value="0.55"),
            DeclareLaunchArgument("sampled_rollout_goal_dist_weight", default_value="3.0"),
            DeclareLaunchArgument("sampled_rollout_joint_path_weight", default_value="0.20"),
            DeclareLaunchArgument("sampled_rollout_tool_path_weight", default_value="1.0"),
            DeclareLaunchArgument("sampled_rollout_clearance_penalty_weight", default_value="8.0"),
            DeclareLaunchArgument("sampled_rollout_topk_edge_checks", default_value="64"),
            DeclareLaunchArgument("interactive_goal_enabled", default_value="false"),
            DeclareLaunchArgument(
                "interactive_goal_mode",
                default_value="tool_point",
                description="Goal candidate mode for clicked/fixed point goals: tool_point or camera view mode.",
            ),
            DeclareLaunchArgument("fixed_goal_sequence_enabled", default_value="true"),
            DeclareLaunchArgument(
                "fixed_goal_mode",
                default_value="joint",
                description="Fixed goal mode: joint, point, or point_candidate_sequence.",
            ),
            DeclareLaunchArgument("fixed_goal_return_to_first", default_value="true"),
            DeclareLaunchArgument(
                "fixed_goal_anchor_routing_enabled",
                default_value="false",
                description="Route inter-shelf fixed-goal legs through a centered free-space cabinet anchor.",
            ),
            DeclareLaunchArgument(
                "fixed_goal_joint_positions_csv",
                default_value="",
                description="Optional comma-separated joint goals; six values per goal.",
            ),
            DeclareLaunchArgument("fixed_goal_point_x", default_value="0.60"),
            DeclareLaunchArgument("fixed_goal_point_y", default_value="0.35"),
            DeclareLaunchArgument("fixed_goal_point_z", default_value="0.68"),
            DeclareLaunchArgument("max_goal_candidates", default_value="10"),
            DeclareLaunchArgument("goal_candidate_dedupe_rad", default_value="0.035"),
            DeclareLaunchArgument(
                "goal_candidate_sweep_enabled",
                default_value="false",
                description="Plan/log all generated IK candidates for a point instead of stopping at the first valid path.",
            ),
            DeclareLaunchArgument(
                "goal_candidate_sweep_execute_best",
                default_value="false",
                description="When sweeping candidates, execute the best valid candidate after logging the full sweep.",
            ),
            DeclareLaunchArgument(
                "goal_candidate_sweep_save_path",
                default_value="",
                description="Optional .npz file or directory for fixed/clicked point candidate sweep metrics.",
            ),
            DeclareLaunchArgument(
                "goal_candidate_execute_indices_csv",
                default_value="16,22,27",
                description="1-based tool-point candidate indices to execute when fixed_goal_mode=point_candidate_sequence.",
            ),
            DeclareLaunchArgument(
                "goal_candidate_execute_return_startup",
                default_value="true",
                description="Return to startup between selected same-point candidate executions.",
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
