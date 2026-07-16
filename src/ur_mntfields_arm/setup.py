from setuptools import find_packages, setup


package_name = "ur_mntfields_arm"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/ur_mntfields_arm.launch.py"]),
        (f"share/{package_name}/config", ["config/arm_explorer.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mayank",
    maintainer_email="mayank@example.com",
    description="UR5 arm exploration and MNTFields training in 6D configuration space.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "arm_mntfields_explorer = ur_mntfields_arm.exploration_manager:main",
            "inspect_training_dump = ur_mntfields_arm.inspect_training_dump:main",
            "offline_replay_train = ur_mntfields_arm.offline_replay_train:main",
            "test_trained_field = ur_mntfields_arm.test_trained_field:main",
            "visualize_field_slices = ur_mntfields_arm.visualize_field_slices:main",
            "joint_teleop = ur_mntfields_arm.joint_teleop:main",
            "sphere_urdf_overlay = ur_mntfields_arm.sphere_urdf_overlay:main",
            "check_sphere_motion = ur_mntfields_arm.check_sphere_motion:main",
            "view_arm_samples_3d = ur_mntfields_arm.view_arm_samples_3d:main",
            "view_field_speed_3d = ur_mntfields_arm.view_field_speed_3d:main",
            "view_field_trajectories_3d = ur_mntfields_arm.view_field_trajectories_3d:main",
            "view_planner_score_sweep_3d = ur_mntfields_arm.view_planner_score_sweep_3d:main",
            "view_goal_candidate_sweep_3d = ur_mntfields_arm.view_goal_candidate_sweep_3d:main",
            "view_offline_planner_benchmark_3d = ur_mntfields_arm.view_offline_planner_benchmark_3d:main",
            "offline_planner_benchmark = ur_mntfields_arm.offline_planner_benchmark:main",
        ],
    },
)
