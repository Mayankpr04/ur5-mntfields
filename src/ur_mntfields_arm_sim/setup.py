from setuptools import find_packages, setup


package_name = "ur_mntfields_arm_sim"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/ur_mntfields_arm_sim.launch.py",
                "launch/ur_mntfields_arm_gz.launch.py",
                "launch/ur_mntfields_arm_real.launch.py",
                "launch/ur_mntfields_arm_field_test_gz.launch.py",
                "launch/ur5_box_teleop_gz.launch.py",
                "launch/ur_mntfields_rgbd_gz.launch.py",
                "launch/ur_mntfields_rgbd_curobo_gz.launch.py",
            ],
        ),
        (
            f"share/{package_name}/config",
            [
                "config/sim_scene.yaml",
                "config/real_scene.yaml",
                "config/real_scene_boxes.yaml",
                "config/initial_positions.yaml",
                "config/gz_controllers.yaml",
                "config/startup_poses.yaml",
            ],
        ),
        (f"share/{package_name}/rviz", ["rviz/rviz_ur5_sim.rviz"]),
        (f"share/{package_name}/worlds", ["worlds/ur5_cabinet_world.sdf"]),
        (
            f"share/{package_name}/config/ur5",
            [
                "config/ur5/joint_limits.yaml",
                "config/ur5/default_kinematics.yaml",
                "config/ur5/physical_parameters.yaml",
                "config/ur5/visual_parameters.yaml",
            ],
        ),
        (f"share/{package_name}/urdf", ["urdf/ur_with_wrist_camera.urdf.xacro"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mayank",
    maintainer_email="mayank@example.com",
    description="Synthetic simulation helpers for UR arm MNTFields exploration.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "synthetic_depth_camera = ur_mntfields_arm_sim.synthetic_depth_camera:main",
            "trajectory_executor = ur_mntfields_arm_sim.trajectory_executor:main",
            "cabinet_marker_publisher = ur_mntfields_arm_sim.cabinet_marker_publisher:main",
            "scene_boxes_publisher = ur_mntfields_arm_sim.scene_boxes_publisher:main",
        ],
    },
)
