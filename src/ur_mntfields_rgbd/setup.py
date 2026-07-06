from setuptools import find_packages, setup


package_name = "ur_mntfields_rgbd"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/mntfields_rgbd_sampler.launch.py"]),
        (f"share/{package_name}/config", ["config/sampler.yaml", "config/online_curobo.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mayank",
    maintainer_email="mayank@example.com",
    description="RGB-D capture and MNTFields sample generation for a UR-mounted RealSense camera.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rgbd_mntfields_sampler = ur_mntfields_rgbd.rgbd_mntfields_sampler:main",
            "online_rgbd_field_curobo = ur_mntfields_rgbd.online_field_curobo:main",
            "offline_mntfields_train = ur_mntfields_rgbd.offline_field_train:main",
            "offline_mntfields_path = ur_mntfields_rgbd.offline_field_path:main",
            "view_mntfields_samples_3d = ur_mntfields_rgbd.view_samples_3d:main",
            "view_mntfields_field_path_3d = ur_mntfields_rgbd.view_field_path_3d:main",
        ],
    },
)
