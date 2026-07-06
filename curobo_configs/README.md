# cuRobo UR5 Wrist-Camera Config

This directory contains a first-cut cuRobo robot config for the UR5 simulation
with the fixed wrist camera.

Files:

- `ur5_wrist_camera.yml`
  cuRobo robot config used by the new 3D field + cuRobo pipeline.
- `generate_ur5_wrist_camera_urdf.sh`
  Expands the ROS xacro into the URDF file referenced by the cuRobo config.

Before using the config:

```bash
cd /home/mayank/ur_ws
bash curobo_configs/generate_ur5_wrist_camera_urdf.sh
```

Then launch the 3D pipeline with:

```bash
source ~/curobo/.venv/bin/activate
source /opt/ros/humble/setup.bash
source /home/mayank/ur_ws/install/setup.bash
ros2 launch ur_mntfields_arm_sim ur_mntfields_rgbd_curobo_gz.launch.py \
  enable_curobo:=true \
  curobo_follow_mode:=anchor_chain \
  curobo_robot_config:=/home/mayank/ur_ws/curobo_configs/ur5_wrist_camera.yml \
  curobo_tool_frame:=camera_link
```

This is a first-pass config. The collision spheres and self-collision settings
may need tuning after the first live planning tests.
