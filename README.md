# UR5 MNTFields Online Exploration

ROS 2 workspace packages for online MNTFields/NTFields-style exploration with a UR5 arm and wrist-mounted RGB-D camera.

## Repository Layout

- `src/ur_mntfields_arm`: field model, sampler, planner, training loop, diagnostics, and offline viewers.
- `src/ur_mntfields_arm_sim`: Gazebo/RViz simulation launch files, URDF camera mount, cabinet worlds, scene configs, and trajectory utilities.
- `src/ur_mntfields_rgbd`: older RGB-D sampler/training utilities and cuRobo-related experiments.
- `curobo_configs`: small UR5 wrist-camera cuRobo configuration files.

Generated training outputs, checkpoints, point clouds, images, build products, and external reference repositories are intentionally ignored.

## Branches

- `main`: empty-cabinet simulation and baseline online training stack.
- `obstacle-cabinet`: same stack, with additional cabinet objects for testing field learning/planning around obstacles.

## Baseline Empty Cabinet

```bash
colcon build --packages-select ur_mntfields_arm ur_mntfields_arm_sim ur_mntfields_rgbd --symlink-install
source install/setup.bash
ros2 launch ur_mntfields_arm_sim ur_mntfields_arm_gz.launch.py clearance_backend:=sdf
```

## Obstacle Cabinet

```bash
colcon build --packages-select ur_mntfields_arm ur_mntfields_arm_sim ur_mntfields_rgbd --symlink-install
source install/setup.bash
ros2 launch ur_mntfields_arm_sim ur_mntfields_arm_obstacles_gz.launch.py clearance_backend:=sdf
```

The obstacle-cabinet run writes artifacts to `src/ur5_sim_training_obstacles`.
