# ur_mntfields_arm

ROS 2 package for UR5 wrist-camera exploration with MNTFields trained in 6D joint configuration space.

This package:

- fuses depth into a workspace voxel map
- keeps a persistent global frontier bank
- samples training pairs in UR5 configuration space
- trains the `mntfields_tb` network in `dim=6`
- proposes camera view poses for frontiers, solves IK, and rolls out joint-space trajectories

The implementation keeps the `mntfields_tb` network and loss, but does not use the older workspace XYZ training pipeline.
