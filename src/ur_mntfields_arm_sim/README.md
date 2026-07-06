# ur_mntfields_arm_sim

Synthetic testbed for `ur_mntfields_arm`.

It launches:

- UR5 fake hardware with controllers
- a URDF wrapper that adds a wrist RGB-D camera frame chain
- a custom depth simulator that raycasts against cabinet boxes
- the arm MNTFields explorer
- a trajectory executor that forwards planned joint trajectories to the fake UR controller
