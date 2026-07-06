# ur_mntfields_rgbd

ROS 2 package for collecting synchronized Intel RealSense D435 RGB-D frames on a UR arm and converting each frame into MNTFields-style training samples.

The node expects a valid TF chain from `base_frame` to the camera optical frame carried by the depth image. That lets you provide the static transform separately, as planned.

## Output

Each accepted frame writes:

- `color/000000.png`: synchronized color frame
- `depth/000000.npy`: aligned depth in meters
- `pose/000000.npy`: `4x4` camera pose in the `base_frame`
- `samples/000000.npz`: per-frame training arrays

The sample archive now contains both formats used by current MNTFields code:

- `raw_frame_data`: `N x 12`, with columns `[x0, x1, cp0, cp1]` in world coordinates
- `frame_data`: `N x 14`, with columns `[x0_n, x1_n, y0, y1, n0, n1]`
- `normalization_bound`: `2 x 3`, the bound used to normalize `x0` and `x1`

`frame_data` is produced from `raw_frame_data` using the same conversion pattern as `mntfields_core/core/region.py`.

## Launch

```bash
ros2 launch ur_mntfields_rgbd mntfields_rgbd_sampler.launch.py
```

Override parameters as needed, for example:

```bash
ros2 launch ur_mntfields_rgbd mntfields_rgbd_sampler.launch.py \
  output_dir:=/tmp/ur5_d435_samples \
  base_frame:=base \
  camera_frame:=camera_color_optical_frame
```
