from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .sampling import (
    depth_to_meters,
    ray_dirs_camera,
    rotation_distance_deg,
    sample_training_frame,
    transform_to_matrix,
    translation_distance,
)


class RGBDMNTFieldsSampler(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_mntfields_sampler")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("base_frame", "camera_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("output_dir", "mntfields_rgbd_output")
        self.declare_parameter("save_color", True)
        self.declare_parameter("save_depth_npy", True)
        self.declare_parameter("save_depth_png", False)
        self.declare_parameter("save_pointcloud_debug", False)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth", 0.15)
        self.declare_parameter("max_depth", 2.5)
        self.declare_parameter("n_rays", 5000)
        self.declare_parameter("n_strat_samples", 8)
        self.declare_parameter("dist_behind_surf", 0.4)
        self.declare_parameter("sample_min", 0.07)
        self.declare_parameter("sample_max", 0.3)
        self.declare_parameter("num_pairs", 10000)
        self.declare_parameter("scale_factor", 1.0)
        self.declare_parameter("bound_padding_xy", 0.05)
        self.declare_parameter("bound_padding_z", 0.05)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("capture_stride", 1)
        self.declare_parameter("min_translation_m", 0.0) #change these
        self.declare_parameter("min_rotation_deg", 0.0)
        self.declare_parameter("random_seed", 0)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame_override = self.get_parameter("camera_frame").value
        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self.save_color = bool(self.get_parameter("save_color").value)
        self.save_depth_npy = bool(self.get_parameter("save_depth_npy").value)
        self.save_depth_png = bool(self.get_parameter("save_depth_png").value)
        self.save_pointcloud_debug = bool(self.get_parameter("save_pointcloud_debug").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.n_rays = int(self.get_parameter("n_rays").value)
        self.n_strat_samples = int(self.get_parameter("n_strat_samples").value)
        self.dist_behind_surf = float(self.get_parameter("dist_behind_surf").value)
        self.sample_min = float(self.get_parameter("sample_min").value)
        self.sample_max = float(self.get_parameter("sample_max").value)
        self.num_pairs = int(self.get_parameter("num_pairs").value)
        self.scale_factor = float(self.get_parameter("scale_factor").value)
        self.bound_padding_xy = float(self.get_parameter("bound_padding_xy").value)
        self.bound_padding_z = float(self.get_parameter("bound_padding_z").value)
        self.capture_stride = max(1, int(self.get_parameter("capture_stride").value))
        self.min_translation_m = float(self.get_parameter("min_translation_m").value)
        self.min_rotation_deg = float(self.get_parameter("min_rotation_deg").value)
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        queue_size = int(self.get_parameter("queue_size").value)
        sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("color", "depth", "pose", "samples", "debug"):
            (self.output_dir / name).mkdir(exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_pose = None
        self.frame_index = 0
        self.capture_count = 0
        self.dirs_c = None
        self.camera_intrinsics = None

        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=queue_size,
            slop=sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"Writing captures to {self.output_dir}")

    def synced_callback(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        self.frame_index += 1
        if self.frame_index % self.capture_stride != 0:
            return

        frame_id = self.camera_frame_override or depth_msg.header.frame_id or color_msg.header.frame_id
        if not frame_id:
            self.get_logger().warning("Skipping frame without camera frame_id.")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                depth_msg.header.stamp,
            )
        except TransformException as exc:
            self.get_logger().warning(f"TF lookup failed for {self.base_frame} -> {frame_id}: {exc}")
            return

        t_world_camera = transform_to_matrix(transform.transform.translation, transform.transform.rotation)

        if self.last_pose is not None:
            moved = translation_distance(t_world_camera, self.last_pose)
            rotated = rotation_distance_deg(t_world_camera, self.last_pose)
            if moved < self.min_translation_m and rotated < self.min_rotation_deg:
                return

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        height = int(info_msg.height)
        width = int(info_msg.width)
        fx = float(info_msg.k[0])
        fy = float(info_msg.k[4])
        cx = float(info_msg.k[2])
        cy = float(info_msg.k[5])

        if depth.shape[0] != height or depth.shape[1] != width:
            self.get_logger().warning(
                f"Depth shape {depth.shape[:2]} does not match camera info {(height, width)}. Skipping frame."
            )
            return

        if self.dirs_c is None or self.camera_intrinsics != (height, width, fx, fy, cx, cy):
            self.dirs_c = ray_dirs_camera(height, width, fx, fy, cx, cy)
            self.camera_intrinsics = (height, width, fx, fy, cx, cy)

        depth_m = depth_to_meters(depth, self.depth_scale, self.max_depth)

        try:
            samples = sample_training_frame(
                depth_m=depth_m,
                t_world_camera=t_world_camera,
                dirs_c=self.dirs_c,
                n_rays=self.n_rays,
                n_strat_samples=self.n_strat_samples,
                dist_behind_surf=self.dist_behind_surf,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                sample_min=self.sample_min,
                sample_max=self.sample_max,
                num_pairs=self.num_pairs,
                scale_factor=self.scale_factor,
                bound_padding_xy=self.bound_padding_xy,
                bound_padding_z=self.bound_padding_z,
                rng=self.rng,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Skipping frame: {exc}")
            return

        if samples["frame_data"].shape[0] == 0:
            self.get_logger().warning("Skipping frame: raw data converted to empty Nx14 training data.")
            return

        stem = f"{self.capture_count:06d}"
        np.save(self.output_dir / "pose" / f"{stem}.npy", t_world_camera)
        np.savez_compressed(
            self.output_dir / "samples" / f"{stem}.npz",
            frame_data=samples["frame_data"],
            raw_frame_data=samples["raw_frame_data"],
            normalization_bound=samples["normalization_bound"],
            pc_world=samples["pc_world"],
            surf_pc_world=samples["surf_pc_world"],
            ray_pixels=samples["ray_pixels"],
            z_vals=samples["z_vals"],
            depth_sample=samples["depth_sample"],
            camera_pose_world=samples["camera_pose_world"],
            camera_intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
        )

        if self.save_color:
            cv2.imwrite(str(self.output_dir / "color" / f"{stem}.png"), color)

        if self.save_depth_npy:
            np.save(self.output_dir / "depth" / f"{stem}.npy", depth_m)

        if self.save_depth_png and depth.dtype == np.uint16:
            cv2.imwrite(str(self.output_dir / "depth" / f"{stem}.png"), depth)

        if self.save_pointcloud_debug:
            np.save(self.output_dir / "debug" / f"{stem}_pc_world.npy", samples["pc_world"])
            np.save(self.output_dir / "debug" / f"{stem}_surf_pc_world.npy", samples["surf_pc_world"])

        self.last_pose = t_world_camera
        self.capture_count += 1
        self.get_logger().info(
            f"Saved capture {stem} with raw {samples['raw_frame_data'].shape} and train {samples['frame_data'].shape} from {frame_id}."
        )


def main() -> None:
    rclpy.init()
    node = RGBDMNTFieldsSampler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
