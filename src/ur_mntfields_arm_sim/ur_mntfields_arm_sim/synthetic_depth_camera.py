from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class SceneBox:
    center: np.ndarray
    size: np.ndarray

    @property
    def mins(self) -> np.ndarray:
        return self.center - 0.5 * self.size

    @property
    def maxs(self) -> np.ndarray:
        return self.center + 0.5 * self.size


def _transform_to_matrix(tf_msg) -> np.ndarray:
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = [t.x, t.y, t.z]
    return out


def ray_box_intersection(origin: np.ndarray, direction: np.ndarray, box: SceneBox) -> float | None:
    mins = box.mins
    maxs = box.maxs
    inv_dir = np.where(np.abs(direction) > 1e-9, 1.0 / direction, 1e9)
    t0 = (mins - origin) * inv_dir
    t1 = (maxs - origin) * inv_dir
    tmin = np.maximum.reduce(np.minimum(t0, t1))
    tmax = np.minimum.reduce(np.maximum(t0, t1))
    if tmax < 0.0 or tmin > tmax:
        return None
    return float(tmin if tmin > 0.0 else tmax)


class SyntheticDepthCamera(Node):
    def __init__(self):
        super().__init__("synthetic_depth_camera")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("marker_topic", "/ur_mntfields_arm_sim/cabinet_markers")
        self.declare_parameter("width", 424)
        self.declare_parameter("height", 240)
        self.declare_parameter("fx", 212.0)
        self.declare_parameter("fy", 212.0)
        self.declare_parameter("cx", 212.0)
        self.declare_parameter("cy", 120.0)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("noise_std_m", 0.002)
        self.declare_parameter("publish_hz", 6.0)
        self.declare_parameter("scene_boxes", ["0.70,0.00,0.45,0.04,0.70,0.90"])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.min_depth = float(self.get_parameter("min_depth_m").value)
        self.max_depth = float(self.get_parameter("max_depth_m").value)
        self.noise_std = float(self.get_parameter("noise_std_m").value)
        self.rng = np.random.default_rng(5)

        self.boxes = [
            SceneBox(
                center=np.asarray([float(x) for x in entry.split(",")[:3]], dtype=np.float64),
                size=np.asarray([float(x) for x in entry.split(",")[3:6]], dtype=np.float64),
            )
            for entry in self.get_parameter("scene_boxes").value
        ]

        self.depth_pub = self.create_publisher(Image, str(self.get_parameter("depth_topic").value), 5)
        self.info_pub = self.create_publisher(CameraInfo, str(self.get_parameter("camera_info_topic").value), 5)
        self.marker_pub = self.create_publisher(MarkerArray, str(self.get_parameter("marker_topic").value), 2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("publish_hz").value), self._tick)
        self.frame_count = 0

        self._dirs_c = self._compute_dirs()

    def _compute_dirs(self) -> np.ndarray:
        rows = np.arange(self.height, dtype=np.float32)
        cols = np.arange(self.width, dtype=np.float32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        x = (cc - self.cx) / self.fx
        y = (rr - self.cy) / self.fy
        dirs = np.stack((x, y, np.ones_like(x)), axis=-1)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return (dirs / norms).astype(np.float32)

    def _lookup_pose(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"Camera TF unavailable: {exc}")
            return None
        return _transform_to_matrix(tf)

    def _tick(self):
        pose = self._lookup_pose()
        if pose is None:
            return
        self.frame_count += 1
        depth = np.full((self.height, self.width), 0.0, dtype=np.float32)
        origin = pose[:3, 3]
        rot = pose[:3, :3]
        dirs_w = self._dirs_c.reshape(-1, 3) @ rot.T
        best = np.full((dirs_w.shape[0],), np.inf, dtype=np.float32)
        for box in self.boxes:
            inv_dir = np.where(np.abs(dirs_w) > 1.0e-9, 1.0 / dirs_w, 1.0e9)
            t0 = (box.mins[None, :] - origin[None, :]) * inv_dir
            t1 = (box.maxs[None, :] - origin[None, :]) * inv_dir
            tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
            tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
            dist = np.where(tmin > 0.0, tmin, tmax)
            hit = (tmax >= 0.0) & (tmin <= tmax) & (dist >= self.min_depth) & (dist <= self.max_depth)
            best = np.where(hit & (dist < best), dist, best)
        hit = np.isfinite(best)
        depth.reshape(-1)[hit] = best[hit].astype(np.float32, copy=False)

        valid = depth > 0.0
        if np.any(valid):
            depth[valid] += self.rng.normal(0.0, self.noise_std, size=int(np.count_nonzero(valid))).astype(np.float32)
            depth[valid] = np.clip(depth[valid], self.min_depth, self.max_depth)
        if self.frame_count % 15 == 0:
            self.get_logger().info(
                f"Published synthetic depth: valid_pixels={int(np.count_nonzero(valid))} "
                f"camera_xyz={origin.round(3).tolist()}"
            )

        img = Image()
        img.header = Header(frame_id=self.camera_frame, stamp=self.get_clock().now().to_msg())
        img.height = self.height
        img.width = self.width
        img.encoding = "32FC1"
        img.is_bigendian = False
        img.step = self.width * 4
        img.data = depth.tobytes()
        self.depth_pub.publish(img)

        info = CameraInfo()
        info.header = img.header
        info.width = self.width
        info.height = self.height
        info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        self._publish_markers()

    def _publish_markers(self):
        arr = MarkerArray()
        for idx, box in enumerate(self.boxes):
            m = Marker()
            m.header = Header(frame_id=self.base_frame)
            m.ns = "cabinet"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(box.center[0])
            m.pose.position.y = float(box.center[1])
            m.pose.position.z = float(box.center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(box.size[0])
            m.scale.y = float(box.size[1])
            m.scale.z = float(box.size[2])
            m.color.r = 0.7
            m.color.g = 0.6
            m.color.b = 0.4
            m.color.a = 0.45
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = SyntheticDepthCamera()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
