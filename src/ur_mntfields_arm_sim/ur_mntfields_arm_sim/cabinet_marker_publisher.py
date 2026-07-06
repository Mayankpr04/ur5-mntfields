from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class CabinetMarkerPublisher(Node):
    def __init__(self):
        super().__init__("cabinet_marker_publisher")
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("marker_topic", "/ur_mntfields_arm_sim/cabinet_markers")
        self.declare_parameter("scene_boxes", ["0.70,0.00,0.45,0.04,0.70,0.90"])
        self.declare_parameter("pedestal_boxes", ["0.15,0.35,0.25,0.55,0.55,0.50"])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.boxes = [
            np.asarray([float(x) for x in entry.split(",")], dtype=np.float64)
            for entry in self.get_parameter("scene_boxes").value
        ]
        self.pedestal_boxes = [
            np.asarray([float(x) for x in entry.split(",")], dtype=np.float64)
            for entry in self.get_parameter("pedestal_boxes").value
        ]
        self.pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 2
        )
        self.get_logger().info(
            f"Cabinet markers enabled: topic={self.get_parameter('marker_topic').value} "
            f"frame={self.base_frame} scene_boxes={len(self.boxes)} pedestal_boxes={len(self.pedestal_boxes)}"
        )
        self.create_timer(0.5, self._tick)

    def _tick(self):
        arr = MarkerArray()
        for idx, values in enumerate(self.boxes):
            cx, cy, cz, sx, sy, sz = values.tolist()
            marker = Marker()
            marker.header = Header(frame_id=self.base_frame)
            marker.ns = "cabinet"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz
            marker.pose.orientation.w = 1.0
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz
            marker.color.r = 0.70
            marker.color.g = 0.60
            marker.color.b = 0.45
            marker.color.a = 0.55
            arr.markers.append(marker)
        for idx, values in enumerate(self.pedestal_boxes, start=len(self.boxes)):
            cx, cy, cz, sx, sy, sz = values.tolist()
            marker = Marker()
            marker.header = Header(frame_id=self.base_frame)
            marker.ns = "pedestal"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz
            marker.pose.orientation.w = 1.0
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz
            marker.color.r = 0.45
            marker.color.g = 0.45
            marker.color.b = 0.48
            marker.color.a = 0.65
            arr.markers.append(marker)
        self.pub.publish(arr)


def main():
    rclpy.init()
    node = CabinetMarkerPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
