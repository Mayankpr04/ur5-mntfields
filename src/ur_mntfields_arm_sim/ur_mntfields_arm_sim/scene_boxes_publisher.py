from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class SceneBoxesPublisher(Node):
    def __init__(self):
        super().__init__("scene_boxes_publisher")
        self.declare_parameter("scene_boxes_topic", "/scene_boxes")
        self.declare_parameter("scene_boxes_frame", "base_link")
        self.declare_parameter("scene_boxes", [""])
        self.declare_parameter("publish_hz", 1.0)

        self.topic = str(self.get_parameter("scene_boxes_topic").value)
        self.frame = str(self.get_parameter("scene_boxes_frame").value)
        self.boxes = self._parse_boxes(self.get_parameter("scene_boxes").value)
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(String, self.topic, qos)

        if not self.boxes:
            self.get_logger().warn(
                "scene_boxes_publisher has no boxes configured. Edit real_scene_boxes.yaml "
                "or pass scene_boxes:=['x,y,z,sx,sy,sz'] before relying on ROI initialization."
            )
        else:
            self.get_logger().info(
                f"Publishing persistent scene_boxes: topic={self.topic} frame={self.frame} boxes={len(self.boxes)}"
            )
        period = 1.0 / max(1.0e-6, float(self.get_parameter("publish_hz").value))
        self.create_timer(period, self._tick)
        self._tick()

    def _parse_boxes(self, entries) -> list[list[float]]:
        boxes: list[list[float]] = []
        for entry in entries:
            if isinstance(entry, str):
                if not entry.strip():
                    continue
                values = [float(x.strip()) for x in entry.split(",") if x.strip()]
            else:
                values = [float(x) for x in entry]
            if len(values) != 6:
                self.get_logger().warn(f"Ignoring invalid scene box entry with {len(values)} values: {entry}")
                continue
            boxes.append(values)
        return boxes

    def _tick(self):
        if not self.boxes:
            return
        msg = String()
        msg.data = json.dumps({"frame": self.frame, "boxes": self.boxes})
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SceneBoxesPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
