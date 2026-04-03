#!/usr/bin/env python3
"""Subscribe to the depth map topic and display it with OpenCV in real time."""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

TOPIC = 'mavros/vision_pose/depth_map'
MAX_DEPTH = 5.0  # metres – clamp for colour mapping


class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, TOPIC, self._cb, qos)
        self.get_logger().info(f'Listening on {TOPIC}')

    def _cb(self, msg: Image):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

        # Normalise to 0-255 for visualisation
        vis = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        # Black out pixels with no depth
        vis[depth <= 0] = 0

        cv2.imshow('Depth Map', vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
