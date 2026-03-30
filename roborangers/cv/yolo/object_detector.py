#!/usr/bin/env python3
"""
ROS2 node to take centroid from IMX219 YOLO detector,
transform it to the T265 left fisheye frame, get depth from
T265 stereo, and publish a PoseStamped message in the fisheye1 frame.

Uses a MultiThreadedExecutor to spin both this node and
CameraPoseDepthForward together, so get_depth_at_pixel() works directly.

Subscribes to:
  /vision/target_centroid   (Float32MultiArray from centroid_detector_node)
  /camera/fisheye1/camera_info  (CameraInfo – used once to derive scaled K)

Publishes:
  /vision/centroid_3d       (PoseStamped in camera_fisheye1_optical_frame)
"""

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo

from roborangers.realsense import CameraPoseDepthForward

# Scale factor must match realsense.py _configure_stereo
DEPTH_MAP_SCALE = 0.5


class CentroidDepthPublisher(Node):
    def __init__(self, depth_node: CameraPoseDepthForward):
        super().__init__('centroid_depth_publisher')
        self._depth_node = depth_node

        # --- Parameters ---
        self.declare_parameter('publish_topic', '/vision/centroid_3d')
        self.declare_parameter('centroid_topic', '/vision/target_centroid')
        self.declare_parameter('camera_info_topic', '/camera/fisheye1/camera_info')
        self.declare_parameter('publish_frame_id', 'camera_fisheye1_optical_frame')

        # Extrinsics IMX → T265 left fisheye (calibrated with Kalibr)
        T_imx_to_fish = np.array([
            [0.99987, -0.00441, 0.01544, 0.00940],
            [0.00466,  0.99985, -0.01634, 0.01165],
            [-0.01537, 0.01641, 0.99974, -0.01066],
            [0.0,      0.0,     0.0,      1.0    ]
        ])
        self._R_imx_to_fish = T_imx_to_fish[:3, :3]

        # IMX219 intrinsics and distortion (pinhole model)
        self._K_imx = np.array([
            [825.89945, 0,         831.00564],
            [0,         810.45643, 688.74875],
            [0,         0,         1        ]
        ], dtype=np.float64)
        self._D_imx = np.array([-0.27493, 0.05107, -0.00923, -0.00031, 0.0],
                                dtype=np.float64)

        # Fisheye1 scaled intrinsics (set once from camera_info)
        self._K_fish_scaled = None

        # --- Subscriptions ---
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter('centroid_topic').value,
            self._centroid_cb, 10)

        qos_besteffort = QoSProfile(depth=10)
        qos_besteffort.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, qos_besteffort)

        # --- Publisher ---
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('publish_topic').value, 10)

        self.get_logger().info('Centroid depth publisher initialized.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo):
        if self._K_fish_scaled is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # Scale to match depth map resolution (realsense.py uses 0.5×)
        self._K_fish_scaled = K.copy()
        self._K_fish_scaled[0, 0] *= DEPTH_MAP_SCALE
        self._K_fish_scaled[1, 1] *= DEPTH_MAP_SCALE
        self._K_fish_scaled[0, 2] *= DEPTH_MAP_SCALE
        self._K_fish_scaled[1, 2] *= DEPTH_MAP_SCALE
        self.get_logger().info(
            f'Fisheye1 scaled K: fx={self._K_fish_scaled[0,0]:.1f} '
            f'fy={self._K_fish_scaled[1,1]:.1f} '
            f'cx={self._K_fish_scaled[0,2]:.1f} cy={self._K_fish_scaled[1,2]:.1f}')

    def _centroid_cb(self, msg: Float32MultiArray):
        if self._K_fish_scaled is None:
            self.get_logger().warn(
                'No fisheye camera info yet, skipping.',
                throttle_duration_sec=5.0)
            return

        cx, cy, nx, ny, conf, cls_id = msg.data

        # 1. Undistort IMX pixel → normalised ray in IMX frame
        pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        undist = cv2.undistortPoints(
            pts, cameraMatrix=self._K_imx, distCoeffs=self._D_imx)
        ray_imx = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])

        # 2. Rotate ray direction into fisheye1 frame (rotation only)
        ray_fish = self._R_imx_to_fish @ ray_imx

        # 3. Project normalised ray to depth-map pixel coordinates
        nx_fish = ray_fish[0] / ray_fish[2]
        ny_fish = ray_fish[1] / ray_fish[2]
        pixel = self._K_fish_scaled @ np.array([nx_fish, ny_fish, 1.0])
        u, v = int(round(pixel[0])), int(round(pixel[1]))

        # 4. Depth lookup via the shared depth node
        depth = self._depth_node.get_depth_at_pixel(u, v)
        if depth is None:
            self.get_logger().debug(
                f'No valid depth at ({u},{v}), skipping.')
            return

        # 5. Back-project to 3D in fisheye1 frame
        X = nx_fish * depth
        Y = ny_fish * depth
        Z = depth

        # 6. Publish PoseStamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.get_parameter('publish_frame_id').value
        pose.pose.position.x = float(X)
        pose.pose.position.y = float(Y)
        pose.pose.position.z = float(Z)
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        self.get_logger().debug(
            f'Centroid 3D: ({X:.3f}, {Y:.3f}, {Z:.3f}) depth={depth:.3f}m '
            f'pixel=({u},{v}) conf={conf:.2f}')


def main(args=None):
    rclpy.init(args=args)

    depth_node = CameraPoseDepthForward()
    centroid_node = CentroidDepthPublisher(depth_node)

    executor = MultiThreadedExecutor()
    executor.add_node(depth_node)
    executor.add_node(centroid_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        centroid_node.destroy_node()
        depth_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()