#!/usr/bin/env python3
"""
ROS2 node to take centroid from IMX219 YOLO detector,
transform it to the T265 left fisheye frame, get depth from
T265 stereo, and publish a PoseStamped message in the fisheye1 frame.
"""

from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import CameraInfo, Image
from roborangers.realsense import CameraPoseDepthForward

# (ros_camera_matrix, ros_fisheye_distortion, image_msg_to_numpy, etc.)
# For this example, we will assume CameraPoseDepthForward is imported.

def undistort_to_normalized(cx, cy, K, D):
    pts = np.array([[[cx, cy]]], dtype=np.float32)

    undistorted = cv2.undistortPoints(
        pts,
        cameraMatrix=K,
        distCoeffs=D
    )

    x = undistorted[0, 0, 0]
    y = undistorted[0, 0, 1]

    return x, y

def transform_point(R, T, pt):
    """
    Apply extrinsic transform to 3D point: cam0 -> cam1
    """
    return R @ pt + T

class CentroidDepthPublisher(Node):
    def __init__(self):
        super().__init__('centroid_depth_publisher')

        # --- Parameters ---
        self.declare_parameter('publish_topic', '/vision/centroid_3d')
        self.declare_parameter('centroid_topic', '/vision/target_centroid')
        self.declare_parameter('camera_info_topic', '/camera/fisheye1/camera_info')
        self.declare_parameter('publish_frame_id', 'fisheye1_frame')

        # Extrinsics IMX to T265 left fisheye (calibrated with Kalibr)
        self.T_cn_cnm1 = np.array([
            [0.99987, -0.00441, 0.01544, 0.00940],
            [0.00466, 0.99985, -0.01634, 0.01165],
            [-0.01537, 0.01641, 0.99974, -0.01066],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.R = self.T_cn_cnm1[:3, :3]
        self.T = self.T_cn_cnm1[:3, 3]

        # Camera intrinsics 
        self.K_0 = np.array([
            [825.89945, 0, 831.00564],
            [0, 810.456425, 688.748746],
            [0, 0, 1]
        ])
        
        self.K_1 = np.array([
            [288.39133, 0.0, 417.73537],
            [0.0, 288.80452, 400.45697],
            [0.0, 0.0, 1.0]
        ])

        # Distortion coefficients for imx
        self.D_0 = np.array([
            -0.27493,
            0.05107,
            -0.00923,
            -0.00031,
            0.0
        ])

        # init pubs and subs
        self.centroid_sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter('centroid_topic').value,
            self.centroid_callback,
            10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('publish_topic').value,
            10
        )

        self.create_subscription(CameraInfo,
                                 self.get_parameter('camera_info_topic').value,
                                 self.camera_info_callback, 10)

        self.depth_node = CameraPoseDepthForward()
        self.get_logger().info('Centroid depth publisher initialized.')

    def camera_info_callback(self, msg: CameraInfo):
        if self.K_0 is None:
            self.K_0 = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f'Camera intrinsics loaded: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}')

    def centroid_callback(self, msg: Float32MultiArray):
        if self.K_0 is None:
            self.get_logger().warn('No camera intrinsics yet, skipping centroid.')
            return

        if self.depth_node._depth_map is None:
            self.get_logger().warn('No depth map yet, skipping centroid.')
            return

        # Extract pixel centroid
        cx, cy, nx, ny, conf, cls_id = msg.data

        # Undistort imx pixels (bc of wide lens resolution) and convert into normalized image point
        pt_cam0 = undistort_to_normalized(int(cx), int(cy), self.K_0, self.D_0)
        # Transform into fisheye frame 
        x_0, y_0 = pt_cam0
        ray_imx = [x_0, y_0, 1]

        pt_cam1 = transform_point(self.R, self.T, ray_imx)

        #normalized in fisheye frame
        nx_rect = pt_cam1[0] / pt_cam1[2]
        ny_rect = pt_cam1[1] / pt_cam1[2]

        fisheye_rect_pixel_coords = self.K_1 @ np.array([nx_rect, ny_rect, 1.0])
        depth = self.depth_node.get_depth_at_pixel(fisheye_rect_pixel_coords[0], fisheye_rect_pixel_coords[1])

        if depth is None:
            self.get_logger().warn('No valid depth at centroid, skipping.')
            return

        #final pose in fisheye frame
        pt_cam1 = np.array([nx_rect * depth, ny_rect * depth, depth])

        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.get_parameter('publish_frame_id').value
        pose_msg.pose.position.x = float(pt_cam1[0])
        pose_msg.pose.position.y = float(pt_cam1[1])
        pose_msg.pose.position.z = float(pt_cam1[2])
        # Orientation unknown, set identity quaternion
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)
        self.get_logger().debug(f'Published centroid 3D: {pt_cam1}')

def main(args=None):
    rclpy.init(args=args)
    node = CentroidDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()