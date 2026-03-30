#!/usr/bin/env python3

import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

def ros_camera_matrix(camera_info: CameraInfo):
    return np.array(camera_info.k, dtype=np.float64).reshape(3, 3)


def ros_fisheye_distortion(camera_info: CameraInfo):
    coeffs = list(camera_info.d[:4])
    if len(coeffs) < 4:
        coeffs.extend([0.0] * (4 - len(coeffs)))
    return np.array(coeffs, dtype=np.float64)


def image_msg_to_numpy(msg: Image):
    if msg.encoding not in ('mono8', '8UC1'):
        raise ValueError(f'Unsupported fisheye encoding: {msg.encoding}')

    image = np.frombuffer(msg.data, dtype=np.uint8)
    image = image.reshape((msg.height, msg.step))
    return image[:, :msg.width].copy()


def invert_extrinsics(R, T):
    R_inv = R.T
    T_inv = -R_inv @ T
    return R_inv, T_inv

class CameraPoseDepthForward(Node):

    DEFAULT_EXTRINSICS_PATH = (
        Path(__file__).resolve().parent / 'cv' / 'calibration' / 't265_calibration' / 't265_stereo_extrinsics.yaml'
    )

    def __init__(self):
        super().__init__('camera_pose_depth_forward')

        # ── ROS2 pub/sub ────────────────────────────────────────────────────
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(
            Odometry, '/camera/pose/sample', self.pose_callback, qos)

        self.pub_vision_pose = self.create_publisher(
            PoseStamped, '/mavros/vision_pose/pose', 10)

        self.pub_mavros_setpoint = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 20)

        self.pub_depth_map = self.create_publisher(
            Image, 'mavros/vision_pose/depth_map', 10
        )
            
        # ── Shared depth state ──────────────────────────────────────────────
        self._depth_lock   = threading.Lock()
        self._depth_map    = None          # float32 array (cropped centre)
        self._frame_mutex  = threading.Lock()
        self._frame_data   = {"left": None, "right": None, "ts": None}
        self._depth_thread = None
        self._stereo_ready = False
        self._depth_enabled = False
        self._ros_left_info = None
        self._ros_right_info = None
        self._ros_depth_error_logged = False
        self._extrinsics_path = Path(
            self.declare_parameter(
                't265_extrinsics_path',
                str(self.DEFAULT_EXTRINSICS_PATH),
            ).value
        ).expanduser()

        # ── Start ROS-topic depth source if calibration is available ───────
        self._setup_ros_fisheye_subscribers(qos)

        self.get_logger().info('camera_pose_depth_forward node started')

    # -----------------------------------------------------------------------
    # ROS fisheye setup
    # -----------------------------------------------------------------------

    # helper function for depth visualization
    def _numpy_to_image_msg(self, depth: np.ndarray) -> Image:
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_fisheye1_optical_frame'
        msg.height, msg.width = depth.shape
        msg.encoding = '32FC1'          # float32 single-channel
        msg.is_bigendian = False
        msg.step = msg.width * 4        # 4 bytes per float32
        msg.data = depth.tobytes()
        return msg

    def _setup_ros_fisheye_subscribers(self, qos):
        extrinsics = self._load_ros_extrinsics()
        if extrinsics is None:
            self.get_logger().warn(
                'Depth estimation disabled. ROS fisheye topics are available, but stereo extrinsics '
                f'file was not found at {self._extrinsics_path}. Generate it once with '
                'export_t265_intrinsics.py --source device when the camera is not in use.')
            return

        self._ros_R, self._ros_T = extrinsics
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self._left_info_callback, qos)
        self.create_subscription(CameraInfo, '/camera/fisheye2/camera_info', self._right_info_callback, qos)
        self.create_subscription(Image, '/camera/fisheye1/image_raw', self._left_image_callback, qos)
        self.create_subscription(Image, '/camera/fisheye2/image_raw', self._right_image_callback, qos)

    def _load_ros_extrinsics(self):
        if not self._extrinsics_path.is_file():
            return None

        data = yaml.safe_load(self._extrinsics_path.read_text())
        rotation = np.array(data['rotation_row_major'], dtype=np.float64).reshape(3, 3)
        translation = np.array(data['translation_m'], dtype=np.float64)
        source = data.get('source')
        target = data.get('target')

        if source == 'fisheye1' and target == 'fisheye2':
            return rotation, translation
        if source == 'fisheye2' and target == 'fisheye1':
            return invert_extrinsics(rotation, translation)

        raise RuntimeError(
            f'Unsupported stereo extrinsics frame order in {self._extrinsics_path}: '
            f'source={source}, target={target}')

    def _left_info_callback(self, msg):
        self._ros_left_info = msg
        self._maybe_configure_ros_stereo()

    def _right_info_callback(self, msg):
        self._ros_right_info = msg
        self._maybe_configure_ros_stereo()

    def _maybe_configure_ros_stereo(self):
        if self._stereo_ready or self._ros_left_info is None or self._ros_right_info is None:
            return

        self._configure_stereo(
            K_left=ros_camera_matrix(self._ros_left_info),
            D_left=ros_fisheye_distortion(self._ros_left_info),
            K_right=ros_camera_matrix(self._ros_right_info),
            D_right=ros_fisheye_distortion(self._ros_right_info),
            width=self._ros_left_info.width,
            height=self._ros_left_info.height,
            R=self._ros_R,
            T=self._ros_T,
            source='ros-topics',
        )

    def _left_image_callback(self, msg):
        try:
            left = image_msg_to_numpy(msg)
        except ValueError as exc:
            if not self._ros_depth_error_logged:
                self._ros_depth_error_logged = True
                self.get_logger().warn(str(exc))
            return

        with self._frame_mutex:
            self._frame_data['left'] = left
            self._frame_data['ts'] = msg.header.stamp

    def _right_image_callback(self, msg):
        try:
            right = image_msg_to_numpy(msg)
        except ValueError as exc:
            if not self._ros_depth_error_logged:
                self._ros_depth_error_logged = True
                self.get_logger().warn(str(exc))
            return

        with self._frame_mutex:
            self._frame_data['right'] = right
            self._frame_data['ts'] = msg.header.stamp

    def _configure_stereo(self, *, K_left, D_left, K_right, D_right, width, height, R, T, source):
        # Downscale for performance (matching working reference)
        scale = 0.5

        # --- Manual half-rotation split (avoiding buggy cv2.fisheye.stereoRectify) ---
        # Split the inter-camera rotation evenly so both cameras rotate
        # towards a common virtual fronto-parallel plane.
        rvec, _ = cv2.Rodrigues(R)
        R1, _ = cv2.Rodrigues(rvec / 2.0)
        R2, _ = cv2.Rodrigues(-rvec / 2.0)

        # Scaled intrinsics for the downscaled output images.
        # Using the original K (scaled) as the new camera matrix preserves
        # the real focal length and principal point.
        K_left_scaled = K_left.copy()
        K_left_scaled[0, 0] *= scale
        K_left_scaled[1, 1] *= scale
        K_left_scaled[0, 2] *= scale
        K_left_scaled[1, 2] *= scale

        K_right_scaled = K_right.copy()
        K_right_scaled[0, 0] *= scale
        K_right_scaled[1, 1] *= scale
        K_right_scaled[0, 2] *= scale
        K_right_scaled[1, 2] *= scale

        scaled_size = (int(width * scale), int(height * scale))

        m1type = cv2.CV_32FC1
        lm1, lm2 = cv2.fisheye.initUndistortRectifyMap(
            K_left, D_left, R1, K_left_scaled, scaled_size, m1type)
        rm1, rm2 = cv2.fisheye.initUndistortRectifyMap(
            K_right, D_right, R2, K_right_scaled, scaled_size, m1type)
        self._undistort_rectify = {'left': (lm1, lm2), 'right': (rm1, rm2)}

        self._focal_len = float(K_left_scaled[0, 0])
        self._baseline = float(abs(T[0]))
        self._scaled_size = scaled_size

        # Vertical offset correction for slight camera misalignment
        self._warp_M = np.float32([[1, 0, 0], [0, 1, -2]])

        # Disparity bounds from depth range
        min_depth = 0.1
        max_depth = 5.0
        self._min_disp = float(self._focal_len * self._baseline / max_depth)
        self._max_disp = float(self._focal_len * self._baseline / min_depth)

        # CLAHE for contrast-boosting low-texture fisheye images
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # SGBM parameters (matched to working reference)
        num_disp = 64
        block_size = 5
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=25,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # WLS post-filter to fill black splotches from failed SGBM matches
        self._right_matcher = cv2.ximgproc.createRightMatcher(self._stereo)
        self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._stereo)
        self._wls_filter.setLambda(8000)
        self._wls_filter.setSigmaColor(1.5)

        # Mask out the IMX camera body mounted between the fisheye lenses.
        # It sits at roughly the image centre on both rectified views and
        # creates false matches.  Black-out a small rectangle there.
        # Coordinates are in the downscaled frame (424×400 at scale=0.5).
        sw, sh = scaled_size
        self._imx_mask = np.ones((sh, sw), dtype=np.uint8) * 255
        mask_hw, mask_hh = int(sw * 0.06), int(sh * 0.08)  # half-widths
        cx_mask, cy_mask = sw // 2, sh // 2
        self._imx_mask[
            cy_mask - mask_hh : cy_mask + mask_hh,
            cx_mask - mask_hw : cx_mask + mask_hw,
        ] = 0

        if self._depth_thread is None:
            self._depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
            self._depth_thread.start()

        self._stereo_ready = True
        self._depth_enabled = True

        self.get_logger().info(
            f'Stereo depth ready from {source} | baseline={self._baseline*100:.1f} cm | '
            f'focal={self._focal_len:.1f}px | output={scaled_size[0]}x{scaled_size[1]}')

    # Background depth worker  (continuous; daemon so it exits with the node)


    def _depth_worker(self):
        while rclpy.ok():
            if not self._stereo_ready:
                time.sleep(0.05)
                continue

            with self._frame_mutex:
                if self._frame_data['left'] is None or self._frame_data['right'] is None:
                    time.sleep(0.01)
                    continue
                left = self._frame_data['left'].copy()
                right = self._frame_data['right'].copy()

            # Undistort + rectify + downscale (all in one remap)
            lm1, lm2 = self._undistort_rectify["left"]
            rm1, rm2 = self._undistort_rectify["right"]
            left_r  = cv2.remap(left,  lm1, lm2, cv2.INTER_LINEAR)
            right_r = cv2.remap(right, rm1, rm2, cv2.INTER_LINEAR)

            # Vertical alignment correction
            right_r = cv2.warpAffine(
                right_r, self._warp_M,
                (right_r.shape[1], right_r.shape[0]))

            # Boost contrast on low-texture fisheye images
            left_r  = self._clahe.apply(left_r)
            right_r = self._clahe.apply(right_r)

            # Black out IMX camera body in both views so SGBM ignores it
            left_r  = cv2.bitwise_and(left_r, self._imx_mask)
            right_r = cv2.bitwise_and(right_r, self._imx_mask)

            # Disparity: left→right and right→left for WLS hole-filling
            disp_left  = self._stereo.compute(left_r, right_r)
            disp_right = self._right_matcher.compute(right_r, left_r)

            # WLS filter fills black splotches from failed SGBM matches
            disp_filtered = self._wls_filter.filter(
                disp_left, left_r, disparity_map_right=disp_right)

            disp = disp_filtered.astype(np.float32) / 16.0

            # Zero out invalid disparities (outside useful depth range)
            disp = np.where(
                (disp > self._min_disp) & (disp < self._max_disp),
                disp, 0.0)

            # depth = f * b / disparity   (metres)
            with np.errstate(divide='ignore', invalid='ignore'):
                depth = np.where(
                    disp > 0,
                    self._focal_len * self._baseline / disp,
                    0.0).astype(np.float32)

            with self._depth_lock:
                self._depth_map = depth

                depth_msg = self._numpy_to_image_msg(depth)
                self.pub_depth_map.publish(depth_msg)


    def get_depth_at_pixel(self, u: int, v: int) -> Optional[float]:
        """
        Returns depth in metres at pixel (u, v) of the rectified depth map,
        or None if the depth map is not yet available / pixel invalid.

        Coordinate origin (0, 0) is the top-left of the downscaled
        rectified image (half the original fisheye resolution).
        """
        with self._depth_lock:
            if self._depth_map is None:
                return None
            h, w = self._depth_map.shape
            if not (0 <= v < h and 0 <= u < w):
                self.get_logger().warn(
                    f'get_depth_at_pixel: ({u},{v}) out of bounds [{w}×{h}]')
                return None
            depth = float(self._depth_map[v, u])

        return depth if depth > 0.0 else None

    # -----------------------------------------------------------------------
    # ROS2 pose callback
    # -----------------------------------------------------------------------

    def pose_callback(self, msg):
        # Forward pose to MAVROS
        pose_out = PoseStamped()
        pose_out.header.stamp    = msg.header.stamp
        pose_out.header.frame_id = msg.header.frame_id
        pose_out.pose            = msg.pose.pose
        self.pub_vision_pose.publish(pose_out)

        # Hover setpoint
        setpoint = PoseStamped()
        setpoint.header.frame_id       = 'map'
        setpoint.pose.position.z = 1.5
        self.pub_mavros_setpoint.publish(setpoint)

        # ── Example: log depth at image centre ──────────────────────────────
        with self._depth_lock:
            dm = self._depth_map
        if dm is not None:
            cx, cy = dm.shape[1] // 2, dm.shape[0] // 2
            d = self.get_depth_at_pixel(cx, cy)
            if d is not None:
                self.get_logger().debug(f'Depth at centre: {d:.3f} m')

    # -----------------------------------------------------------------------

    def destroy_node(self):
        super().destroy_node()


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseDepthForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()