#!/usr/bin/env python3

import math
import threading

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy


# ---------------------------------------------------------------------------
# Helpers (from t265_stereo.py)
# ---------------------------------------------------------------------------

def get_extrinsics(src, dst):
    """Returns (R, T) transform from src stream to dst stream."""
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3, 3]).T
    T = np.array(extrinsics.translation)
    return R, T


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])


def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class CameraPoseForward(Node):

    def __init__(self):
        super().__init__('camera_pose_forward')

        # ── ROS2 pub/sub ────────────────────────────────────────────────────
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(
            Odometry, '/camera/pose/sample', self.pose_callback, qos)

        self.pub_vision_pose = self.create_publisher(
            PoseStamped, '/mavros/vision_pose/pose', 10)

        self.pub_mavros_setpoint = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 20)

        # ── Shared depth state (written by RS thread, read by ROS) ──────────
        self._depth_lock   = threading.Lock()
        self._depth_map    = None          # float32 array (cropped centre)
        self._crop_offsets = None          # (rs, re, cs_valid, ce_valid) row/col slice info

        # ── Start the RealSense stereo pipeline ─────────────────────────────
        self._setup_realsense()

        self.get_logger().info('camera_pose_forward node started')

    # -----------------------------------------------------------------------
    # RealSense setup
    # -----------------------------------------------------------------------

    def _setup_realsense(self):
        """
        Configure the T265 pipeline, compute rectification maps once, and
        start an async callback that writes disparity→depth into self._depth_map.

        Frame layout (body / pose frame origin):
          - Sits midway between the two fisheye cameras.
          - fisheye(1) = left camera,  fisheye(2) = right camera.
          - Positive X  → right (toward fisheye-2 side)
          - Positive Y  → down
          - Positive Z  → backward (out of back of device)
          All depths returned below are in metres along the camera Z axis
          of the *rectified left* frame.
        """
        self._frame_mutex = threading.Lock()
        self._frame_data  = {"left": None, "right": None, "ts": None}

        self._pipe = rs.pipeline()
        cfg = rs.config()
        # enable both fisheye streams (pose is streamed separately by the
        # realsense-ros wrapper, so we don't need to enable it here)
        cfg.enable_stream(rs.stream.fisheye, 1)
        cfg.enable_stream(rs.stream.fisheye, 2)
        self._pipe.start(cfg, self._rs_callback)

        # ── Intrinsics / extrinsics ─────────────────────────────────────────
        profiles = self._pipe.get_active_profile()
        streams  = {
            "left":  profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile(),
        }
        intr = {k: streams[k].get_intrinsics() for k in streams}

        K_left  = camera_matrix(intr["left"])
        D_left  = fisheye_distortion(intr["left"])
        K_right = camera_matrix(intr["right"])
        D_right = fisheye_distortion(intr["right"])
        (width, height) = (intr["left"].width, intr["left"].height)

        R, T = get_extrinsics(streams["left"], streams["right"])

        # ── Stereo rectification ────────────────────────────────────────────
        (R_left, R_right, P_left, P_right, Q) = \
            cv2.fisheye.stereoRectify(
                K1=K_left,  D1=D_left,
                K2=K_right, D2=D_right,
                imageSize=(width, height),
                R=R, tvec=T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                newImageSize=(width, height),
                balance=0, fov_scale=1.0)[0:5]

        # Centre the principal point
        P_left[0][2]  = P_right[0][2] = width  / 2
        P_left[1][2]  = P_right[1][2] = height / 2

        # ── Undistort / rectify maps (computed once) ─────────────────────────
        m1type = cv2.CV_32FC1
        lm1, lm2 = cv2.fisheye.initUndistortRectifyMap(
            K_left,  D_left,  R_left,  P_left,  (width, height), m1type)
        rm1, rm2 = cv2.fisheye.initUndistortRectifyMap(
            K_right, D_right, R_right, P_right, (width, height), m1type)
        self._undistort_rectify = {"left": (lm1, lm2), "right": (rm1, rm2)}

        # ── Centre-crop region (edges of fisheye are unreliable) ─────────────
        half = int((height / 3) / 2)
        rs_row = int(height / 2 - half)
        re_row = int(height / 2 + half)
        cs_col = int(width  / 2 - half)
        ce_col = int(width  / 2 + half)
        Q[0][3] = Q[1][3] = -half

        # SGBM needs a head-start on the left for max_disp blank pixels
        min_disp = 0
        num_disp = 112            # must be divisible by 16
        max_disp = min_disp + num_disp
        cs_offset = min(max_disp, cs_col)
        cs_col -= cs_offset

        self._crop = (rs_row, re_row, cs_col, ce_row := ce_col, cs_offset)

        # Store Q and focal length for depth conversion
        self._Q           = Q
        self._focal_len   = Q[2][3]          # focal length in pixels
        self._baseline    = abs(T[0])         # ~0.064 m for T265
        self._P_left      = P_left
        self._min_disp    = min_disp
        self._num_disp    = num_disp

        # ── SGBM stereo matcher ──────────────────────────────────────────────
        ws = 3
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=16,
            P1=8  * 3 * ws ** 2,
            P2=32 * 3 * ws ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32)

        # ── Background worker that keeps depth_map fresh ─────────────────────
        self._depth_thread = threading.Thread(
            target=self._depth_worker, daemon=True)
        self._depth_thread.start()

        fov_h = 2 * math.atan((ce_col - cs_col) / self._focal_len / 2) * 180 / math.pi
        fov_v = 2 * math.atan((re_row - rs_row) / self._focal_len / 2) * 180 / math.pi
        self.get_logger().info(
            f'Stereo depth ready | baseline={self._baseline*100:.1f} cm | '
            f'FOV {fov_h:.0f}°W × {fov_v:.0f}°H (centre crop only)')

    # -----------------------------------------------------------------------
    # RealSense async callback  (runs on a librealsense thread)
    # -----------------------------------------------------------------------

    def _rs_callback(self, frame):
        if frame.is_frameset():
            fs = frame.as_frameset()
            left  = np.asanyarray(fs.get_fisheye_frame(1).as_video_frame().get_data())
            right = np.asanyarray(fs.get_fisheye_frame(2).as_video_frame().get_data())
            ts    = fs.get_timestamp()
            with self._frame_mutex:
                self._frame_data = {"left": left, "right": right, "ts": ts}

    # -----------------------------------------------------------------------
    # Background depth worker  (continuous; daemon so it exits with the node)
    # -----------------------------------------------------------------------

    def _depth_worker(self):
        rs_row, re_row, cs_col, ce_col, cs_offset = self._crop

        while rclpy.ok():
            with self._frame_mutex:
                if self._frame_data["ts"] is None:
                    continue
                left  = self._frame_data["left"].copy()
                right = self._frame_data["right"].copy()

            # Undistort + rectify
            lm1, lm2 = self._undistort_rectify["left"]
            rm1, rm2 = self._undistort_rectify["right"]
            left_r  = cv2.remap(left,  lm1, lm2, cv2.INTER_LINEAR)[rs_row:re_row, cs_col:ce_col]
            right_r = cv2.remap(right, rm1, rm2, cv2.INTER_LINEAR)[rs_row:re_row, cs_col:ce_col]

            # Disparity (SGBM fixed-point → divide by 16)
            disp = self._stereo.compute(left_r, right_r).astype(np.float32) / 16.0
            disp = disp[:, cs_offset:]   # trim invalid left-edge columns

            # depth = f * b / disparity   (metres)
            with np.errstate(divide='ignore', invalid='ignore'):
                depth = np.where(
                    disp > self._min_disp,
                    self._focal_len * self._baseline / disp,
                    0.0).astype(np.float32)

            with self._depth_lock:
                self._depth_map = depth

    # -----------------------------------------------------------------------
    # Public API: get depth at an (u, v) pixel in the rectified centre crop
    # -----------------------------------------------------------------------

    def get_depth_at_pixel(self, u: int, v: int) -> float | None:
        """
        Returns depth in metres at pixel (u, v) of the rectified centre-crop
        image, or None if the depth map is not yet available / pixel invalid.

        Coordinate origin (0, 0) is the top-left of the centre crop.
        The crop covers roughly the central 1/3 of the full fisheye frame,
        which is the most reliable region for passive stereo.

        Depth is measured along the optical axis (Z) of the rectified left
        camera, which is co-aligned with the T265 body/pose frame Z axis.
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
        self._pipe.stop()
        super().destroy_node()


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()