#!/usr/bin/env python3
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

try:
    from roborangers.cv.calibration.common import (
        camera_backend_diagnostic,
        clone_camera_info,
        gstreamer_pipeline,
        load_camera_info_yaml,
        make_default_camera_info,
        numpy_to_image_msg,
        opencv_has_gstreamer,
    )
except ModuleNotFoundError:
    from common import (
        camera_backend_diagnostic,
        clone_camera_info,
        gstreamer_pipeline,
        load_camera_info_yaml,
        make_default_camera_info,
        numpy_to_image_msg,
        opencv_has_gstreamer,
    )


class Imx219CalibrationPublisher(Node):
    def __init__(self) -> None:
        super().__init__("imx219_calibration_publisher")

        self.declare_parameter("camera_name",        "imx219")
        self.declare_parameter("frame_id",           "imx219_optical_frame")
        self.declare_parameter("image_topic",        "/imx219/image_raw")
        self.declare_parameter("camera_info_topic",  "/imx219/camera_info")
        self.declare_parameter("camera_info_path",   "")
        self.declare_parameter("distortion_model",   "plumb_bob")
        self.declare_parameter("capture_width",      1640)
        self.declare_parameter("capture_height",     1232)
        self.declare_parameter("display_width",      1640)
        self.declare_parameter("display_height",     1232)
        self.declare_parameter("framerate",          30)
        self.declare_parameter("flip_method",        2)
        self.declare_parameter("sensor_mode",        3)
        self.declare_parameter("sensor_id",          0)
        self.declare_parameter("publish_rate_hz",    30.0)
        self.declare_parameter("jpeg_quality",       90)     # ← new
        self.declare_parameter("preview_enabled",    True)
        self.declare_parameter("preview_window_name", "IMX219 Calibration Preview")

        self.camera_name     = str(self.get_parameter("camera_name").value)
        self.frame_id        = str(self.get_parameter("frame_id").value)
        image_topic          = str(self.get_parameter("image_topic").value)
        camera_info_topic    = str(self.get_parameter("camera_info_topic").value)
        capture_width        = int(self.get_parameter("capture_width").value)
        capture_height       = int(self.get_parameter("capture_height").value)
        display_width        = int(self.get_parameter("display_width").value)
        display_height       = int(self.get_parameter("display_height").value)
        framerate            = int(self.get_parameter("framerate").value)
        flip_method          = int(self.get_parameter("flip_method").value)
        sensor_mode          = int(self.get_parameter("sensor_mode").value)
        sensor_id            = int(self.get_parameter("sensor_id").value)
        distortion_model     = str(self.get_parameter("distortion_model").value)
        publish_rate_hz      = float(self.get_parameter("publish_rate_hz").value)
        self._jpeg_quality   = int(self.get_parameter("jpeg_quality").value)
        self._preview_enabled = bool(self.get_parameter("preview_enabled").value)
        self._preview_window_name = str(self.get_parameter("preview_window_name").value)
        self._preview_available = self._preview_enabled

        if not opencv_has_gstreamer(cv2):
            raise RuntimeError(camera_backend_diagnostic(cv2))

        pipeline = gstreamer_pipeline(
            capture_width=capture_width,
            capture_height=capture_height,
            display_width=display_width,
            display_height=display_height,
            framerate=framerate,
            flip_method=flip_method,
            sensor_mode=sensor_mode,
            sensor_id=sensor_id,
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open IMX219 camera pipeline.\nPipeline: {pipeline}\n"
                f"{camera_backend_diagnostic(cv2)}"
            )

        # ── Shared frame state ───────────────────────────────────────────────
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._frame_ready  = False

        # ── Camera reader thread ─────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._camera_reader, daemon=True)
        self._reader_thread.start()

        # ── Publishers ───────────────────────────────────────────────────────
        # Raw publisher kept for camera_calibration (intrinsics) workflow
        #self.image_pub = self.create_publisher(Image, image_topic, 10)

        # Compressed publisher — used for Kalibr bag recording
        compressed_topic = image_topic# + "/compressed"
        self.compressed_pub = self.create_publisher(
            CompressedImage, compressed_topic, 10)

        self.camera_info_pub = self.create_publisher(
            CameraInfo, camera_info_topic, 10)

        # ── Camera info template ─────────────────────────────────────────────
        camera_info_path = str(self.get_parameter("camera_info_path").value).strip()
        if camera_info_path:
            path = Path(camera_info_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"camera_info_path does not exist: {path}")
            self.camera_info_template = load_camera_info_yaml(
                path,
                frame_id=self.frame_id,
                camera_name=self.camera_name,
                default_width=display_width,
                default_height=display_height,
                default_distortion_model=distortion_model,
            )
        else:
            self.camera_info_template = make_default_camera_info(
                width=display_width,
                height=display_height,
                frame_id=self.frame_id,
                camera_name=self.camera_name,
                distortion_model=distortion_model,
            )

        self.timer = self.create_timer(
            1.0 / max(1.0, publish_rate_hz), self._publish_frame)

        self.get_logger().info(
            f"Publishing IMX219 frames:\n"
            f"  raw        → {image_topic}\n"
            f"  compressed → {compressed_topic}  (JPEG q={self._jpeg_quality})\n"
            f"  camera info→ {camera_info_topic}\n"
            f"  preview    → {'enabled' if self._preview_enabled else 'disabled'}"
        )

    # -----------------------------------------------------------------------
    # Camera reader — runs in its own thread, never blocks the ROS executor
    # -----------------------------------------------------------------------

    def _camera_reader(self) -> None:
        """Continuously drain frames from GStreamer as fast as they arrive."""
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warning(
                    "Failed to read frame from IMX219", throttle_duration_sec=2.0)
                continue
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_ready  = True

    # -----------------------------------------------------------------------
    # ROS timer callback
    # -----------------------------------------------------------------------

    def _publish_frame(self) -> None:
        with self._frame_lock:
            if not self._frame_ready:
                return
            frame = self._latest_frame.copy()

        stamp = self.get_clock().now().to_msg()

        if self._preview_available:
            self._show_preview(frame)

        # ── Raw image (for camera_calibration / intrinsics workflow) ─────────
        #image_msg = numpy_to_image_msg(frame, stamp, self.frame_id, "bgr8")
        #self.image_pub.publish(image_msg)

        # ── Compressed image (for Kalibr bag recording) ──────────────────────
        ok, jpeg_buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if ok:
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp    = stamp
            compressed_msg.header.frame_id = self.frame_id
            compressed_msg.format          = "jpeg"
            compressed_msg.data            = jpeg_buf.tobytes()
            self.compressed_pub.publish(compressed_msg)

        # ── Camera info (shared stamp for both image topics) ─────────────────
        camera_info_msg = clone_camera_info(
            self.camera_info_template,
            stamp,
            self.frame_id,
            int(frame.shape[1]),
            int(frame.shape[0]),
        )
        self.camera_info_pub.publish(camera_info_msg)

    def _show_preview(self, frame) -> None:
        preview = frame.copy()
        height, width = preview.shape[:2]
        cv2.putText(
            preview,
            f"{self.camera_name} {width}x{height}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Press q in the preview window to hide it",
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        try:
            cv2.imshow(self._preview_window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(self._preview_window_name)
                self._preview_available = False
                self.get_logger().info("Preview window closed by user")
        except cv2.error as exc:
            self._preview_available = False
            self.get_logger().warning(
                f"Preview disabled because OpenCV GUI is unavailable: {exc}")

    # -----------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._stop_event.set()
        if self._preview_enabled:
            cv2.destroyAllWindows()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = Imx219CalibrationPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()