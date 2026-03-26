#!/usr/bin/env python3
"""Publish T265 fisheye frames and camera info without the realsense-ros wrapper."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

try:
    import pyrealsense2 as rs
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pyrealsense2 is required to publish T265 fisheye topics."
    ) from exc

try:
    from roborangers.cv.calibration.common import (
        clone_camera_info,
        intrinsics_to_camera_info,
        numpy_to_image_msg,
    )
except ModuleNotFoundError:
    from common import clone_camera_info, intrinsics_to_camera_info, numpy_to_image_msg


class T265FisheyePublisher(Node):
    def __init__(self) -> None:
        super().__init__("t265_fisheye_publisher")

        self.declare_parameter("publish_fisheye1", True)
        self.declare_parameter("publish_fisheye2", True)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("f1_image_topic", "/camera/fisheye1/image_raw")
        self.declare_parameter("f1_camera_info_topic", "/camera/fisheye1/camera_info")
        self.declare_parameter("f1_frame_id", "camera_fisheye1_optical_frame")
        self.declare_parameter("f2_image_topic", "/camera/fisheye2/image_raw")
        self.declare_parameter("f2_camera_info_topic", "/camera/fisheye2/camera_info")
        self.declare_parameter("f2_frame_id", "camera_fisheye2_optical_frame")
        self.declare_parameter("distortion_model", "equidistant")

        self.publish_fisheye1 = bool(self.get_parameter("publish_fisheye1").value)
        self.publish_fisheye2 = bool(self.get_parameter("publish_fisheye2").value)
        if not self.publish_fisheye1 and not self.publish_fisheye2:
            raise ValueError("At least one fisheye stream must be enabled")

        self.distortion_model = str(self.get_parameter("distortion_model").value)
        self.frame_ids = {
            1: str(self.get_parameter("f1_frame_id").value),
            2: str(self.get_parameter("f2_frame_id").value),
        }

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.publish_fisheye1:
            config.enable_stream(rs.stream.fisheye, 1)
        if self.publish_fisheye2:
            config.enable_stream(rs.stream.fisheye, 2)

        self.pipeline.start(config)
        active_profile = self.pipeline.get_active_profile()

        self.image_publishers = {}
        self.camera_info_publishers = {}
        self.camera_info_templates = {}

        if self.publish_fisheye1:
            self.image_publishers[1] = self.create_publisher(
                Image,
                str(self.get_parameter("f1_image_topic").value),
                10,
            )
            self.camera_info_publishers[1] = self.create_publisher(
                CameraInfo,
                str(self.get_parameter("f1_camera_info_topic").value),
                10,
            )
            intrinsics1 = active_profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile().get_intrinsics()
            self.camera_info_templates[1] = intrinsics_to_camera_info(
                intrinsics1,
                frame_id=self.frame_ids[1],
                distortion_model=self.distortion_model,
            )

        if self.publish_fisheye2:
            self.image_publishers[2] = self.create_publisher(
                Image,
                str(self.get_parameter("f2_image_topic").value),
                10,
            )
            self.camera_info_publishers[2] = self.create_publisher(
                CameraInfo,
                str(self.get_parameter("f2_camera_info_topic").value),
                10,
            )
            intrinsics2 = active_profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile().get_intrinsics()
            self.camera_info_templates[2] = intrinsics_to_camera_info(
                intrinsics2,
                frame_id=self.frame_ids[2],
                distortion_model=self.distortion_model,
            )

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, publish_rate_hz), self._publish_frames)
        self.get_logger().info("Publishing T265 fisheye frames for calibration")

    def _publish_frames(self) -> None:
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        stamp = self.get_clock().now().to_msg()

        for stream_index in self.image_publishers:
            frame = frames.get_fisheye_frame(stream_index)
            if not frame:
                self.get_logger().warning(f"Missing fisheye frame {stream_index}")
                continue

            image = np.asanyarray(frame.get_data())
            image_msg = numpy_to_image_msg(image, stamp, self.frame_ids[stream_index], "mono8")
            camera_info_msg = clone_camera_info(
                self.camera_info_templates[stream_index],
                stamp,
                self.frame_ids[stream_index],
                int(frame.get_width()),
                int(frame.get_height()),
            )

            self.image_publishers[stream_index].publish(image_msg)
            self.camera_info_publishers[stream_index].publish(camera_info_msg)

    def destroy_node(self):
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.stop()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = T265FisheyePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()