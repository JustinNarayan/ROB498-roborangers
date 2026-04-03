#!/usr/bin/env python3
"""Dump T265 fisheye intrinsics and stereo extrinsics to YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

try:
    from roborangers.cv.calibration.common import (
        camera_info_to_yaml_dict,
        flatten_matrix_rows,
        intrinsics_to_camera_info,
        write_yaml,
    )
except ModuleNotFoundError:
    from common import camera_info_to_yaml_dict, flatten_matrix_rows, intrinsics_to_camera_info, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export T265 fisheye intrinsics and extrinsics")
    parser.add_argument("--output-dir", default="./t265_calibration", help="Directory to write YAML files")
    parser.add_argument("--distortion-model", default="equidistant", help="ROS distortion model string")
    parser.add_argument("--left-frame-id", default="camera_fisheye1_optical_frame")
    parser.add_argument("--right-frame-id", default="camera_fisheye2_optical_frame")
    parser.add_argument("--left-camera-info-topic", default="/camera/fisheye1/camera_info")
    parser.add_argument("--right-camera-info-topic", default="/camera/fisheye2/camera_info")
    parser.add_argument("--ros-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--source",
        choices=("auto", "device", "topics"),
        default="auto",
        help="Calibration source: direct RealSense device access, ROS topics, or automatic fallback.",
    )
    return parser


class CameraInfoCollector(Node):
    def __init__(self, left_topic: str, right_topic: str) -> None:
        super().__init__("t265_intrinsics_exporter")
        self.left_camera_info: Optional[CameraInfo] = None
        self.right_camera_info: Optional[CameraInfo] = None

        self.create_subscription(CameraInfo, left_topic, self._left_callback, 10)
        self.create_subscription(CameraInfo, right_topic, self._right_callback, 10)

    def _left_callback(self, msg: CameraInfo) -> None:
        self.left_camera_info = msg

    def _right_callback(self, msg: CameraInfo) -> None:
        self.right_camera_info = msg


def export_from_ros_topics(args, output_dir: Path) -> None:
    rclpy.init(args=None)
    node = CameraInfoCollector(args.left_camera_info_topic, args.right_camera_info_topic)

    try:
        deadline_ns = node.get_clock().now().nanoseconds + int(args.ros_timeout_sec * 1e9)
        while node.get_clock().now().nanoseconds < deadline_ns:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.left_camera_info is not None and node.right_camera_info is not None:
                break

        if node.left_camera_info is None or node.right_camera_info is None:
            missing_topics = []
            if node.left_camera_info is None:
                missing_topics.append(args.left_camera_info_topic)
            if node.right_camera_info is None:
                missing_topics.append(args.right_camera_info_topic)
            raise RuntimeError(
                "Timed out waiting for camera_info messages on: " + ", ".join(missing_topics)
            )

        write_yaml(
            output_dir / "t265_fisheye1.yaml",
            camera_info_to_yaml_dict(node.left_camera_info, "t265_fisheye1"),
        )
        write_yaml(
            output_dir / "t265_fisheye2.yaml",
            camera_info_to_yaml_dict(node.right_camera_info, "t265_fisheye2"),
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"Wrote T265 fisheye YAML files to {output_dir} from ROS camera_info topics")
    print("Skipped t265_stereo_extrinsics.yaml because camera_info topics do not expose stereo extrinsics")


def export_from_device(args, output_dir: Path) -> None:
    if rs is None:
        raise ModuleNotFoundError("pyrealsense2 is required for --source device")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.fisheye, 1)
    config.enable_stream(rs.stream.fisheye, 2)
    pipeline.start(config)

    try:
        profile = pipeline.get_active_profile()
        left_profile = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
        right_profile = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
        left_intrinsics = left_profile.get_intrinsics()
        right_intrinsics = right_profile.get_intrinsics()
        right_to_left = right_profile.get_extrinsics_to(left_profile)

        left_info = intrinsics_to_camera_info(
            left_intrinsics,
            frame_id=args.left_frame_id,
            distortion_model=args.distortion_model,
        )
        right_info = intrinsics_to_camera_info(
            right_intrinsics,
            frame_id=args.right_frame_id,
            distortion_model=args.distortion_model,
        )

        write_yaml(output_dir / "t265_fisheye1.yaml", camera_info_to_yaml_dict(left_info, "t265_fisheye1"))
        write_yaml(output_dir / "t265_fisheye2.yaml", camera_info_to_yaml_dict(right_info, "t265_fisheye2"))
        write_yaml(
            output_dir / "t265_stereo_extrinsics.yaml",
            {
                "source": "fisheye2",
                "target": "fisheye1",
                "rotation_row_major": flatten_matrix_rows(
                    [
                        right_to_left.rotation[0:3],
                        right_to_left.rotation[3:6],
                        right_to_left.rotation[6:9],
                    ]
                ),
                "translation_m": [float(value) for value in right_to_left.translation],
            },
        )
    finally:
        pipeline.stop()

    print(f"Wrote T265 calibration YAML files to {output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "topics":
        export_from_ros_topics(args, output_dir)
        return

    if args.source == "device":
        export_from_device(args, output_dir)
        return

    try:
        export_from_device(args, output_dir)
    except (ModuleNotFoundError, RuntimeError) as exc:
        print(f"Direct RealSense device export failed: {exc}")
        print("Falling back to ROS camera_info topics")
        export_from_ros_topics(args, output_dir)


if __name__ == "__main__":
    main()