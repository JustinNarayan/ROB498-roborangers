#!/usr/bin/env python3
"""Dump T265 fisheye intrinsics and stereo extrinsics to YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pyrealsense2 as rs
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pyrealsense2 is required to query T265 calibration data."
    ) from exc

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()