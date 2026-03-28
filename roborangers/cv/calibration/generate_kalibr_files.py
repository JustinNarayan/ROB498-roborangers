#!/usr/bin/env python3
"""Generate Kalibr camera-chain and Aprilgrid YAML files from ROS calibration data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_ros_camera_yaml(path: str | Path) -> dict:
    yaml_path = Path(path).expanduser().resolve()
    return yaml.safe_load(yaml_path.read_text())


def ros_yaml_to_kalibr_cam(
    ros_yaml: dict,
    *,
    rostopic: str,
    kalibr_camera_model: str,
    kalibr_distortion_model: str,
    xi: float | None,
) -> dict:
    matrix = ros_yaml["camera_matrix"]["data"]
    coeffs = [float(value) for value in ros_yaml["distortion_coefficients"]["data"]]
    fx = float(matrix[0])
    fy = float(matrix[4])
    cx = float(matrix[2])
    cy = float(matrix[5])

    if kalibr_camera_model == "omni":
        if xi is None:
            raise ValueError("The omni camera model requires --fisheye-xi")
        intrinsics = [float(xi), fx, fy, cx, cy]
    else:
        intrinsics = [fx, fy, cx, cy]

    return {
        "camera_model": kalibr_camera_model,
        "intrinsics": intrinsics,
        "distortion_model": kalibr_distortion_model,
        "distortion_coeffs": coeffs,
        "resolution": [int(ros_yaml["image_width"]), int(ros_yaml["image_height"])],
        "rostopic": rostopic,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Kalibr YAML files from camera calibration output")
    parser.add_argument("--imx-yaml", required=True, help="Path to IMX219 YAML from camera_calibration")
    parser.add_argument("--fisheye-yaml", required=True, help="Path to T265 fisheye YAML")
    parser.add_argument("--output-dir", default="./kalibr", help="Directory to write aprilgrid.yaml and cameras.yaml")
    parser.add_argument("--imx-topic", default="/imx219/image_raw")
    parser.add_argument("--fisheye-topic", default="/camera/fisheye1/image_raw")
    parser.add_argument("--imx-kalibr-model", default="pinhole")
    parser.add_argument("--imx-distortion-model", default="radtan")
    parser.add_argument("--fisheye-kalibr-model", default="pinhole")
    parser.add_argument("--fisheye-distortion-model", default="equidistant")
    parser.add_argument("--fisheye-xi", type=float, default=None)
    parser.add_argument("--tag-cols", type=int, default=6)
    parser.add_argument("--tag-rows", type=int, default=6)
    parser.add_argument("--tag-size", type=float, default=0.024)
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    imx_yaml = load_ros_camera_yaml(args.imx_yaml)
    fisheye_yaml = load_ros_camera_yaml(args.fisheye_yaml)

    cameras_yaml = {
        "cam0": ros_yaml_to_kalibr_cam(
            imx_yaml,
            rostopic=args.imx_topic,
            kalibr_camera_model=args.imx_kalibr_model,
            kalibr_distortion_model=args.imx_distortion_model,
            xi=None,
        ),
        "cam1": ros_yaml_to_kalibr_cam(
            fisheye_yaml,
            rostopic=args.fisheye_topic,
            kalibr_camera_model=args.fisheye_kalibr_model,
            kalibr_distortion_model=args.fisheye_distortion_model,
            xi=args.fisheye_xi,
        ),
    }

    aprilgrid_yaml = {
        "target_type": "aprilgrid",
        "tagCols": int(args.tag_cols),
        "tagRows": int(args.tag_rows),
        "tagSize": float(args.tag_size),
        "tagSpacing": float(args.tag_spacing),
    }

    (output_dir / "cameras.yaml").write_text(yaml.safe_dump(cameras_yaml, sort_keys=False))
    (output_dir / "aprilgrid.yaml").write_text(yaml.safe_dump(aprilgrid_yaml, sort_keys=False))

    print(f"Wrote {output_dir / 'cameras.yaml'}")
    print(f"Wrote {output_dir / 'aprilgrid.yaml'}")


if __name__ == "__main__":
    main()