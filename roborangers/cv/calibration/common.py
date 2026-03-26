#!/usr/bin/env python3
"""Shared helpers for calibration publishers and YAML export tools."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from sensor_msgs.msg import CameraInfo, Image


def gstreamer_pipeline(
    capture_width: int = 1640,
    capture_height: int = 1232,
    display_width: int = 1640,
    display_height: int = 1232,
    framerate: int = 30,
    flip_method: int = 2,
    sensor_mode: int = 3,
    sensor_id: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, "
        f"width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false"
    )


def opencv_has_gstreamer(cv2_module) -> bool:
    build_info = cv2_module.getBuildInformation()
    return "GStreamer:                   YES" in build_info or "GStreamer: YES" in build_info


def camera_backend_diagnostic(cv2_module) -> str:
    if opencv_has_gstreamer(cv2_module):
        return "OpenCV reports GStreamer support, but the IMX219 pipeline still failed to open."

    return (
        "OpenCV was built without GStreamer support, so cv2.CAP_GSTREAMER cannot open "
        "nvarguscamerasrc. Use the JetPack OpenCV build or rebuild OpenCV with GStreamer."
    )


def numpy_to_image_msg(image, stamp, frame_id: str, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    msg.step = int(image.shape[1] * channels * image.dtype.itemsize)
    msg.data = image.tobytes()
    return msg


def make_default_camera_info(
    *,
    width: int,
    height: int,
    frame_id: str,
    camera_name: str,
    distortion_model: str,
) -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = distortion_model
    msg.k = [0.0] * 9
    msg.r = [0.0] * 9
    msg.p = [0.0] * 12
    msg.k[0] = 1.0
    msg.k[4] = 1.0
    msg.k[8] = 1.0
    msg.r[0] = 1.0
    msg.r[4] = 1.0
    msg.r[8] = 1.0
    msg.p[0] = 1.0
    msg.p[5] = 1.0
    msg.p[10] = 1.0
    msg.d = []
    msg.binning_x = 0
    msg.binning_y = 0
    msg.roi.do_rectify = False
    msg.roi.x_offset = 0
    msg.roi.y_offset = 0
    msg.roi.height = 0
    msg.roi.width = 0
    msg.header.frame_id = frame_id
    return msg


def _coerce_matrix(values: Sequence[float], expected_len: int, default: Sequence[float]) -> list[float]:
    if len(values) == expected_len:
        return [float(value) for value in values]
    return [float(value) for value in default]


def load_camera_info_yaml(
    yaml_path: str | Path,
    *,
    frame_id: str,
    camera_name: str,
    default_width: int,
    default_height: int,
    default_distortion_model: str,
) -> CameraInfo:
    path = Path(yaml_path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())

    width = int(data.get("image_width", default_width))
    height = int(data.get("image_height", default_height))
    distortion_model = data.get("distortion_model", default_distortion_model)

    msg = make_default_camera_info(
        width=width,
        height=height,
        frame_id=frame_id,
        camera_name=camera_name,
        distortion_model=distortion_model,
    )

    msg.k = _coerce_matrix(
        data.get("camera_matrix", {}).get("data", []),
        9,
        msg.k,
    )
    msg.d = [float(value) for value in data.get("distortion_coefficients", {}).get("data", [])]
    msg.r = _coerce_matrix(
        data.get("rectification_matrix", {}).get("data", []),
        9,
        msg.r,
    )
    msg.p = _coerce_matrix(
        data.get("projection_matrix", {}).get("data", []),
        12,
        msg.p,
    )
    return msg


def clone_camera_info(camera_info: CameraInfo, stamp, frame_id: str, width: int, height: int) -> CameraInfo:
    msg = deepcopy(camera_info)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    return msg


def intrinsics_to_camera_info(
    intrinsics,
    *,
    frame_id: str,
    distortion_model: str,
) -> CameraInfo:
    msg = make_default_camera_info(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        frame_id=frame_id,
        camera_name=frame_id,
        distortion_model=distortion_model,
    )
    msg.k = [
        float(intrinsics.fx), 0.0, float(intrinsics.ppx),
        0.0, float(intrinsics.fy), float(intrinsics.ppy),
        0.0, 0.0, 1.0,
    ]
    msg.p = [
        float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0,
        0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    msg.d = [float(value) for value in intrinsics.coeffs]
    return msg


def camera_info_to_yaml_dict(camera_info: CameraInfo, camera_name: str) -> dict:
    return {
        "image_width": int(camera_info.width),
        "image_height": int(camera_info.height),
        "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(value) for value in camera_info.k]},
        "distortion_model": camera_info.distortion_model,
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(camera_info.d),
            "data": [float(value) for value in camera_info.d],
        },
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [float(value) for value in camera_info.r]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(value) for value in camera_info.p]},
    }


def write_yaml(path: str | Path, data: dict) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False))


def flatten_matrix_rows(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in rows]