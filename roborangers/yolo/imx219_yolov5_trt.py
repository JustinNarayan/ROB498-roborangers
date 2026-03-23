#!/usr/bin/env python3
"""Run YOLOv5 TensorRT inference on IMX219 camera and compute centroids.

Replaces cv2.dnn ONNX backend with Ultralytics TensorRT engine for maximum
GPU throughput on Jetson Nano (JP4.6.1, CUDA 10.2, TensorRT 8.0).

Export your model before running:
    yolo export model=yolov5n.pt format=onnx imgsz=640 opset=12
    /usr/src/tensorrt/bin/trtexec \
        --onnx=yolov5n.onnx \
        --saveEngine=yolov5n.engine \
        --fp16 \
        --workspace=512
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np


def gstreamer_pipeline(
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        "nvarguscamerasrc ! "
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


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def centroid(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YoloV5TRTDetector:
    """YOLOv5 detector backed by a TensorRT .engine via Ultralytics."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics not installed. Run: pip3 install ultralytics"
            ) from exc

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = list(class_names) if class_names is not None else None

        self.model = YOLO(model_path)  # accepts .engine, .onnx, or .pt
        # Warm up the engine — first inference is always slow due to TRT init
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        self.model.predict(source=dummy, imgsz=input_size, verbose=False)

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf.item()),
                    class_id=int(box.cls.item()),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    class_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    rendered = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cx, cy = map(int, det.centroid)

        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 230, 70), 2)
        cv2.circle(rendered, (cx, cy), 4, (0, 70, 255), -1)

        label = str(det.class_id)
        if class_names and 0 <= det.class_id < len(class_names):
            label = class_names[det.class_id]
        text = f"{label} {det.confidence:.2f}"
        cv2.putText(
            rendered,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 220, 220),
            2,
            cv2.LINE_AA,
        )

    return rendered


def load_class_names(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names if names else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMX219 YOLOv5 TensorRT centroid detector")
    parser.add_argument("--model", required=True, help="Path to .engine (or .onnx/.pt) model")
    parser.add_argument("--class-names", default="", help="Text file with class names")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    class_names = load_class_names(args.class_names)
    detector = YoloV5TRTDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.input_size,
        class_names=class_names,
    )

    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=args.camera_width,
            capture_height=args.camera_height,
            display_width=args.display_width,
            display_height=args.display_height,
            framerate=args.fps,
            flip_method=args.flip_method,
        ),
        cv2.CAP_GSTREAMER,
    )

    if not cap.isOpened():
        raise RuntimeError("Failed to open IMX219 camera via GStreamer")

    prev_ts = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            detections = detector.infer(frame)
            vis = draw_detections(frame, detections, class_names=class_names)

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_ts)
            prev_ts = now
            cv2.putText(
                vis,
                f"FPS {fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 180, 50),
                2,
                cv2.LINE_AA,
            )

            if detections:
                cx, cy = detections[0].centroid
                print(
                    f"det={len(detections)} top_cls={detections[0].class_id} "
                    f"conf={detections[0].confidence:.3f} cx={cx:.1f} cy={cy:.1f}"
                )

            cv2.imshow("IMX219 YOLOv5 TRT", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()