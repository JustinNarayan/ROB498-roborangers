#!/usr/bin/env python3
"""Run YOLOv5 ONNX inference on IMX219 camera and compute centroids."""

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


class YoloV5OnnxDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        class_names: Optional[Sequence[str]] = None,
        use_cuda: bool = True,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = list(class_names) if class_names is not None else None

        self.net = cv2.dnn.readNet(self.model_path)
        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except cv2.error:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _letterbox(image: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        return image, ratio, (dw, dh)

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        dwdh: tuple,
        original_shape: tuple,
    ) -> List[Detection]:
        predictions = output.squeeze(axis=0)
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            return []

        boxes = []
        confidences = []
        class_ids = []

        for row in predictions:
            objectness = row[4]
            if objectness < self.conf_threshold:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id] * objectness)
            if score < self.conf_threshold:
                continue

            x_c, y_c, w, h = row[:4]
            x1 = x_c - (w / 2.0)
            y1 = y_c - (h / 2.0)

            x1 = (x1 - dwdh[0]) / ratio
            y1 = (y1 - dwdh[1]) / ratio
            w = w / ratio
            h = h / ratio

            x1 = float(np.clip(x1, 0, original_shape[1] - 1))
            y1 = float(np.clip(y1, 0, original_shape[0] - 1))
            w = float(np.clip(w, 1, original_shape[1] - x1))
            h = float(np.clip(h, 1, original_shape[0] - y1))

            boxes.append([x1, y1, w, h])
            confidences.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        nms_idx = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        if len(nms_idx) == 0:
            return []

        detections = []
        for idx in np.array(nms_idx).flatten():
            x1, y1, w, h = boxes[idx]
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x1 + w,
                    y2=y1 + h,
                    confidence=float(confidences[idx]),
                    class_id=int(class_ids[idx]),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        original_shape = frame_bgr.shape[:2]
        image, ratio, dwdh = self._letterbox(frame_bgr, self.input_size)

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self._postprocess(outputs, ratio, dwdh, original_shape)


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
    parser = argparse.ArgumentParser(description="IMX219 YOLOv5 ONNX centroid detector")
    parser.add_argument("--model", required=True, help="Path to YOLOv5 ONNX model")
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
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    class_names = load_class_names(args.class_names)
    detector = YoloV5OnnxDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.input_size,
        class_names=class_names,
        use_cuda=not args.cpu,
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

        cv2.imshow("IMX219 YOLOv5", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
