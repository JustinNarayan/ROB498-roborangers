#!/usr/bin/env python3
"""Run YOLOv8 TensorRT inference on IMX219 camera and compute centroids.

Target deployment flow for Jetson Nano:
    1. Export yolov8n.pt to ONNX on a desktop or laptop.
    2. Copy the ONNX file to the Nano.
    3. Build a TensorRT engine on the Nano with trtexec.
    4. Run this script against the .engine file.

Example:
    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
    /usr/src/tensorrt/bin/trtexec \
        --onnx=yolov8n.onnx \
        --saveEngine=yolov8n.engine \
        --fp16 \
        --workspace=1024
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import pycuda.driver as cuda
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pycuda is required on the Jetson Nano. Install it before running this detector."
    ) from exc

try:
    import tensorrt as trt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tensorrt Python bindings are required on the Jetson Nano."
    ) from exc


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


class YoloV8TRTDetector:
    """YOLOv8 detector backed directly by TensorRT and PyCUDA."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        class_names: Optional[Sequence[str]] = None,
        cuda_device: int = 0,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = list(class_names) if class_names is not None else None
        self.engine = None
        self.context = None
        self.cuda_context = None
        self.stream = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.input_binding_index = -1
        self.output_binding_index = -1
        self.bindings = []
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}

        cuda.init()
        self.cuda_context = cuda.Device(cuda_device).make_context()

        try:
            self.stream = cuda.Stream()
            with open(model_path, "rb") as engine_file, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            self.bindings = [0] * self.engine.num_bindings
            self._configure_bindings()

            # Warm-up pass — takes 10-15s on first run on Jetson Nano, this is normal
            print("Running warm-up inference (this takes ~15s on Jetson Nano, please wait)...")
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.infer(dummy)
            print("Warm-up complete.")
        except Exception:
            self.close()
            raise
        finally:
            if self.cuda_context is not None:
                self.cuda_context.pop()

    def _configure_bindings(self) -> None:
        for binding_index in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding_index):
                self.input_binding_index = binding_index
                break

        if self.input_binding_index < 0:
            raise RuntimeError("TensorRT engine has no input binding")

        input_shape = tuple(self.engine.get_binding_shape(self.input_binding_index))
        if -1 in input_shape:
            self.context.set_binding_shape(
                self.input_binding_index,
                (1, 3, self.input_size, self.input_size),
            )
            input_shape = tuple(self.context.get_binding_shape(self.input_binding_index))
        else:
            if len(input_shape) != 4:
                raise RuntimeError(f"Unexpected input binding shape: {input_shape}")
            self.input_size = int(input_shape[2])

        self._allocate_binding(self.input_binding_index, input_shape)

        output_indices = [
            idx for idx in range(self.engine.num_bindings) if not self.engine.binding_is_input(idx)
        ]
        if not output_indices:
            raise RuntimeError("TensorRT engine has no output bindings")

        self.output_binding_index = output_indices[0]
        output_shape = tuple(self.context.get_binding_shape(self.output_binding_index))
        self._allocate_binding(self.output_binding_index, output_shape)

    def _allocate_binding(self, binding_index: int, shape: Tuple[int, ...]) -> None:
        dtype = trt.nptype(self.engine.get_binding_dtype(binding_index))
        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.bindings[binding_index] = int(device_mem)

        if self.engine.binding_is_input(binding_index):
            self.host_inputs[binding_index] = host_mem
            self.device_inputs[binding_index] = device_mem
        else:
            self.host_outputs[binding_index] = host_mem
            self.device_outputs[binding_index] = device_mem

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0

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

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        image, ratio, dwdh = self._letterbox(frame_bgr, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(np.ascontiguousarray(image), axis=0)
        return image, ratio, dwdh

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        dwdh: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> List[Detection]:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            return []

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            return []

        boxes = []
        confidences = []
        class_ids = []

        for row in predictions:
            x_c, y_c, w, h = row[:4]
            class_scores = row[4:]
            if class_scores.size == 0:
                continue

            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.conf_threshold:
                continue

            x1 = (x_c - (w / 2.0) - dwdh[0]) / ratio
            y1 = (y_c - (h / 2.0) - dwdh[1]) / ratio
            x2 = (x_c + (w / 2.0) - dwdh[0]) / ratio
            y2 = (y_c + (h / 2.0) - dwdh[1]) / ratio

            x1 = float(np.clip(x1, 0, original_shape[1] - 1))
            y1 = float(np.clip(y1, 0, original_shape[0] - 1))
            x2 = float(np.clip(x2, x1 + 1, original_shape[1]))
            y2 = float(np.clip(y2, y1 + 1, original_shape[0]))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )
        if len(nms_indices) == 0:
            return []

        detections = []
        for idx in np.array(nms_indices).flatten():
            x1, y1, w, h = boxes[idx]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x1 + w),
                    y2=float(y1 + h),
                    confidence=float(confidences[idx]),
                    class_id=int(class_ids[idx]),
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        original_shape = frame_bgr.shape[:2]
        input_tensor, ratio, dwdh = self._preprocess(frame_bgr)

        self.cuda_context.push()
        try:
            host_input = self.host_inputs[self.input_binding_index]
            np.copyto(host_input, input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.device_inputs[self.input_binding_index],
                host_input,
                self.stream,
            )

            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            host_output = self.host_outputs[self.output_binding_index]
            cuda.memcpy_dtoh_async(
                host_output,
                self.device_outputs[self.output_binding_index],
                self.stream,
            )
            self.stream.synchronize()

            output_shape = tuple(self.context.get_binding_shape(self.output_binding_index))
            output = np.array(host_output).reshape(output_shape)
            return self._postprocess(output, ratio, dwdh, original_shape)
        finally:
            self.cuda_context.pop()

    def close(self) -> None:
        # FIX: push context before freeing GPU memory to prevent shutdown freeze
        if self.cuda_context is not None:
            try:
                self.cuda_context.push()
            except Exception:
                pass

        if hasattr(self, "device_inputs"):
            for device_mem in self.device_inputs.values():
                try:
                    device_mem.free()
                except Exception:
                    pass
            self.device_inputs = {}

        if hasattr(self, "device_outputs"):
            for device_mem in self.device_outputs.values():
                try:
                    device_mem.free()
                except Exception:
                    pass
            self.device_outputs = {}

        self.host_inputs = {}
        self.host_outputs = {}
        self.bindings = []
        self.stream = None
        self.context = None
        self.engine = None

        if self.cuda_context is not None:
            try:
                self.cuda_context.pop()   # pop before detach
                self.cuda_context.detach()
            except Exception:
                pass
            self.cuda_context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


YoloV5TRTDetector = YoloV8TRTDetector


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
    with open(path, "r", encoding="utf-8") as class_file:
        names = [line.strip() for line in class_file.readlines() if line.strip()]
    return names if names else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMX219 YOLOv8 TensorRT centroid detector")
    parser.add_argument("--model", required=True, help="Path to a TensorRT .engine built on this Nano")
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
    detector = YoloV8TRTDetector(
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
        detector.close()
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

            cv2.imshow("IMX219 YOLOv8 TRT", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()