#!/usr/bin/env python3
"""YOLOv8 TensorRT video inference visualizer for Jetson Nano.

Expected deployment flow:
  1. Export yolov8n.pt to ONNX on a desktop or laptop.
  2. Copy the ONNX to the Jetson Nano.
  3. Build a TensorRT engine on the Nano with trtexec.
  4. Run this script against the .engine file.

This script uses TensorRT and PyCUDA only for inference on the Nano.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo.imx219_yolov5_trt import YoloV8TRTDetector, draw_detections, load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8 TensorRT video visualizer optimized for Jetson Nano."
    )
    parser.add_argument(
        "--video",
        default="",
        help="Path to input video. If omitted, first .mp4 in ./videos/ is used.",
    )
    parser.add_argument(
        "--model",
        default="../yolov8n.engine",
        help="Path to TensorRT engine file built on this Nano.",
    )
    parser.add_argument(
        "--class-names",
        default="",
        help="Optional text file with one class name per line.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size (default: 320). Lower uses less memory.",
    )
    parser.add_argument(
        "--car-class-id",
        type=int,
        default=2,
        help="Class id to track when --all-classes is not set (default: 2).",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Show all detected classes instead of only --car-class-id.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Run inference every N frames (default: 2).",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=15.0,
        help="Cap the display loop to this FPS (default: 15).",
    )
    return parser.parse_args()


def resolve_default_video(script_path: Path) -> Path:
    videos_dir = script_path.parent / "videos"
    candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")
    return candidates[0]


def resolve_path(path_str: str, script_path: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (script_path.parent / path).resolve()


def filter_detections(detections, show_all: bool, class_id: int):
    if show_all:
        return detections
    return [det for det in detections if det.class_id == class_id]


def run() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()

    video_path = (
        resolve_path(args.video, script_path) if args.video else resolve_default_video(script_path)
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_path = resolve_path(args.model, script_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {model_path}")

    class_names = load_class_names(args.class_names) if args.class_names else None

    print(f"Loading TensorRT model: {model_path}")
    detector = YoloV8TRTDetector(
        model_path=str(model_path),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
        class_names=class_names,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    window_name = "YOLOv8 TensorRT | q to quit"
    frame_idx = 0
    cached_detections = []

    # FIX: track FPS only on inference frames for accurate measurement
    inference_fps = 0.0
    inference_start = time.monotonic()

    print("Starting display loop. Press 'q' to quit.")

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                break

            run_inference = (frame_idx % max(1, args.skip_frames) == 0)
            frame_idx += 1

            if run_inference:
                # FIX: measure elapsed time between inference frames only
                elapsed = time.monotonic() - inference_start
                inference_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                inference_start = time.monotonic()

                detections = detector.infer(frame)
                cached_detections = filter_detections(
                    detections,
                    show_all=args.all_classes,
                    class_id=args.car_class_id,
                )

            rendered = draw_detections(frame, cached_detections, class_names=class_names)

            # FIX: display inference FPS (not loop FPS) — accurate and meaningful
            cv2.putText(
                rendered,
                f"Inference FPS: {inference_fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, rendered)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            if min_frame_time > 0:
                sleep_needed = min_frame_time - (time.monotonic() - loop_start)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Done.")


if __name__ == "__main__":
    run()