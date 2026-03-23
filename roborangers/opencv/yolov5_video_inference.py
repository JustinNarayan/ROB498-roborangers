#!/usr/bin/env python3
"""YOLOv5 inference on a local video with bbox + centroid overlay.

Jetson Nano-friendly: frame skipping, reduced input size, headless-safe display,
and graceful teardown to avoid GPU freeze.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLOv5 on a video and draw bbox + centroid."
    )
    parser.add_argument(
        "--video",
        default="",
        help="Path to input video. If omitted, first .mp4 in opencv/videos is used.",
    )
    parser.add_argument(
        "--model",
        default="yolov5su.pt",
        help="Pretrained model name/path for Ultralytics YOLO (default: yolov5su.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,  # Reduced from 640 — critical for Jetson Nano
        help="Inference image size (square). Use 320 or 416 on Jetson Nano.",
    )
    parser.add_argument(
        "--car-class-id",
        type=int,
        default=2,
        help="COCO class id for car (default: 2)",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Draw all detected classes instead of only class id in --car-class-id.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output video path with overlays.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Run inference every N frames (default: 2). Set higher on slow hardware.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow (use when headless or over SSH).",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=0.0,
        help="Cap processing loop to this FPS (0 = uncapped). Helps prevent thermal throttle.",
    )
    return parser.parse_args()


def resolve_default_video(script_path: Path) -> Path:
    videos_dir = script_path.parent / "videos"
    candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")
    return candidates[0]


def run() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'ultralytics'. Install it with: pip install ultralytics"
        ) from exc

    script_path = Path(__file__).resolve()
    video_path = (
        Path(args.video).expanduser().resolve()
        if args.video
        else resolve_default_video(script_path)
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load model once; keep off GPU until first predict call
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer: Optional[cv2.VideoWriter] = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(Path(args.save).expanduser().resolve()),
            fourcc,
            fps,
            (width, height),
        )

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    window_name = "YOLOv5 detection (press q to quit)"
    frame_idx = 0
    last_result_boxes: list = []  # Reuse last detections on skipped frames

    try:
        while True:
            loop_start = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                break

            # --- Frame skipping: only run inference every N frames ---
            run_inference = (frame_idx % max(1, args.skip_frames) == 0)
            frame_idx += 1

            if run_inference:
                result = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]

                last_result_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        if not args.all_classes and cls_id != args.car_class_id:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf_val = float(box.conf.item())
                        label = result.names.get(cls_id, str(cls_id))
                        last_result_boxes.append((x1, y1, x2, y2, conf_val, label))

            # --- Draw cached boxes on every frame ---
            for x1, y1, x2, y2, conf_val, label in last_result_boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 90, 255), -1)
                cv2.putText(
                    frame,
                    f"{label} {conf_val:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)

            # --- Display (skip entirely if headless) ---
            if not args.no_display:
                cv2.imshow(window_name, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            # --- FPS cap: sleep to avoid melting the Nano ---
            if min_frame_time > 0:
                elapsed = time.monotonic() - loop_start
                sleep_time = min_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        # Always release resources — avoids GPU/driver hang on Jetson
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

# python detect_jetson.py --imgsz 320 --skip-frames 3 --max-fps 10 --no-display --save out.mp4