#!/usr/bin/env python3
"""YOLOv5 real-time inference visualizer — optimised for Jetson Nano.

Designed to simulate real-time drone footage inference with minimal memory
footprint. No video saving — display only.

Jetson Nano tips:
  - Use yolov5n.pt (nano model, ~4 MB) instead of yolov5su.pt (~28 MB)
  - half=True halves GPU memory usage via FP16 (requires CUDA)
  - Resize frames before inference to avoid internal allocation spikes
  - Delete result objects immediately to release GPU memory each loop
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv5 visualizer optimised for Jetson Nano."
    )
    parser.add_argument(
        "--video",
        default="",
        help="Path to input video. If omitted, first .mp4 in ./videos/ is used.",
    )
    parser.add_argument(
        "--model",
        default="yolov5n.pt",  # Nano model — use this on Jetson, not yolov5su
        help="Ultralytics YOLO model name/path. Recommended: yolov5n.pt for Jetson Nano.",
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
        help="Inference image size (default: 320). Lower = faster + less memory.",
    )
    parser.add_argument(
        "--car-class-id",
        type=int,
        default=2,
        help="COCO class id to track (default: 2 = car).",
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
        help="Run inference every N frames (default: 2). Raise to 3-4 if still slow.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=15.0,
        help="Cap the display loop to this FPS (default: 15). Prevents thermal throttle.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 half-precision inference (requires CUDA). Halves GPU memory.",
    )
    parser.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Disable FP16 (use if you see inference errors on your CUDA build).",
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
            "ultralytics not installed. Run: pip install ultralytics"
        ) from exc

    script_path = Path(__file__).resolve()
    video_path = (
        Path(args.video).expanduser().resolve()
        if args.video
        else resolve_default_video(script_path)
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Loading model: {args.model}  (half={args.half})")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    infer_size = (args.imgsz, args.imgsz)
    window_name = "YOLOv5 | q to quit"
    frame_idx = 0
    cached_boxes: list = []  # Detections reused across skipped frames

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
                # Pre-resize before inference — avoids a large internal allocation
                small = cv2.resize(frame, infer_size)

                result = model.predict(
                    source=small,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    half=args.half,
                    verbose=False,
                )[0]

                # Scale factors back to original frame resolution
                fh, fw = frame.shape[:2]
                sx = fw / args.imgsz
                sy = fh / args.imgsz

                cached_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        if not args.all_classes and cls_id != args.car_class_id:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        # Scale coords back to display resolution
                        x1, x2 = int(x1 * sx), int(x2 * sx)
                        y1, y2 = int(y1 * sy), int(y2 * sy)
                        cached_boxes.append((
                            x1, y1, x2, y2,
                            float(box.conf.item()),
                            result.names.get(cls_id, str(cls_id)),
                        ))

                # Explicitly free result to release GPU-side tensors immediately
                del result
                gc.collect()

            # Draw cached boxes on the full-resolution display frame
            for x1, y1, x2, y2, conf_val, label in cached_boxes:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 90, 255), -1)
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

            # FPS overlay — useful for tuning skip-frames / imgsz on the Nano
            elapsed = time.monotonic() - loop_start
            display_fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # Sleep to respect max-fps cap and give thermals a break
            if min_frame_time > 0:
                sleep_needed = min_frame_time - (time.monotonic() - loop_start)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    run()

# yolov5n.pt  →  ONNX  →  TensorRT .engine
# (PyTorch)      (intermediate)   (compiled for YOUR specific GPU)
 
# python detect_jetson.py --video my_drone_footage.mp4 --skip-frames 3 --max-fps 12

# Export to tensorrt
# Run this once on the Jetson
# yolo export model=yolov5n.pt format=engine imgsz=320 half=True device=0  # one-time export
# python detect_jetson.py --model yolov5n.engine --imgsz 320 --skip-frames 2 --max-fps 20