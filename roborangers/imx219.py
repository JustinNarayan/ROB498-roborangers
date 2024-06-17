#!/usr/bin/env python3
"""
IMX219 Camera CV Pipeline for Jetson
-------------------------------------
Captures frames from an IMX219 CSI camera using GStreamer + OpenCV,
and provides a simple loop for plugging in CV algorithms (e.g., tracking).

Usage:
    python3 imx219_cv.py
    Press 'q' to quit.
"""

import cv2
import numpy as np
import time


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,  # 0=none, 2=rotate-180 (use if camera is upside-down on drone)
):
    """
    Build a GStreamer pipeline string for the IMX219 on Jetson.
    Uses nvarguscamerasrc (hardware-accelerated NVIDIA ISP).
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def process_frame(frame):
    """
    Placeholder CV processing function.
    Replace this with your tracking / detection logic later.

    For now, it just converts to grayscale and runs Canny edge detection
    as a sanity check that CV is working.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Stack original + edges side by side for visualization
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edges_bgr))
    return combined


def main():
    # --- Configuration ---
    SENSOR_ID = 0          # CSI camera index (0 for single camera)
    CAPTURE_W = 1280
    CAPTURE_H = 720
    DISPLAY_W = 640        # Smaller display for faster rendering
    DISPLAY_H = 360
    FPS = 30
    FLIP = 0               # Set to 2 if your camera is mounted upside-down

    pipeline = gstreamer_pipeline(
        sensor_id=SENSOR_ID,
        capture_width=CAPTURE_W,
        capture_height=CAPTURE_H,
        display_width=DISPLAY_W,
        display_height=DISPLAY_H,
        framerate=FPS,
        flip_method=FLIP,
    )
    print(f"GStreamer pipeline:\n  {pipeline}\n")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: Could not open IMX219 camera.")
        print("Troubleshooting:")
        print("  1. Is the ribbon cable seated properly on both ends?")
        print("  2. Run: nvgstcapture-1.0   (to test outside Python)")
        print("  3. Check: ls /dev/video*    (CSI cameras may not show here)")
        print("  4. Ensure jetson-io has configured the IMX219 overlay.")
        return

    print("Camera opened successfully. Press 'q' to quit.\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame — retrying...")
                continue

            frame_count += 1

            # ---- YOUR CV GOES HERE ----
            output = process_frame(frame)
            # ---------------------------

            # Show FPS overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                output, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            cv2.imshow("IMX219 CV Pipeline", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()
