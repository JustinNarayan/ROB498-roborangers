import cv2
import subprocess
import datetime
import os
import signal

SAVE_PATH = "/home/jetson/ros2_ws/src/ROB498/roborangers/ROB498-roborangers/roborangers/opencv/videos"

WIDTH = 1280
HEIGHT = 720
FPS = 30

recording_process = None

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def preview_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! appsink"
        % (WIDTH, HEIGHT, FPS)
    )


def start_recording():
    global recording_process

    if recording_process is not None:
        print("Already recording")
        return

    filename = datetime.datetime.now().strftime("video_%Y%m%d_%H%M%S.mp4")
    filepath = os.path.join(SAVE_PATH, filename)

    print("Recording to:", filepath)

    gst_cmd = (
        "gst-launch-1.0 -e nvarguscamerasrc ! "
        "'video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1' ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=4000000 ! "
        "h264parse ! "
        "qtmux ! "
        "filesink location=%s"
        % (WIDTH, HEIGHT, FPS, filepath)
    )

    recording_process = subprocess.Popen(
        gst_cmd,
        shell=True,
        executable="/bin/bash"
    )


def stop_recording():
    global recording_process

    if recording_process is None:
        print("Not recording")
        return

    recording_process.send_signal(signal.SIGINT)
    recording_process.wait()
    recording_process = None

    print("Recording stopped")


print("\nControls")
print("s = start recording")
print("e = stop recording")
print("q = quit\n")

cap = cv2.VideoCapture(preview_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("IMX Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        start_recording()

    elif key == ord('e'):
        stop_recording()

    elif key == ord('q'):
        stop_recording()
        break

cap.release()
cv2.destroyAllWindows()