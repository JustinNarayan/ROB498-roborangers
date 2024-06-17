# MIT License
# Modified to support start/stop video recording
# s + Enter -> start recording
# e + Enter -> end recording

import cv2
import time
import threading
import sys
import signal
from datetime import datetime

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue


SAVE_PATH = "/home/jetson/ros2_ws/src/ROB498/roborangers/ROB498-roborangers/roborangers/opencv/videos"   # <-- change this to your desired path


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None

    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera

    def run(self):
        while self._running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            while self.queues:
                queue = self.queues.pop()
                queue.put(frame)

    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout=None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout=timeout)

    def stop(self):
        self._running = False


class Previewer(threading.Thread):
    window_name = "Arducam"
    _running = True
    camera = None

    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera

    def run(self):
        self._running = True
        while self._running:
            frame = self.camera.getFrame(2000)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

        cv2.destroyWindow(self.window_name)

    def start_preview(self):
        self.start()

    def stop_preview(self):
        self._running = False


class Recorder(threading.Thread):
    def __init__(self, camera):
        threading.Thread.__init__(self)
        self.camera = camera
        self.recording = False
        self.writer = None
        self.running = True

    def start_recording(self):
        if self.recording:
            return

        filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.mp4")
        filepath = SAVE_PATH + filename

        print("Recording started:", filepath)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, 60, (640, 360))
        self.recording = True

    def stop_recording(self):
        if not self.recording:
            return

        print("Recording stopped")
        self.recording = False
        self.writer.release()
        self.writer = None

    def run(self):
        while self.running:
            frame = self.camera.getFrame()

            if self.recording and self.writer is not None:
                self.writer.write(frame)

    def stop(self):
        self.running = False
        if self.writer:
            self.writer.release()


class Camera(object):

    def __init__(self):
        self.open_camera()

    def open_camera(self):

        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(flip_method=0),
            cv2.CAP_GSTREAMER
        )

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")

        self.frame_reader = FrameReader(self.cap, "")
        self.frame_reader.daemon = True
        self.frame_reader.start()

        self.previewer = Previewer(self.frame_reader, "")
        self.recorder = Recorder(self.frame_reader)

    def getFrame(self):
        return self.frame_reader.getFrame()

    def start_preview(self):
        self.previewer.daemon = True
        self.previewer.start_preview()

    def stop_preview(self):
        self.previewer.stop_preview()
        self.previewer.join()

    def start_recorder(self):
        self.recorder.daemon = True
        self.recorder.start()

    def close(self):
        self.frame_reader.stop()
        self.recorder.stop()
        self.cap.release()


def keyboard_listener(camera):
    print("\nControls:")
    print("s + Enter -> start recording")
    print("e + Enter -> end recording")
    print("q + Enter -> quit\n")

    while True:
        key = input().strip()

        if key == "space":
            camera.recorder.start_recording()

        elif key == "e":
            camera.recorder.stop_recording()

        elif key == "q":
            break


if __name__ == "__main__":

    camera = Camera()

    camera.start_preview()
    camera.start_recorder()

    try:
        keyboard_listener(camera)

    finally:
        camera.stop_preview()
        camera.close()