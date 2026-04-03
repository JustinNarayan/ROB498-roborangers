#!/usr/bin/env python3
"""ROS2 node for IMX219 YOLOv8 TensorRT detection and centroid publishing.

This node expects a TensorRT engine produced on the Jetson Nano from an ONNX
export of yolov8n.pt. Inference runs through TensorRT and PyCUDA only.
"""

from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from roborangers.cv.yolo.imx219_yolov8n_trt import (
    camera_backend_diagnostic,
    opencv_has_gstreamer,
    YoloV8TRTDetector,
    draw_detections,
    filter_detections_by_class,
    gstreamer_pipeline,
    load_class_names,
    resolve_target_class_id,
)


class CentroidDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("centroid_detector_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("class_names_path", "")
        self.declare_parameter("target_class_name", "")
        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("input_size", 640)
        # Camera defaults updated to match training data capture:
        #   sensor-mode=3 => 1640x1232 @ 30fps, flip_method=2 (180deg rotation)
        self.declare_parameter("camera_width", 1640)
        self.declare_parameter("camera_height", 1232)
        self.declare_parameter("display_width", 1640)
        self.declare_parameter("display_height", 1232)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("flip_method", 2)
        self.declare_parameter("sensor_mode", 3)
        self.declare_parameter("sensor_id", 0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("display", False)

        model_path = str(self.get_parameter("model_path").value)
        if not model_path:
            raise ValueError("model_path parameter must be set")

        class_names_path = str(self.get_parameter("class_names_path").value)
        self.class_names = load_class_names(class_names_path)
        self.target_class_name = str(self.get_parameter("target_class_name").value).strip()
        self.target_class_id_param = int(self.get_parameter("target_class_id").value)
        self.target_class_id = self._resolve_target_class_id()

        if not opencv_has_gstreamer():
            raise RuntimeError(camera_backend_diagnostic())

        conf_th = float(self.get_parameter("confidence_threshold").value)
        iou_th = float(self.get_parameter("iou_threshold").value)
        input_size = int(self.get_parameter("input_size").value)

        self.get_logger().info(f"Loading TensorRT model: {model_path}")
        self.detector = YoloV8TRTDetector(
            model_path=model_path,
            conf_threshold=conf_th,
            iou_threshold=iou_th,
            input_size=input_size,
            class_names=self.class_names,
        )
        self.get_logger().info("TensorRT engine loaded and warmed up")

        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.display_width = int(self.get_parameter("display_width").value)
        self.display_height = int(self.get_parameter("display_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.flip_method = int(self.get_parameter("flip_method").value)
        self.sensor_mode = int(self.get_parameter("sensor_mode").value)
        self.sensor_id = int(self.get_parameter("sensor_id").value)
        self.display = bool(self.get_parameter("display").value)

        self.centroid_pub = self.create_publisher(
            Float32MultiArray, "/vision/target_centroid", 10
        )
        self.bbox_pub = self.create_publisher(Float32MultiArray, "/vision/target_bbox", 10)
        self.status_pub = self.create_publisher(String, "/vision/target_status", 10)

        pipeline = gstreamer_pipeline(
            capture_width=self.camera_width,
            capture_height=self.camera_height,
            display_width=self.display_width,
            display_height=self.display_height,
            framerate=self.camera_fps,
            flip_method=self.flip_method,
            sensor_mode=self.sensor_mode,
            sensor_id = self.sensor_id,
        )
        self.get_logger().info(f"Opening camera pipeline: {pipeline}")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open IMX219 camera pipeline\nPipeline: {pipeline}")

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        period_sec = 1.0 / max(1.0, publish_rate_hz)

        self._frame_count = 0
        self._detect_count = 0
        self._lost_frames = 0
        self._log_interval = max(1, int(publish_rate_hz))  # log every ~1 second

        self.timer = self.create_timer(period_sec, self._on_timer)
        self.get_logger().info("Centroid detector node started (TensorRT backend)")

    def _resolve_target_class_id(self) -> Optional[int]:
        explicit_target_class_id = (
            self.target_class_id_param if self.target_class_id_param >= 0 else None
        )
        target_class_id = resolve_target_class_id(
            self.class_names,
            target_class_name=self.target_class_name,
            target_class_id=explicit_target_class_id,
        )

        if target_class_id is not None:
            target_label = str(target_class_id)
            if self.class_names and 0 <= target_class_id < len(self.class_names):
                target_label = self.class_names[target_class_id]
            self.get_logger().info(
                f"Tracking target class {target_class_id} ({target_label})"
            )
            return target_class_id

        if self.target_class_name and self.class_names:
            self.get_logger().warn(
                f"target_class_name '{self.target_class_name}' not found in classes; "
                "using top detection"
            )
        elif self.target_class_name:
            self.get_logger().warn(
                f"target_class_name '{self.target_class_name}' was set but class_names_path "
                "is empty; using top detection. For a single-class custom model, use "
                "'-p target_class_id:=0' instead of target_class_name, or create a "
                "class_names.txt file (one class name per line) and pass it via "
                "'-p class_names_path:=/path/to/class_names.txt'"
            )
        else:
            self.get_logger().info("No target class configured; using top detection")
        return None

    def _select_target(self, detections):
        if not detections:
            return None
        if self.target_class_id is None:
            return detections[0]
        filtered = filter_detections_by_class(detections, self.target_class_id)
        return filtered[0] if filtered else None

    def _publish_lost(self) -> None:
        status = String()
        status.data = "LOST"
        self.status_pub.publish(status)

    def _on_timer(self) -> None:
        self._frame_count += 1
        should_log = (self._frame_count % self._log_interval == 0)

        ok, frame = self.cap.read()
        if not ok:
            self._lost_frames += 1
            self._publish_lost()
            if should_log:
                self.get_logger().warn(
                    f"Camera read failed ({self._lost_frames}/{self._frame_count} frames lost)"
                )
            return

        detections = self.detector.infer(frame)
        target = self._select_target(detections)

        if target is None:
            self._publish_lost()
            if should_log:
                self.get_logger().info(
                    f"No target detected (frame {self._frame_count}, "
                    f"{len(detections)} raw detections)"
                )
            if self.display:
                vis = draw_detections(frame, detections, self.class_names)
                cv2.imshow("centroid_detector", vis)
                cv2.waitKey(1)
            return

        self._detect_count += 1
        h, w = frame.shape[:2]
        cx, cy = target.centroid
        nx = (cx - (w / 2.0)) / (w / 2.0)
        ny = (cy - (h / 2.0)) / (h / 2.0)

        centroid_msg = Float32MultiArray()
        centroid_msg.data = [
            float(cx), float(cy),
            float(nx), float(ny),
            float(target.confidence),
            float(target.class_id),
        ]
        self.centroid_pub.publish(centroid_msg)

        bbox_msg = Float32MultiArray()
        bbox_msg.data = [
            float(target.x1), float(target.y1),
            float(target.x2), float(target.y2),
            float(target.confidence),
            float(target.class_id),
        ]
        self.bbox_pub.publish(bbox_msg)

        status = String()
        status.data = "DETECTED"
        self.status_pub.publish(status)

        if should_log:
            self.get_logger().info(
                f"DETECTED cx={cx:.1f} cy={cy:.1f} nx={nx:.3f} ny={ny:.3f} "
                f"conf={target.confidence:.3f} class={target.class_id} "
                f"({self._detect_count}/{self._frame_count} frames with target)"
            )

        if self.display:
            vis = draw_detections(frame, detections, self.class_names)
            cv2.imshow("centroid_detector", vis)
            cv2.waitKey(1)

    def destroy_node(self):
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CentroidDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()