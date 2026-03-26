#!/usr/bin/env python3
"""Compatibility alias for the YOLOv8n TensorRT IMX219 runner."""

try:
    from roborangers.yolo.imx219_yolov8_trt import camera_backend_diagnostic
    from roborangers.yolo.imx219_yolov8_trt import Detection
    from roborangers.yolo.imx219_yolov8_trt import opencv_has_gstreamer
    from roborangers.yolo.imx219_yolov8_trt import YoloV5TRTDetector
    from roborangers.yolo.imx219_yolov8_trt import YoloV8TRTDetector
    from roborangers.yolo.imx219_yolov8_trt import draw_detections
    from roborangers.yolo.imx219_yolov8_trt import filter_detections_by_class
    from roborangers.yolo.imx219_yolov8_trt import gstreamer_pipeline
    from roborangers.yolo.imx219_yolov8_trt import load_class_names
    from roborangers.yolo.imx219_yolov8_trt import main
    from roborangers.yolo.imx219_yolov8_trt import resolve_target_class_id
except ModuleNotFoundError:
    from imx219_yolov8_trt import camera_backend_diagnostic
    from imx219_yolov8_trt import Detection
    from imx219_yolov8_trt import opencv_has_gstreamer
    from imx219_yolov8_trt import YoloV5TRTDetector
    from imx219_yolov8_trt import YoloV8TRTDetector
    from imx219_yolov8_trt import draw_detections
    from imx219_yolov8_trt import filter_detections_by_class
    from imx219_yolov8_trt import gstreamer_pipeline
    from imx219_yolov8_trt import load_class_names
    from imx219_yolov8_trt import main
    from imx219_yolov8_trt import resolve_target_class_id


YoloV8nTRTDetector = YoloV8TRTDetector


if __name__ == "__main__":
    main()