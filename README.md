# ROB498 - Roborangers
## Rafiu Hossain, Isabelle Tung, Yang Niu, Sujit Peramanu, Justin Narayan

# Maintenance
## CMakeLists.txt
Upon creation of a new node, make sure to add it in CMakeLists.txt

## Building
(1) Go to outer workspace folder (i.e. ~/ws) and type `rm -rf build install`
(2) and type `colcon build --symlink-install --packages-select roborangers`
(3) Type `source install/setup.bash`

## Running
Package is `roborangers`.
After executing the "Building" steps:
(1) `ros2 run roborangers realsense.py` or similar
(2) `ros2 launch roborangers mavros.launch.py` or similar

## YOLOv5 RC-Car Centroid Pipeline (IMX219)
The package now includes a Jetson-ready detector that runs YOLOv5 ONNX on IMX219 and publishes centroid targets for control.

### Build
From workspace root:
(1) `colcon build --symlink-install --packages-select roborangers`
(2) `source install/setup.bash`

### Standalone detector test
`python3 roborangers/yolo/imx219_yolov5_onnx.py --model /absolute/path/to/best.onnx --class-names /absolute/path/to/classes.txt --input-size 640 --conf 0.35 --iou 0.45`

### Quick pretrained YOLOv5 video check (no training)
This runs pretrained COCO weights and draws bounding boxes + centroids.

(1) `pip install ultralytics`
(2) `python3 roborangers/opencv/yolov5_video_inference.py`

Optional examples:
- `python3 roborangers/opencv/yolov5_video_inference.py --video roborangers/opencv/videos/nvcamtest_10192_s00_00000.mp4`
- `python3 roborangers/opencv/yolov5_video_inference.py --all-classes`
- `python3 roborangers/opencv/yolov5_video_inference.py --save /tmp/yolo_preview.mp4`

### ROS2 centroid node
`ros2 run roborangers centroid_detector_node.py --ros-args -p model_path:=/absolute/path/to/best.onnx -p class_names_path:=/absolute/path/to/classes.txt -p target_class_name:=rc_car -p display:=true`

### ROS2 launch (recommended)
`ros2 launch roborangers centroid_detector.launch.py model_path:=/absolute/path/to/best.onnx class_names_path:=/absolute/path/to/classes.txt target_class_name:=rc_car display:=true`

Default parameters are in `launch/centroid_detector_params.yaml`.

Published topics:
(1) `/vision/target_centroid` (`Float32MultiArray`): `[cx, cy, nx, ny, confidence, class_id]`
(2) `/vision/target_bbox` (`Float32MultiArray`): `[x1, y1, x2, y2, confidence, class_id]`
(3) `/vision/target_status` (`String`): `DETECTED` or `LOST`

### Training and export guide
See `roborangers/yolo/train_rc_car.md` for end-to-end dataset, training, export, and deployment commands.

## Flight Exercises
# Flight Exercise 2
(1) Change ROS node `USING_REALSENSE = True` or `False`
(2) Run MAVROS `ros2 launch roborangers mavros.launch.py`
(3) { in arena } `ros2 launch realsense2_camera rs_launch.py`
(4) Ensure `ros2 topic echo /camera/pose/sample --truncate-length 5` is logical
(5) Run ROS node `ros2 run roborangers fe2.py`
(6) Wait for `Initial Pose Computed`
(7) Run Launch `ros2 service call /rob498_drone_06/comm/launch std_srvs/srv/Trigger`

# Flight Exercise 3
(1) Set `CURRENT_MISSION` to `MissionType.VICON` or `MissionType.REALSENSE`
(2) Run MAVROS `ros2 launch roborangers mavros.launch.py`
(4) Ensure `ros2 topic echo /camera/pose/sample --truncate-length 5` is logical
(3) { in arena } `ros2 launch realsense2_camera rs_launch.py`
(4) Run ROS node `ros2 run roborangers fe2.py`
(5)