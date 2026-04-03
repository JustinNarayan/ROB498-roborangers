# RC Car YOLOv5 Training and Jetson Deployment

This workflow is tuned for Jetson Nano deployment and ROS2 centroid tracking.

## 1) Label Format and Folder Layout

Use YOLO format labels (`class x_center y_center width height`, normalized [0,1]).

```
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Use a single class first:

```
names:
  0: rc_car
```

## 2) Training on Desktop GPU

Clone YOLOv5 on desktop and install requirements:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

Create `rc_car_data.yaml` based on `roborangers/yolo/rc_car_data_template.yaml`.

Train from pretrained weights:

```bash
python train.py \
  --img 640 \
  --batch 32 \
  --epochs 150 \
  --data rc_car_data.yaml \
  --weights yolov5n.pt \
  --project runs_rc \
  --name yolov5n_rc
```

## 3) Export Best Model to ONNX

```bash
python export.py \
  --weights runs_rc/yolov5n_rc/weights/best.pt \
  --include onnx \
  --img 640 \
  --opset 12
```

Copy `best.onnx` to Jetson.

## 4) Run Standalone Detector on Jetson

```bash
python3 roborangers/yolo/imx219_yolov5_onnx.py \
  --model /absolute/path/to/best.onnx \
  --class-names /absolute/path/to/classes.txt \
  --input-size 640 \
  --conf 0.35 \
  --iou 0.45
```

`classes.txt` should contain one class name per line (example: `rc_car`).

## 5) Run ROS2 Centroid Publisher

Build and source:

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select roborangers
source install/setup.bash
```

Run node:

```bash
ros2 run roborangers centroid_detector_node.py --ros-args \
  -p model_path:=/absolute/path/to/best.onnx \
  -p class_names_path:=/absolute/path/to/classes.txt \
  -p target_class_name:=rc_car \
  -p display:=true
```

Published topics:

- `/vision/target_centroid` (`Float32MultiArray`): `[cx, cy, nx, ny, confidence, class_id]`
- `/vision/target_bbox` (`Float32MultiArray`): `[x1, y1, x2, y2, confidence, class_id]`
- `/vision/target_status` (`String`): `DETECTED` or `LOST`

Where normalized errors are:

- `nx = (cx - W/2) / (W/2)`
- `ny = (cy - H/2) / (H/2)`

These can be consumed directly by the drone controller.
