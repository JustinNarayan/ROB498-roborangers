# Calibration Scripts

## IMX219 publisher

Build the package:

```bash
colcon build --symlink-install --packages-select roborangers
source install/setup.bash
```

Then run:

```bash
ros2 run roborangers imx219_calibration_publisher.py
```

Default topics:

- `/imx219/image_raw`
- `/imx219/camera_info`

Run the calibrator:

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.025 \
  --camera_name imx219 \
  image:=/imx219/image_raw \
  camera:=/imx219
```
If you already have an `ost.yaml`, republish it with:

```bash
ros2 run roborangers imx219_calibration_publisher.py --ros-args -p camera_info_path:=/absolute/path/to/ost.yaml
```

## T265 fisheye publisher

```bash
ros2 run roborangers t265_fisheye_publisher.py
```

Default topics:

- `/camera/fisheye1/image_raw`
- `/camera/fisheye1/camera_info`
- `/camera/fisheye2/image_raw`
- `/camera/fisheye2/camera_info`

## Export T265 intrinsics

```bash
ros2 run roborangers export_t265_intrinsics.py -- --output-dir /tmp/t265_calibration
```

## Generate Kalibr YAML files

```bash
ros2 run roborangers generate_kalibr_files.py -- \
  --imx-yaml /tmp/imx219_ost.yaml \
  --fisheye-yaml /tmp/t265_calibration/t265_fisheye1.yaml \
  --output-dir /tmp/kalibr_inputs
```

For a Kalibr omni model, add `--fisheye-kalibr-model omni --fisheye-xi <value>`.