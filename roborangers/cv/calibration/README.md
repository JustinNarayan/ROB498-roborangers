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

## Export T265 intrinsics with Realsense actively publishing

```bash
ros2 run roborangers export_t265_intrinsics.py -- \
  --source topics \
  --output-dir /tmp/t265_calibration
```

## Generate Kalibr YAML files

```bash
ros2 run roborangers generate_kalibr_files.py -- \
  --imx-yaml /home/jetson/imx219_calibration/ost.yaml \
  --fisheye-yaml /home/jetson/t265_calibration/t265_fisheye1.yaml \
  --output-dir /tmp/kalibr_inputs
```

For a Kalibr omni model, add `--fisheye-kalibr-model omni --fisheye-xi <value>`.

## Record ROS Bag for IMX219 and Left fisheye Camera

```bash
ros2 bag record \
    /imx219/image_raw \
    /camera/fisheye1/image_raw \
    -o multicam_calib_bag
```

## Kalibr installation

```bash
docker pull stereolabs/kalibr
# or build from source:
cd /home/jetson
git clone https://github.com/ethz-asl/kalibr.git
cd kalibr
docker build -t kalibr .
```

`docker build -t kalibr .` must be run inside the cloned ETH Kalibr repository, not inside your local calibration output folder.

## Run Kalibr

```bash
cd /tmp/kalibr_inputs
docker run -it \
    -v $(pwd):/data \
    kalibr \
    kalibr_calibrate_cameras \
  --bag /data/multicam_calib_bag/multicam_calib_bag_0.db3 \
    --target /data/aprilgrid.yaml \
    --models pinhole-radtan omni-equidist \
    --topics /imx219/image_raw /camera/fisheye1/image_raw \
    --dont-show-report
```

If you recorded with `ros2 bag record`, the bag will usually be a directory containing `.db3` data rather than a ROS 1 `.bag` file. Put a copy of that bag directory alongside `aprilgrid.yaml` and `cameras.yaml`, or mount both paths into the container explicitly.

## CMAKE force rebuild:

```bash
cd ~/ros2_ws && colcon build --packages-select roborangers --cmake-force-configure 2>&1 | tail -20
```