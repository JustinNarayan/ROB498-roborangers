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

## Flight Exercises
# Flight Exercise 2
(1) Change ROS node `USING_REALSENSE = True` or `False`
(2) Run MAVROS `ros2 launch roborangers mavros.launch.py`
(3) { in arena } `ros2 launch realsense2_camera rs_launch.py`
(4) Ensure `ros2 topic echo /camera/pose/sample --truncate-length 5` is logical
(5) Run ROS node `ros2 run roborangers fe2.py`
(6) Wait for `Initial Pose Computed`
(7) Run Launch `ros2 service call /rob498_drone_06/comm/launch std_srvs/srv/Trigger`