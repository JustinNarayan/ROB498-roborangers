# ROB498 - Roborangers
## Rafiu Hossain, Isabelle Tung, Yang Niu, Sujit Peramanu, Justin Narayan

# Maintenance
## CMakeLists.txt
Upon creation of a new node, make sure to add it in CMakeLists.txt

## Building
(1) Go to outer workspace folder (i.e. ~/ws) and type `colcon build`
(2) Type `source install/setup.bash`

## Running
Package is `roborangers`.
After executing the "Building" steps:
(1) `ros2 run roborangers realsense.py` or similar
(2) `ros2 launch roborangers mavros.launch.py` or similar