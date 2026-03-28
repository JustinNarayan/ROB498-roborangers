cmake_minimum_required(VERSION 3.10)
project(roborangers)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  roborangers/fe2/fe2.py
  roborangers/fe2/fe2_combined.py
  roborangers/fe2/fe2_realsense.py
  roborangers/realsense.py
  roborangers/vicon.py
  roborangers/fe3/fe3.py
  roborangers/demo/v0/demo_v0.py
  roborangers/demo/v1/demo_v1.py
  roborangers/demo/v2/demo_v2.py
  roborangers/demo/v2/simulate_target_pose.py
  roborangers/demo/v3/demo_v3.py
  roborangers/demo/v3/forward_vicon_target_pose.py
  roborangers/opencv/yolov5_video_inference.py
  roborangers/opencv/yolov8_video_inference.py
  roborangers/yolo/imx219_yolov8_trt.py
  roborangers/yolo/imx219_yolov8n_trt.py
  roborangers/yolo/centroid_detector_node_trt.py
  roborangers/cv/opencv/yolov8_video_inference.py
  roborangers/cv/yolo/imx219_yolov8_trt.py
  roborangers/cv/yolo/imx219_yolov8n_trt.py
  roborangers/cv/yolo/centroid_detector_node_trt.py
  roborangers/cv/calibration/export_t265_intrinsics.py
  roborangers/cv/calibration/generate_kalibr_files.py
  roborangers/cv/calibration/imx219_calibration_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
