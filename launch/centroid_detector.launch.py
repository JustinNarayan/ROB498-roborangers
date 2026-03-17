from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory("roborangers")
    default_params = package_share + "/launch/centroid_detector_params.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to detector parameter YAML",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Absolute path to YOLOv5 ONNX model",
            ),
            DeclareLaunchArgument(
                "class_names_path",
                default_value="",
                description="Absolute path to class names text file",
            ),
            DeclareLaunchArgument(
                "target_class_name",
                default_value="rc_car",
                description="Class name to track",
            ),
            DeclareLaunchArgument(
                "display",
                default_value="false",
                description="Enable OpenCV preview window",
            ),
            Node(
                package="roborangers",
                executable="centroid_detector_node.py",
                name="centroid_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "class_names_path": LaunchConfiguration("class_names_path"),
                        "target_class_name": LaunchConfiguration("target_class_name"),
                        "display": LaunchConfiguration("display"),
                    },
                ],
            ),
        ]
    )
