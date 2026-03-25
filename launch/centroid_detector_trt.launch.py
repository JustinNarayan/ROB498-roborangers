#!/usr/bin/env python3
"""Launch the TensorRT-based centroid detector node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for centroid detector TRT node."""
    return LaunchDescription([
        Node(
            package='roborangers',
            executable='centroid_detector_node_trt.py',
            name='centroid_detector_node',
            parameters=[
                {'model_path': '/home/jetson/yolov8ntrained.engine'},
                {'input_size': 320},
                {'target_class_name': 'cars'},
                {'confidence_threshold': 0.35},
                {'iou_threshold': 0.45},
                {'camera_width': 1280},
                {'camera_height': 720},
                {'display_width': 1280},
                {'display_height': 720},
                {'camera_fps': 30},
                {'publish_rate_hz': 20.0},
                {'display': False},
            ],
            output='screen',
        )
    ])
