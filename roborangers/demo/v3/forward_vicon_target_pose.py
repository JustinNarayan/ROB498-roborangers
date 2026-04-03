#!/usr/bin/env python3

###############################################
#                 U S A G E                   #
###############################################
'''
Subscribes to a Vicon rigid-body pose for the RC car and forwards it
unchanged on the shared /target/pose topic consumed by CommNode.

No transform or header manipulation is applied — the message is forwarded
as-is so that CommNode's TargetState receives it with the original Vicon
timestamp (used for staleness checking).

LAUNCH:
    ros2 run roborangers forward_vicon_target_pose.py
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped

from constants import TARGET_POSE_TOPIC_NAME, VICON_RC_CAR_TOPIC

###############################################
#           F O R W A R D E R   N O D E       #
###############################################

class ForwardViconTargetPoseNode(Node):
    def __init__(self):
        super().__init__('forward_vicon_target_pose')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._pub = self.create_publisher(PoseStamped, TARGET_POSE_TOPIC_NAME, qos)

        self._sub = self.create_subscription(
            PoseStamped,
            VICON_RC_CAR_TOPIC,
            self._forward,
            qos,
        )

        self.get_logger().info(
            f'forward_vicon_target_pose ready.\n'
            f'Forwarding [{VICON_RC_CAR_TOPIC}] -> [{TARGET_POSE_TOPIC_NAME}]'
        )

    def _forward(self, msg: PoseStamped) -> None:
        self._pub.publish(msg)

###############################################
#              M A I N   L O O P              #
###############################################

def main(args=None):
    rclpy.init(args=args)
    node = ForwardViconTargetPoseNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
