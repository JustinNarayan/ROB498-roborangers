#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy

class CameraPoseForward(Node):
    def __init__(self):
        super().__init__('camera_pose_forward')
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        
        self.subscription = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.pose_callback,
            qos_profile
        )
        self.publisher = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )
        self.get_logger().info('camera_pose_forward node started')
        
    def pose_callback(self, msg):
        new_message = PoseStamped()
        new_message.header.stamp = msg.header.stamp
        new_message.header.frame_id = msg.header.frame_id
        new_message.pose = msg.pose.pose
        self.publisher.publish(new_message)
        
def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()