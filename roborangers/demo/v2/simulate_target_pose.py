#!/usr/bin/env python3

###############################################
#                 U S A G E                   #
###############################################
'''
Simulates the target/pose topic for testing without a real CV pipeline.
No custom .srv files or rebuilding required — everything is driven by
ROS2 parameters set from the CLI on the running node.

LAUNCH:
    python3 simulate_target_pose.py

COMMANDS (run in a separate terminal while the node is running):

  Set target position only (orientation defaults to identity 0,0,0,1):
    ros2 param set /simulate_target_pose target_x 1.5
    ros2 param set /simulate_target_pose target_y 2.0
    ros2 param set /simulate_target_pose target_z 0.0

  Set orientation (after setting position above):
    ros2 param set /simulate_target_pose target_qx 0.0
    ros2 param set /simulate_target_pose target_qy 0.0
    ros2 param set /simulate_target_pose target_qz 0.707
    ros2 param set /simulate_target_pose target_qw 0.707

  Clear (return to publishing all-zero sentinel / "no target"):
    ros2 param set /simulate_target_pose clear true

  Toggle publishing on/off (to test staleness timeout):
    ros2 param set /simulate_target_pose publishing_enabled false
    ros2 param set /simulate_target_pose publishing_enabled true

  Inspect all current parameters:
    ros2 param list /simulate_target_pose
    ros2 param dump /simulate_target_pose
'''

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

from constants import DRONE_ID, TARGET_POSE_TOPIC_NAME, TARGET_SIM_PUBLISH_RATE

###############################################
#         S I M U L A T I O N   N O D E       #
###############################################

class SimulateTargetPoseNode(Node):
    def __init__(self):
        super().__init__('simulate_target_pose')

        self._clock = self.get_clock()

        ### Declare all parameters with defaults
        # Position
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 0.0)
        # Orientation
        self.declare_parameter('target_qx', 0.0)
        self.declare_parameter('target_qy', 0.0)
        self.declare_parameter('target_qz', 0.0)
        self.declare_parameter('target_qw', 0.0)  # 0.0 = all-zero sentinel by default
        # Control
        self.declare_parameter('publishing_enabled', True)
        self.declare_parameter('clear', False)

        ### Register parameter change callback
        self.add_on_set_parameters_callback(self._on_parameter_change)

        ### Publisher
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._pub = self.create_publisher(PoseStamped, TARGET_POSE_TOPIC_NAME, qos)

        ### Publish timer
        self.create_timer(1.0 / TARGET_SIM_PUBLISH_RATE, self._publish_loop)

        self.get_logger().info(
            f'simulate_target_pose ready.\n'
            f'Publishing to [{TARGET_POSE_TOPIC_NAME}] at {TARGET_SIM_PUBLISH_RATE} Hz.\n'
            f'Default payload is all-zero (no target). '
            f'Use "ros2 param set /simulate_target_pose <param> <value>" to control.'
        )

    # ------------------------------------------------------------------
    # Parameter change callback
    # ------------------------------------------------------------------

    def _on_parameter_change(self, params: list) -> SetParametersResult:
        for param in params:
            if param.name == 'clear' and param.value is True:
                # Reset all pose parameters back to zero
                self.set_parameters([
                    Parameter('target_x',   Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_y',   Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_z',   Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_qx',  Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_qy',  Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_qz',  Parameter.Type.DOUBLE, 0.0),
                    Parameter('target_qw',  Parameter.Type.DOUBLE, 0.0),
                    Parameter('clear',      Parameter.Type.BOOL,   False),
                ])
                self.get_logger().info('Target cleared — publishing all-zero sentinel.')

            elif param.name == 'publishing_enabled':
                state = 'ENABLED' if param.value else 'DISABLED'
                self.get_logger().info(f'Publishing {state}.')

            elif param.name.startswith('target_'):
                self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')

        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Publish loop
    # ------------------------------------------------------------------

    def _publish_loop(self):
        if not self.get_parameter('publishing_enabled').value:
            return

        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self._clock.now().to_msg()

        msg.pose.position.x    = self.get_parameter('target_x').value
        msg.pose.position.y    = self.get_parameter('target_y').value
        msg.pose.position.z    = self.get_parameter('target_z').value
        msg.pose.orientation.x = self.get_parameter('target_qx').value
        msg.pose.orientation.y = self.get_parameter('target_qy').value
        msg.pose.orientation.z = self.get_parameter('target_qz').value
        msg.pose.orientation.w = self.get_parameter('target_qw').value

        self._pub.publish(msg)

###############################################
#              M A I N   L O O P              #
###############################################

def main(args=None):
    rclpy.init(args=args)
    node = SimulateTargetPoseNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
