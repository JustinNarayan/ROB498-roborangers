#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from roborangers.utils.pose_utils import compute_average_pose, subtract_poses

###############################################
#              C O N S T A N T S              #
###############################################

NODE_LABEL = 'rob498_drone_06'
INIT_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose

###############################################
#            D R O N E   S T A T E            #
###############################################

class DroneState:
    def __init__(self):
        self.drone_flight_commanded = False
        self.init_pose_list = []  # List of PoseStamped for averaging
        self.init_pose = None     # PoseStamped after averaging
        self.current_pose = PoseStamped()

drone_state = DroneState()

###############################################
#               H A N D L E R S               #
###############################################

def handle_launch():
    self.get_logger().info('Launch Requested.')
    # Drone should fly
    drone_state.drone_flight_commanded = True

def handle_test():
    self.get_logger().info('Test Requested.')

def handle_land():
    self.get_logger().info('Land Requested.')
    # Drone should not fly
    drone_state.drone_flight_commanded = False

def handle_abort():
    self.get_logger().info('Abort Requested.')
    # Drone should not fly
    drone_state.drone_flight_commanded = False

def handle_vicon_pose(msg: PoseStamped):
    # Store initial poses to compute neutral init_pose
    if len(drone_state.init_pose_list) < INIT_POSE_COUNT_MAX:
        drone_state.init_pose_list.append(msg)
        # Compute init after receiving enough
        if len(drone_state.init_pose_list) == INIT_POSE_COUNT_MAX:
            drone_state.init_pose = compute_average_pose(drone_state.init_pose_list)
        return
    # Current pose is offset from the init
    drone_state.current_pose = subtract_poses(msg, drone_state.init_pose)

###############################################
#              C A L L B A C K S              #
###############################################

def callback_launch(request, response):
    handle_launch()
    return response

def callback_test(request, response):
    handle_test()
    return test

def callback_land(request, response):
    handle_land()
    return response

def callback_abort(request, response):
    handle_abort()
    return response

def callback_vicon_pose(msg: PoseStamped):
    # handle_vicon_pose()
    pass

###############################################
#         C O M M U N I C A T I O N S         #
###############################################

class CommNode(Node):
    def __init__(self):
        super().__init__(NODE_LABEL)
        
        ### Callbacks
        # Generate callbacks to respond to commands for launch, test, land, abort
        self.srv_launch = self.create_service(
            Trigger, f'{NODE_LABEL}/comm/launch', callback_launch
        )
        self.srv_test = self.create_service(
            Trigger, f'{NODE_LABEL}/comm/test', callback_test
        )
        self.srv_land = self.create_service(
            Trigger, f'{NODE_LABEL}/comm/land', callback_land
        )
        self.srv_abort = self.create_service(
            Trigger, f'{NODE_LABEL}/comm/abort', callback_abort
        )
        
        ### VICON
        # Drone subscribes to Vicon pose
        qos_profile = QoSProfile(depth=10)
        self.sub_vicon_pose = self.create_subscription(
            PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', callback_vicon_pose, qos_profile
        )

###############################################
#              M A I N   L O O P              #
###############################################

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()