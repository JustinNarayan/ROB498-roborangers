#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Services and Clients
from std_srvs.srv import Trigger
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# Messages
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State

# Quality of Service for subscriptions
from rclpy.qos import QoSProfile

# Math utilities
from roborangers.utils.pose_utils import compute_average_pose, subtract_poses

###############################################
#              C O N S T A N T S              #
###############################################

DRONE_ID = 'rob498_drone_06'
VICON_TOPIC_NAME = '/vicon/fixedwing_robot_2/fixedwing_robot_2' # check via `ros2 topic list`
INIT_VICON_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose
TARGET_HEIGHT = 1.5 # meters
QOS_DEPTH = 10 # number of messages to store
COMMAND_RATE = 20 # Hz, recommended in procedure.md
OFFBOARD_MODE = 'OFFBOARD'
DEBUGGING_LOOP_LOGS = False
DEBUGGING_POSE = True

###############################################
#            V I C O N   S T A T E            #
###############################################

class ViconState:
    def __init__(self):
        # Drone VICON data 
        self.init_vicon_pose_list = []  # List of PoseStamped for averaging
        self.init_vicon_pose = None     # PoseStamped after averaging
        self.current_vicon_pose = PoseStamped() # w.r.t. init_vicon_pose
        
        # Target pose - full neutral, since init pose sets the new zero
        # NOTE: time stamp needed for each update
        self.target_hover_pose = PoseStamped()
        self.target_hover_pose.header.frame_id = 'map'
        # This should work?
        self.target_hover_pose.pose.position.x = 0.0
        self.target_hover_pose.pose.position.y = 0.0
        self.target_hover_pose.pose.position.z = TARGET_HEIGHT
        # This may not work, may need to be init_pose orientation?
        self.target_hover_pose.pose.orientation.x = 0.0
        self.target_hover_pose.pose.orientation.y = 0.0
        self.target_hover_pose.pose.orientation.z = 0.0
        self.target_hover_pose.pose.orientation.w = 1.0 # corresponds to "zero" R-P-Y

###############################################
#         C O M M U N I C A T I O N S         #
###############################################

class CommNode(Node):
    def __init__(self):
        super().__init__(DRONE_ID)
        
        ### VICON data
        self.vicon_state = ViconState()
        
        ### Command variables
        self.drone_flight_commanded = False # should try to hover
        
        ### MAVROS State variables
        self.current_mavros_state = State()
        
        ### Control loop information
        self.control_timer = self.create_timer( 1.0 / COMMAND_RATE, self.control_loop )
        
        ### Testing Services
        # Generate callbacks to respond to commands for launch, test, land, abort
        self.srv_launch = self.create_service(
            Trigger, f'{DRONE_ID}/comm/launch', self.callback_launch
        )
        self.srv_test = self.create_service(
            Trigger, f'{DRONE_ID}/comm/test', self.callback_test
        )
        self.srv_land = self.create_service(
            Trigger, f'{DRONE_ID}/comm/land', self.callback_land
        )
        self.srv_abort = self.create_service(
            Trigger, f'{DRONE_ID}/comm/abort', self.callback_abort
        )
        
        ### VICON
        # Drone subscribes to Vicon pose
        qos_vicon_pose = QoSProfile(depth=QOS_DEPTH)
        self.sub_vicon_pose = self.create_subscription(
            PoseStamped, VICON_TOPIC_NAME, self.callback_vicon_pose, qos_vicon_pose
        )
        
        ### MAVROS
        # Drone subscribes to MAVROS state
        qos_mavros_state = QoSProfile(depth=QOS_DEPTH)
        self.sub_mavros_state = self.create_subscription(
            State, '/mavros/state', self.callback_mavros_state, qos_mavros_state
        )
        # Drone publishes target setpoint over MAVROS to flight controller
        qos_mavros_setpoint = QoSProfile(depth=QOS_DEPTH)
        self.pub_mavros_setpoint = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', qos_mavros_setpoint
        )
        
        ### MAVROS Clients
        # Generate callbacks to communicate over MAVROS
        self.cli_set_mode = self.create_client(
            SetMode, '/mavros/set_mode'
        )
        self.cli_arming = self.create_client(
            CommandBool, '/mavros/cmd/arming'
        )
        self.cli_land = self.create_client(
            CommandTOL, '/mavros/cmd/land'
        )
        # Wait for services to enable
        self.cli_set_mode.wait_for_service()
        self.cli_arming.wait_for_service()
        self.cli_land.wait_for_service()
    
    '''
    Service and Subscription Callbacks
    '''
    def callback_launch(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return handle_launch(self, request, response)

    def callback_test(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return handle_test(self, request, response)

    def callback_land(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return handle_land(self, request, response)

    def callback_abort(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return handle_abort(self, request, response)

    def callback_vicon_pose(
        self, 
        msg: PoseStamped
    ) -> None:
        handle_vicon_pose(self, msg)

    def callback_mavros_state(
        self, 
        msg: State
    ) -> None:
        handle_mavros_state(self, msg)      
        
    '''
    Client request commands
    '''
    def request_offboard_mode(self):
        req = SetMode.Request()
        req.custom_mode = OFFBOARD_MODE
        self.cli_set_mode.call_async(req)
        
    def request_arm(self):
        req = CommandBool.Request()
        req.value = True
        self.cli_arming.call_async(req)
    
    def request_land(self):
        req = CommandTOL.Request()
        self.cli_land.call_async(req)
    
    '''
    Drone continuous control logic, running at COMMAND_RATE
    '''
    def control_loop(self):
        if DEBUGGING_LOOP_LOGS:
            self.get_logger().info('Control loop!')
        
        # Ensure initial pose has been calibrated
        if self.vicon_state.init_vicon_pose is None:
            return
        
        # Check MAVROS state flags
        _is_connected = self.current_mavros_state.connected
        _is_armed = self.current_mavros_state.armed
        _is_offboard = self.current_mavros_state.mode == OFFBOARD_MODE
        
        # Ensure MAVROS connected
        if not _is_connected:
            return

        # Check if drone should fly
        if self.drone_flight_commanded:
            # Publish hover setpoint
            # This must be published BEFORE offboard mode is enabled (dummy setpoints would suffice)
            self.vicon_state.target_hover_pose.header.stamp = \
                self.get_clock().now().to_msg()
            self.pub_mavros_setpoint.publish(self.vicon_state.target_hover_pose)
            
            # Enable offboard control if not yet in offboard control
            if not _is_offboard:
                self.request_offboard_mode()
                return # Wait till next loop
            
            # Arm the drone if not yet armed
            if not _is_armed:
                self.request_arm()
                return # Wait till next loop
        else:
            # Land, if armed and in offboard mode
            if _is_armed:
                self.request_land()

###############################################
#               H A N D L E R S               #
###############################################

def handle_launch(
    self: CommNode, 
    request: Trigger.Request,
    response: Trigger.Response
) -> Trigger.Response:
    self.get_logger().info('Launch Requested.')
    
    # Ensure pose has been initialized
    if self.vicon_state.init_vicon_pose is None:
        response.success = False
        response.message = "Init pose still calculating!"
    else:
        self.drone_flight_commanded = True
        response.success = True
        
    return response

def handle_test(
    self: CommNode, 
    request: Trigger.Request,
    response: Trigger.Response
) -> Trigger.Response:
    self.get_logger().info('Test Requested.')
    return response

def handle_land(
    self: CommNode, 
    request: Trigger.Request,
    response: Trigger.Response
) -> Trigger.Response:
    self.get_logger().info('Land Requested.')
    self.drone_flight_commanded = False
    response.success = True
    return response

def handle_abort(
    self: CommNode, 
    request: Trigger.Request,
    response: Trigger.Response
) -> Trigger.Response:
    self.get_logger().info('Abort Requested.')
    # Same behaviour as landing
    self.drone_flight_commanded = False
    response.success = True
    return response

def handle_vicon_pose(
    self: CommNode, 
    msg: PoseStamped
) -> None:
    if DEBUGGING_LOOP_LOGS:
        self.get_logger().info('Vicon received!')
        
    # Store initial poses to compute neutral init_vicon_pose
    if len(self.vicon_state.init_vicon_pose_list) < INIT_VICON_POSE_COUNT_MAX:
        self.vicon_state.init_vicon_pose_list.append(msg)
        # Compute init after receiving enough
        if len(self.vicon_state.init_vicon_pose_list) == INIT_VICON_POSE_COUNT_MAX:
            self.vicon_state.init_vicon_pose = compute_average_pose(
                self.vicon_state.init_vicon_pose_list
            )
        return
    # Current pose is offset from the init
    self.vicon_state.current_vicon_pose = subtract_poses(msg, self.vicon_state.init_vicon_pose)
    
    if DEBUGGING_POSE:
        self.get_logger().info(f'\
            (x:{self.vicon_state.current_vicon_pose.pose.position.x}, \
            y:{self.vicon_state.current_vicon_pose.pose.position.y}, \
            z:{self.vicon_state.current_vicon_pose.pose.position.x}),\
            (x:{self.vicon_state.current_vicon_pose.pose.orientation.x},\
            y:{self.vicon_state.current_vicon_pose.pose.orientation.y},\
            z:{self.vicon_state.current_vicon_pose.pose.orientation.z},\
            w:{self.vicon_state.current_vicon_pose.pose.orientation.w})')

def handle_mavros_state(
    self: CommNode, 
    msg: State
) -> None:
    self.current_mavros_state = msg      

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