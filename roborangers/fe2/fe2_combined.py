#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Services and Clients
from std_srvs.srv import Trigger
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# Messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State

# Quality of Service for subscriptions
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Math utilities
from roborangers.utils.pose_utils import compute_average_pose

###############################################
#               C O M M A N D S               #
###############################################

# ros2 service call /rob498_drone_06/comm/land std_srvs/srv/Trigger {}
# ros2 service call /rob498_drone_06/comm/launch std_srvs/srv/Trigger {}
# ros2 service call /rob498_drone_06/comm/abort std_srvs/srv/Trigger {}
# ros2 service call /rob498_drone_06/comm/test std_srvs/srv/Trigger {}

###############################################
#              C O N S T A N T S              #
###############################################

USING_REALSENSE = False
DRONE_ID = 'rob498_drone_06'
VICON_TOPIC_NAME = '/vicon/ROB498_Drone/ROB498_Drone' # check via `ros2 topic list`
REALSENSE_TOPIC_NAME = '/camera/pose/sample'
INIT_VISION_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose
TARGET_HEIGHT = 1.5 # meters
QOS_DEPTH = 10 # number of messages to store
COMMAND_RATE = 20 # Hz, recommended in procedure.md
OFFBOARD_MODE = 'OFFBOARD'
ALTITUDE_MODE = 'ALTCTL'
DEBUGGING_LOOP_LOGS = False
DEBUGGING_POSE = False

###############################################
#            V I C O N   S T A T E            #
###############################################

class VisionState:
    def __init__(self):
        # Drone vision data 
        self.init_vision_pose_list = []  # List of PoseStamped for averaging
        self.init_vision_pose = None     # PoseStamped after averaging
        self.current_vision_pose = PoseStamped()

###############################################
#         C O M M U N I C A T I O N S         #
###############################################

class CommNode(Node):
    def __init__(self):
        super().__init__(DRONE_ID)
        
        ### VISION data
        self.vision_state = VisionState()
        
        ### Command variables
        self.drone_flight_commanded = False
        
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
        
        ### VISION
        if USING_REALSENSE:
            # Drone subscribes to Camera pose
            qos_camera_pose = QoSProfile(depth=QOS_DEPTH)
            qos_camera_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_camera_pose = self.create_subscription(
                Odometry, 
                REALSENSE_TOPIC_NAME, 
                self.callback_camera_pose, 
                qos_camera_pose
            )
        else:
            # Drone subscribes to vision pose
            qos_vicon_pose = QoSProfile(depth=QOS_DEPTH)
            qos_vicon_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_vicon_pose = self.create_subscription(
                PoseStamped, 
                VICON_TOPIC_NAME, 
                self.callback_vicon_pose, 
                qos_vicon_pose
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
        # Drone publishes Vision pose to vision EKF source for Cube
        qos_mavros_vision_pose = QoSProfile(depth=QOS_DEPTH)
        self.pub_mavros_vision_pose = self.create_publisher(
            PoseStamped, '/mavros/vision_pose/pose', qos_mavros_vision_pose
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
        return self.handle_launch(request, response)

    def callback_test(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return self.handle_test(request, response)

    def callback_land(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return self.handle_land(request, response)

    def callback_abort(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        return self.handle_abort(request, response)

    def callback_vicon_pose(
        self, 
        msg: Odometry
    ) -> None:
        self.handle_vicon_pose(msg)
    
    def callback_camera_pose(
        self, 
        msg: Odometry
    ) -> None:
        self.handle_camera_pose(msg)

    def callback_mavros_state(
        self, 
        msg: State
    ) -> None:
        self.handle_mavros_state(msg)      
        
    '''
    Client request commands
    '''
    def request_offboard_mode(self):
        self.get_logger().info('Requesting offboard')
        req = SetMode.Request()
        req.custom_mode = OFFBOARD_MODE
        self.cli_set_mode.call_async(req)
        self.get_logger().info('Requested offboard')
    
    def request_altitude_mode(self):
        self.get_logger().info('Requesting altitude')
        req = SetMode.Request()
        req.custom_mode = ALTITUDE_MODE
        self.cli_set_mode.call_async(req)
        self.get_logger().info('Requested altitude')
    
    def request_arm(self):
        self.get_logger().info('Requesting arm')
        req = CommandBool.Request()
        req.value = True
        self.cli_arming.call_async(req)
        self.get_logger().info('Requested arm')
    
    def request_land(self):
        self.get_logger().info('Requesting land')
        req = CommandTOL.Request()
        self.cli_land.call_async(req)
        self.get_logger().info('Requested land')
    
    '''
    Drone continuous control logic, running at COMMAND_RATE
    '''
    def control_loop(self):
        if DEBUGGING_LOOP_LOGS:
            self.get_logger().info('Control loop!')
            
        # Redirect received vision data to mavros
        redirected_pose = self.vision_state.current_vision_pose
        redirected_pose.header.frame_id = 'map'
        self.pub_mavros_vision_pose.publish(redirected_pose)
        
        # Ensure initial pose has been calibrated
        if self.vision_state.init_vision_pose is None:
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
            # Construct target hover
            target_hover_pose = PoseStamped()
            target_hover_pose.header.frame_id = 'map'
            target_hover_pose.pose = self.vision_state.init_vision_pose.pose
            if self.drone_flight_test_commanded:
                target_hover_pose.pose.position.z += TARGET_HEIGHT
            
            # Publish hover setpoint
            # This must be published BEFORE offboard mode is enabled (dummy setpoints would suffice)
            target_hover_pose.header.stamp = \
                self.get_clock().now().to_msg()
            self.pub_mavros_setpoint.publish(target_hover_pose)
            
            # Arm the drone if not yet armed
            if not _is_armed:
                # Set to Altitude (Manual) mode before arming
                if _is_offboard:
                    self.request_altitude_mode()
                    return
                self.request_arm()
                return # Wait till next loop
            
            # Enable offboard control if not yet in offboard control
            if not _is_offboard:
                self.request_offboard_mode()
                return # Wait till next loop
            
            if DEBUGGING_LOOP_LOGS:
                self.get_logger().info('Ready!')
        else:
            # Land, if armed and in offboard mode
            if _is_armed:
                self.request_land()

###############################################
#               H A N D L E R S               #
###############################################

    def handle_launch(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Launch Requested.')
        
        # Ensure pose has been initialized
        if self.vision_state.init_vision_pose is None:
            response.success = False
            response.message = "Init pose still calculating!"
        else:
            self.drone_flight_commanded = True
            response.success = True
            
        return response

    def handle_test(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Test Requested.')
        self.drone_flight_test_commanded = True
        return response

    def handle_land(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Land Requested.')
        self.drone_flight_commanded = False
        self.drone_flight_test_commanded = False
        response.success = True
        return response

    def handle_abort(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Abort Requested.')
        # Same behaviour as landing
        self.drone_flight_commanded = False
        self.drone_flight_test_commanded = False
        response.success = True
        return response

    def handle_camera_pose(
        self, 
        msg: Odometry
    ) -> None:
        if DEBUGGING_LOOP_LOGS:
            self.get_logger().info('Camera received!')
            
        new_message = PoseStamped()
        new_message.header.stamp = msg.header.stamp
        new_message.header.frame_id = msg.header.frame_id
        new_message.pose = msg.pose.pose
            
        # Store initial poses to compute neutral init_vision_pose
        if len(self.vision_state.init_vision_pose_list) < INIT_VISION_POSE_COUNT_MAX:
            self.vision_state.init_vision_pose_list.append(new_message)
            # Compute init after receiving enough
            if len(self.vision_state.init_vision_pose_list) == INIT_VISION_POSE_COUNT_MAX:
                self.vision_state.init_vision_pose = compute_average_pose(
                    self.vision_state.init_vision_pose_list
                )
                self.get_logger().info('Initial Pose Computed!')
        
        # Current pose is offset from the init
        self.vision_state.current_vision_pose = new_message
        
        if DEBUGGING_POSE:
            self.get_logger().info(f'\
                (x:{self.vision_state.current_vision_pose.pose.position.x}, \
                y:{self.vision_state.current_vision_pose.pose.position.y}, \
                z:{self.vision_state.current_vision_pose.pose.position.x}),\
                (x:{self.vision_state.current_vision_pose.pose.orientation.x},\
                y:{self.vision_state.current_vision_pose.pose.orientation.y},\
                z:{self.vision_state.current_vision_pose.pose.orientation.z},\
                w:{self.vision_state.current_vision_pose.pose.orientation.w})')
            
    def handle_vicon_pose(
        self, 
        msg: PoseStamped
    ) -> None:
        if DEBUGGING_LOOP_LOGS:
            self.get_logger().info('Vicon received!')
            
        # Store initial poses to compute neutral init_vicon_pose
        if len(self.vision_state.init_vision_pose_list) < INIT_VISION_POSE_COUNT_MAX:
            self.vision_state.init_vision_pose_list.append(msg)
            # Compute init after receiving enough
            if len(self.vision_state.init_vision_pose_list) == INIT_VISION_POSE_COUNT_MAX:
                self.vision_state.init_vision_pose = compute_average_pose(
                    self.vision_state.init_vision_pose_list
                )
                self.get_logger().info('Initial Pose Computed!')
        
        # Current pose is offset from the init
        self.vision_state.current_vision_pose = msg
        
        if DEBUGGING_POSE:
            self.get_logger().info(f'\
                (x:{self.vision_state.current_vision_pose.pose.position.x}, \
                y:{self.vision_state.current_vision_pose.pose.position.y}, \
                z:{self.vision_state.current_vision_pose.pose.position.x}),\
                (x:{self.vision_state.current_vision_pose.pose.orientation.x},\
                y:{self.vision_state.current_vision_pose.pose.orientation.y},\
                z:{self.vision_state.current_vision_pose.pose.orientation.z},\
                w:{self.vision_state.current_vision_pose.pose.orientation.w})')

    def handle_mavros_state(
        self, 
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