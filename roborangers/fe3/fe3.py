#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum, auto

# Services and Clients
from std_srvs.srv import Trigger
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# Messages
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State

# Quality of Service for subscriptions
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Math utilities
from roborangers.utils.pose_utils import \
    compute_average_pose, subtract_poses, distance_poses, unpack_pose_array, transform_realsense_pose_to_vicon_frame

###############################################
#               C O M M A N D S               #
###############################################

'''
ros2 service call /rob498_drone_6/comm/land std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/launch std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/abort std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/test std_srvs/srv/Trigger {}

ros2 topic echo /rob498_drone_6/comm/waypoints

header:
  stamp:
    sec: 1772809884
    nanosec: 428233609
  frame_id: vicon/world
poses:
- position:
    x: -2.3
    y: 2.3
    z: 0.5
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
- < more poses >

'''

###############################################
#      M I S S I O N   V A R I A B L E S      #
###############################################

class MissionType(Enum):
    VICON = auto()
    REALSENSE = auto()

CURRENT_MISSION = MissionType.VICON
PERMIT_MANUAL_OVERRIDE = True # for manual landing
MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT =  6 # m

COMMAND_RATE = 50 # Hz
SUCCESS_RADIUS = 0.20 # m, smaller than the 40 cm used during the test
LOITER_TIME_NANOSECONDS = 3e9

HOVER_ALTITUDE = 0.5

###############################################
#              R O S   C O M M S              #
###############################################

DRONE_ID = 'rob498_drone_6'

VICON_TOPIC_NAME = '/vicon/ROB498_Drone/ROB498_Drone'
REALSENSE_TOPIC_NAME = '/camera/pose/sample'

LAND_SERVICE_NAME = f'{DRONE_ID}/comm/land'
LAUNCH_SERVICE_NAME = f'{DRONE_ID}/comm/launch'
ABORT_SERVICE_NAME = f'{DRONE_ID}/comm/abort'
TEST_SERVICE_NAME = f'{DRONE_ID}/comm/test'
WAYPOINTS_TOPIC_NAME = f'{DRONE_ID}/comm/waypoints'

MAVROS_STATE_TOPIC_NAME = f'/mavros/state'
MAVROS_SETPOINT_TOPIC_NAME = f'/mavros/setpoint_position/local'
MAVROS_VISION_POSE_TOPIC_NAME = f'/mavros/vision_pose/pose'

QOS_DEPTH = 10 # number of messages to store
OFFBOARD_MODE = 'OFFBOARD'
ALTITUDE_MODE = 'ALTCTL'

###############################################
#          M I S S I O N   S T A T E          #
###############################################

class MissionState(Enum):
    INITIALIZING            = 'INITIALIZING'
    AWAITING_LAUNCH         = 'AWAITING_LAUNCH' # Make sure we are AT launch hover altitude?
    AWAITING_TEST           = 'AWAITING_TEST'
    EN_ROUTE_TO_WAYPOINT    = 'EN_ROUTE_TO_WAYPOINT'
    LOITERING_AT_WAYPOINT   = 'LOITERING_AT_WAYPOINT'
    UPDATING_WAYPOINT       = 'UPDATING_WAYPOINT'
    LANDING                 = 'LANDING'
    MANUAL_OVERRIDE         = 'MANUAL_OVERRIDE'

###############################################
#            V I C O N   S T A T E            #
###############################################

INIT_VISION_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose
class VisionState:
    def __init__(self):
        # Drone vision data 
        self.init_vision_pose_list = []  # List of PoseStamped for averaging
        self.init_vision_pose = None     # PoseStamped after averaging
        self.current_vision_pose = PoseStamped()
    
    def is_init_pose_computed(self):
        return self.init_vision_pose is not None
        
    def update_init_pose(self, pose: PoseStamped):
        # Still compiling init poses
        self.init_vision_pose_list.append(pose)
        
        # Compute init if all poses recieved
        if len(self.init_vision_pose_list) >= INIT_VISION_POSE_COUNT_MAX:
            self.init_vision_pose = compute_average_pose(
                self.init_vision_pose_list
            )
            
    def update_current_pose(self, pose: PoseStamped):
        self.current_vision_pose = pose
        
        # Check if init needed
        if not self.is_init_pose_computed():
            self.update_init_pose(pose)
    
    def get_init_hover_pose(self):
        # Get the init pose
        init_hover_pose = PoseStamped()
        init_hover_pose.pose.position.x    = self.init_vision_pose.pose.position.x
        init_hover_pose.pose.position.y    = self.init_vision_pose.pose.position.y
        init_hover_pose.pose.position.z    = self.init_vision_pose.pose.position.z + HOVER_ALTITUDE
        init_hover_pose.pose.orientation.x = self.init_vision_pose.pose.orientation.x
        init_hover_pose.pose.orientation.y = self.init_vision_pose.pose.orientation.y
        init_hover_pose.pose.orientation.z = self.init_vision_pose.pose.orientation.z
        init_hover_pose.pose.orientation.w = self.init_vision_pose.pose.orientation.w
        return init_hover_pose

###############################################
#         C O M M U N I C A T I O N S         #
###############################################

class CommNode(Node):
    def __init__(self):
        super().__init__(DRONE_ID)
        
        ### VISION data
        self.vision_state = VisionState()
        
        ### MISSION data
        self.mission_state = MissionState.INITIALIZING
        self.waypoints = []
        self.num_waypoints = 0
        
        ### State variables
        self.has_got_to_offboard = False # for manual override, prevents auto kick-back to OFFBOARD
        self.current_waypoint = 0
        self.launch_requested = False
        self.test_requested = False
        self.land_requested = False
        self.abort_requested = False
        self.manual_override_requested = False
        self.emergency_stop_requested = False
        self.time_waypoint_reached = None
        
        ### MAVROS State variables
        self.current_mavros_state = State()
        
        ### Control loop information
        self.control_timer = self.create_timer( 1.0 / COMMAND_RATE, self.control_loop )
        
        ### Testing Services
        # Generate callbacks to respond to commands for launch, test, land, abort
        self.srv_launch = self.create_service(
            Trigger, LAUNCH_SERVICE_NAME, self.callback_launch
        )
        self.srv_test = self.create_service(
            Trigger, TEST_SERVICE_NAME, self.callback_test
        )
        self.srv_land = self.create_service(
            Trigger, LAND_SERVICE_NAME, self.callback_land
        )
        self.srv_abort = self.create_service(
            Trigger, ABORT_SERVICE_NAME, self.callback_abort
        )
        
        ### VISION
        if CURRENT_MISSION is MissionType.VICON:
            # Drone subscribes to vision pose
            qos_vicon_pose = QoSProfile(depth=QOS_DEPTH)
            qos_vicon_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_vicon_pose = self.create_subscription(
                PoseStamped, 
                VICON_TOPIC_NAME, 
                self.callback_vicon_pose, 
                qos_vicon_pose
            )
        else: # MissionType.REALSENSE
            # Drone subscribes to Camera pose
            qos_camera_pose = QoSProfile(depth=QOS_DEPTH)
            qos_camera_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_camera_pose = self.create_subscription(
                Odometry, 
                REALSENSE_TOPIC_NAME, 
                self.callback_camera_pose, 
                qos_camera_pose
            )
            
        ### MISSION
        # Drone receives waypoints
        qos_waypoints = QoSProfile(depth=QOS_DEPTH)
        qos_waypoints.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub_waypoints = self.create_subscription(
            PoseArray, 
            WAYPOINTS_TOPIC_NAME, 
            self.callback_waypoints, 
            qos_waypoints
        )
        
        ### MAVROS
        # Drone subscribes to MAVROS state
        qos_mavros_state = QoSProfile(depth=QOS_DEPTH)
        self.sub_mavros_state = self.create_subscription(
            State, MAVROS_STATE_TOPIC_NAME, self.callback_mavros_state, qos_mavros_state
        )
        # Drone publishes target setpoint over MAVROS to flight controller
        qos_mavros_setpoint = QoSProfile(depth=QOS_DEPTH)
        self.pub_mavros_setpoint = self.create_publisher(
            PoseStamped, MAVROS_SETPOINT_TOPIC_NAME, qos_mavros_setpoint
        )
        # Drone publishes Vision pose to vision EKF source for Cube
        qos_mavros_vision_pose = QoSProfile(depth=QOS_DEPTH)
        self.pub_mavros_vision_pose = self.create_publisher(
            PoseStamped, MAVROS_VISION_POSE_TOPIC_NAME, qos_mavros_vision_pose
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
    General utilities
    '''
    def update_pose_header(self, pose: PoseStamped):
        pose.header.frame_id = 'map' # we send everything for MAVROS in 'map' frame
        pose.header.stamp = self.get_clock().now().to_msg()
        return pose
        
    def finished_waypoints(self):
        return self.current_waypoint >= self.num_waypoints
    
    def get_current_target(self):
        if self.finished_waypoints():
            return self.vision_state.get_init_hover_pose() # hovering above init pose
        else:
            return self.waypoints[self.current_waypoint] # normal waypoint
    
    def at_waypoint(self):
        return distance_poses(
            self.get_current_target(), 
            self.vision_state.current_vision_pose
        ) <= SUCCESS_RADIUS
            
    def record_loitering_at_waypoint(self):
        if self.time_waypoint_reached is None:
            self.time_waypoint_reached = self.get_clock().now()
    
    def have_loitered_at_waypoint_long_enough(self):
        current_time = self.get_clock().now()
        time_elapsed_nanoseconds = (current_time - self.time_waypoint_reached).nanoseconds
        return time_elapsed_nanoseconds >= LOITER_TIME_NANOSECONDS

    def update_waypoint(self):
        self.time_waypoint_reached = None # reset time
        self.current_waypoint += 1 # proceed to next
    
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
  
    def callback_camera_pose(
        self, 
        msg: Odometry
    ) -> None:
        self.handle_camera_pose(msg)
        
    def callback_vicon_pose(
        self, 
        msg: PoseStamped
    ) -> None:
        self.handle_vicon_pose(msg)

    def callback_mavros_state(
        self, 
        msg: State
    ) -> None:
        self.handle_mavros_state(msg)      
    
    def callback_waypoints(
        self,
        msg: PoseArray
    ) -> None:
        self.handle_waypoints(msg)
    
    '''
    Publish utilities
    '''
    def publish_setpoint(
        self, pose: PoseStamped = None
    ):
        # Initialize pose to publish
        setpoint_pose = PoseStamped()
        
        # Copy over pose data, immutable elements only
        # If no pose is provided, a default home pose is published
        setpoint_pose.pose.position.x    = 0 if pose is None else pose.pose.position.x
        setpoint_pose.pose.position.y    = 0 if pose is None else pose.pose.position.y
        setpoint_pose.pose.position.z    = 0 if pose is None else pose.pose.position.z
        setpoint_pose.pose.orientation.x = 0 if pose is None else pose.pose.orientation.x
        setpoint_pose.pose.orientation.y = 0 if pose is None else pose.pose.orientation.y
        setpoint_pose.pose.orientation.z = 0 if pose is None else pose.pose.orientation.z
        setpoint_pose.pose.orientation.w = 1 if pose is None else pose.pose.orientation.w
        
        # Update header -> okay to mutate setpoint pose
        setpoint_pose = self.update_pose_header(setpoint_pose)
        
        # Publish
        self.pub_mavros_setpoint.publish(setpoint_pose)
        
    def publish_vision_pose(
        self
    ):
        # Initialize pose to publish
        redirected_pose = PoseStamped()
        
        # Copy over pose data, prevents mutation
        redirected_pose.pose = self.vision_state.current_vision_pose.pose
        redirected_pose.header = self.vision_state.current_vision_pose.header
        
        # Update header -> okay to mutate redicted pose
        redirected_pose = self.update_pose_header(redirected_pose)
        
        # Publish
        self.pub_mavros_vision_pose.publish(redirected_pose)
    
    '''
    Client request commands
    '''
    def request_offboard_mode(self):
        self.get_logger().info('Requesting offboard')
        req = SetMode.Request()
        req.custom_mode = OFFBOARD_MODE
        self.cli_set_mode.call_async(req) # Ok not to check await response
        self.get_logger().info('Requested offboard')
    
    def request_altitude_mode(self):
        self.get_logger().info('Requesting altitude')
        req = SetMode.Request()
        req.custom_mode = ALTITUDE_MODE
        self.cli_set_mode.call_async(req) # Ok not to check await response
        self.get_logger().info('Requested altitude')
    
    def request_arm(self):
        self.get_logger().info('Requesting arm')
        req = CommandBool.Request()
        req.value = True
        self.cli_arming.call_async(req) # Ok not to check await response
        self.get_logger().info('Requested arm')
    
    def request_land(self):
        self.get_logger().info('Requesting land')
        req = CommandTOL.Request()
        self.cli_land.call_async(req) # Ok not to check await response
        self.get_logger().info('Requested land')
    
    '''
    State transition logic
    
    INITIALIZING            ==> Computing init pose
    AWAITING_LAUNCH         ==> Waiting for /launch
    AWAITING_TEST           ==> Waiting for /test
    EN_ROUTE_TO_WAYPOINT    ==> Moving to next waypoint
    LOITERING_AT_WAYPOINT   ==> Reached waypoint
    UPDATING_WAYPOINT       ==> No processing should occur in state transition, rather exec loop
    LANDING                 ==> Landing in place
    MANUAL_OVERRIDE         ==> Under manual control
    '''
    def update_state(self):
        '''
        Only ONE state transiiton EVER occurs per-loop with FSMs
        This makes sure all state actions can be executed at least once.
        Transitioning between multiple states per loop is DANGEROUS.
        '''
        
        initial_state = self.mission_state
        
        ### Universal Commands
        # These will take control of the drone no matter what the current state
        if self.land_requested:
            self.mission_state = MissionState.LANDING
        elif self.abort_requested:
            self.mission_state = MissionState.LANDING
        elif self.emergency_stop_requested:
            self.mission_state = MissionState.LANDING
        elif self.manual_override_requested:
            self.mission_state = MissionState.MANUAL_OVERRIDE
        else:
            ### Ordinary commands
            # INITIALIZING
            if self.mission_state == MissionState.INITIALIZING:
                # -> AWAITING_LAUNCH
                if self.vision_state.is_init_pose_computed():
                    self.mission_state = MissionState.AWAITING_LAUNCH
            
            # AWAITING_LAUNCH
            elif self.mission_state == MissionState.AWAITING_LAUNCH:
                # -> AWAITING_TEST
                if self.launch_requested:
                    self.mission_state = MissionState.AWAITING_TEST
            
            # AWAITING_TEST
            elif self.mission_state == MissionState.AWAITING_TEST:
                # -> EN_ROUTE_TO_WAYPOINT
                if self.test_requested:
                    self.mission_state = MissionState.EN_ROUTE_TO_WAYPOINT
            
            # EN_ROUTE_TO_WAYPOINT
            elif self.mission_state == MissionState.EN_ROUTE_TO_WAYPOINT:
                if self.at_waypoint():
                    # -> LANDING
                    if self.finished_waypoints():
                        self.mission_state = MissionState.LANDING
                        
                    # -> LOITERING_AT_WAYPOINT
                    else:
                        self.mission_state = MissionState.LOITERING_AT_WAYPOINT
            
            # LOITERING_AT_WAYPOINT
            elif self.mission_state == MissionState.LOITERING_AT_WAYPOINT:
                # -> UPDATING_WAYPOINT
                if self.have_loitered_at_waypoint_long_enough():
                    self.mission_state = MissionState.UPDATING_WAYPOINT

            # UPDATING_WAYPOINT
            elif self.mission_state == MissionState.UPDATING_WAYPOINT:
                # Always proceed, the one loop is enough for processing
                self.mission_state = MissionState.EN_ROUTE_TO_WAYPOINT
                
        # Report if state changed
        final_state = self.mission_state
        if initial_state != final_state:
            self.get_logger().info(f'Transition! {initial_state} -> {final_state}')
    
    def execute_state(self):
        ### LANDING
        if self.mission_state in [
            MissionState.LANDING
        ]:
            self.request_land()
        
        ### CLEAR FLAGS
        # We only really are about launch and test flags
        # Land, Abort, Emergency Stop, Manual Override can all persist since they end flight
        if self.mission_state != MissionState.AWAITING_LAUNCH:
            self.launch_requested = False
        if self.mission_state != MissionState.AWAITING_TEST:
            self.test_requested = False
        
        ### PUBLISHING
        # Publish setpoint
        if self.mission_state in [
            MissionState.AWAITING_LAUNCH, 
            MissionState.AWAITING_TEST
        ]:
            self.publish_setpoint(self.vision_state.get_init_hover_pose())
        elif self.mission_state in [
            MissionState.EN_ROUTE_TO_WAYPOINT,
            MissionState.LOITERING_AT_WAYPOINT,
            MissionState.UPDATING_WAYPOINT,
            MissionState.LANDING,
            MissionState.MANUAL_OVERRIDE
        ]:
            self.publish_setpoint(self.get_current_target())
            
        # Redirect vision pose
        if self.mission_state != MissionState.INITIALIZING:
            self.publish_vision_pose()
        
            # Monitor for emergency stop
            # Need valid init pose for this
            self.emergency_stop_requested = self.emergency_stop_requested or \
                distance_poses(
                    self.vision_state.current_vision_pose, self.vision_state.init_vision_pose
                ) >= MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT
            
        ### MAVROS STATE CHECKS
        _is_connected = self.current_mavros_state.connected
        _is_armed = self.current_mavros_state.armed
        _is_offboard = self.current_mavros_state.mode == OFFBOARD_MODE
        if not _is_connected:
            return # Not connected
            
        ### GET TO OFFBOARD
        if self.mission_state not in [
            MissionState.INITIALIZING,
            MissionState.AWAITING_LAUNCH,
            MissionState.LANDING,
            MissionState.MANUAL_OVERRIDE
        ]:
            # Arm
            if not _is_armed:                
                # Set to Altitude (Manual) mode before arming
                if _is_offboard:
                    self.request_altitude_mode()
                else:
                    self.request_arm()
            # Get to offboard
            else:
                if not _is_offboard:
                    # Monitor for manual override
                    self.manual_override_requested = \
                        self.has_got_to_offboard and PERMIT_MANUAL_OVERRIDE
                    # Try for offboard if not manually overriden
                    if not self.manual_override_requested:
                        # OK if we get rejected due to setpoints not published
                        # Keep trying until we get it
                        self.request_offboard_mode()
                        
                else:
                    # Record we have entered offboard
                    self.has_got_to_offboard = True
        else:
            # Record we have exited offboard
            self.has_got_to_offboard = False
        
        ### WAYPOINT MANAGEMENT
        if self.mission_state == MissionState.LOITERING_AT_WAYPOINT:
            self.record_loitering_at_waypoint()
        
        if self.mission_state == MissionState.UPDATING_WAYPOINT:
            self.update_waypoint()
                
    '''
    Drone continuous control logic, running at COMMAND_RATE
    '''
    def control_loop(self):
        # Manage state transitions
        self.update_state()
        
        # Actuate drone based on state
        self.execute_state()

###############################################
#               H A N D L E R S               #
###############################################

    def handle_waypoints(
        self,
        msg: PoseArray
    ) -> Trigger.Response:
        self.waypoints = unpack_pose_array(msg)
        self.num_waypoints = len(self.waypoints)
    
    def handle_launch(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Launch Requested.')
        self.launch_requested = True
        response.success = True
        return response

    def handle_test(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Test Requested.')
        self.test_requested = True
        response.success = True
        return response

    def handle_land(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Land Requested.')
        self.land_requested = True
        response.success = True
        return response

    def handle_abort(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Abort Requested.')
        self.abort_requested = True
        response.success = True
        return response

    def handle_camera_pose(
        self, 
        msg: Odometry
    ) -> None:        
        # Extract message data for pose
        camera_pose = PoseStamped()
        camera_pose.header.stamp = msg.header.stamp
        camera_pose.header.frame_id = msg.header.frame_id
        camera_pose.pose = msg.pose.pose
        
        # Convert to VICON frame
        transformed_camera_pose = transform_realsense_pose_to_vicon_frame(camera_pose)
        
        # Inform the drone's current pose
        # If init pose is not yet initialized, this will do that
        self.vision_state.update_current_pose(transformed_camera_pose)
            
    def handle_vicon_pose(
        self, 
        msg: PoseStamped
    ) -> None:
        # Extract message data for pose
        vicon_pose = PoseStamped()
        vicon_pose.header.stamp = msg.header.stamp
        vicon_pose.header.frame_id = msg.header.frame_id
        vicon_pose.pose = msg.pose
        
        # No conversion to VICON frame necessary, already done
        
        # Inform the drone's current pose
        # If init pose is not yet initialized, this will do that
        self.vision_state.update_current_pose(vicon_pose)

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