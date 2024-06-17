from rclpy.node import Node

# Services and Clients
from std_srvs.srv import Trigger
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# Messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from std_msgs.msg import Float32MultiArray

# Quality of Service
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Math utilities
from pose_utils import distance_poses, compute_tracking_pose

# Local modules
from constants import (
    DRONE_ID,
    CURRENT_MISSION, MissionType, MissionState,
    PERMIT_MANUAL_OVERRIDE, MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT,
    COMMAND_RATE, SUCCESS_RADIUS,
    VICON_TOPIC_NAME, REALSENSE_TOPIC_NAME, TARGET_POSE_TOPIC_NAME,
    LAND_SERVICE_NAME, LAUNCH_SERVICE_NAME, ABORT_SERVICE_NAME, TEST_SERVICE_NAME,
    MAVROS_STATE_TOPIC_NAME, MAVROS_SETPOINT_TOPIC_NAME, MAVROS_VISION_POSE_TOPIC_NAME,
    QOS_DEPTH, OFFBOARD_MODE, ALTITUDE_MODE,
    TARGET_STANDOFF_RADIUS, TARGET_HOVER_ABOVE,
    DEBUG_VISION_DIVERGENCE,
    DEBUG_VISION_DIVERGENCE_POSITION_TOPIC_NAME,
    DEBUG_VISION_DIVERGENCE_ORIENTATION_TOPIC_NAME,
)
from vision_state import VisionState
from target_state import TargetState
from survey_state import SurveyState
from handlers import HandlersMixin

###############################################
#         C O M M U N I C A T I O N S         #
###############################################

class CommNode(HandlersMixin, Node):
    def __init__(self):
        super().__init__(DRONE_ID)

        ### Create initial debug publishers for VisionState
        # Publish divergence between Vicon and Realsense
        if DEBUG_VISION_DIVERGENCE:
            self._debug_pos_pub = self.create_publisher(
                Float32MultiArray,
                DEBUG_VISION_DIVERGENCE_POSITION_TOPIC_NAME,
                QoSProfile(depth=QOS_DEPTH)
            )
            self._debug_ori_pub = self.create_publisher(
                Float32MultiArray,
                DEBUG_VISION_DIVERGENCE_ORIENTATION_TOPIC_NAME,
                QoSProfile(depth=QOS_DEPTH)
            )
        else:
            self._debug_pos_pub = None
            self._debug_ori_pub = None

        ### Sub-states
        self.vision_state = VisionState(
            logger=self.get_logger(),
            debug_pos_publisher=self._debug_pos_pub,
            debug_ori_publisher=self._debug_ori_pub,
        )
        self.target_state = TargetState(self.get_clock())
        self.survey_state = SurveyState()

        ### Mission flags
        self.mission_state          = MissionState.INITIALIZING
        self.launch_requested       = False
        self.test_requested         = False
        self.land_requested         = False   # graceful: go home first
        self.abort_requested        = False   # immediate: land in place
        self.manual_override_requested  = False
        self.emergency_stop_requested   = False

        ### MAVROS tracking
        self.current_mavros_state   = State()
        self.has_got_to_offboard    = False

        ### Control loop
        self.control_timer = self.create_timer(1.0 / COMMAND_RATE, self.control_loop)

        ### Services
        self.srv_launch = self.create_service(Trigger, LAUNCH_SERVICE_NAME, self.callback_launch)
        self.srv_test   = self.create_service(Trigger, TEST_SERVICE_NAME,   self.callback_test)
        self.srv_land   = self.create_service(Trigger, LAND_SERVICE_NAME,   self.callback_land)
        self.srv_abort  = self.create_service(Trigger, ABORT_SERVICE_NAME,  self.callback_abort)

        ### Vision subscriptions
        if CURRENT_MISSION in (MissionType.VICON, MissionType.REALSENSE_WITH_FALLBACK):
            # Vicon received
            qos_vicon_pose = QoSProfile(depth=QOS_DEPTH)
            qos_vicon_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_vicon_pose = self.create_subscription(
                PoseStamped, VICON_TOPIC_NAME, self.callback_vicon_pose, qos_vicon_pose
            )

        if CURRENT_MISSION in (MissionType.REALSENSE, MissionType.REALSENSE_WITH_FALLBACK):
            # Realsense received
            qos_camera_pose = QoSProfile(depth=QOS_DEPTH)
            qos_camera_pose.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub_camera_pose = self.create_subscription(
                Odometry, REALSENSE_TOPIC_NAME, self.callback_camera_pose, qos_camera_pose
            )
            
        ### Target pose subscription (always active)
        qos_target_pose = QoSProfile(depth=QOS_DEPTH)
        qos_target_pose.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub_target_pose = self.create_subscription(
            PoseStamped, TARGET_POSE_TOPIC_NAME, self.callback_target_pose, qos_target_pose
        )

        ### MAVROS subscriptions / publishers
        self.sub_mavros_state = self.create_subscription(
            State, MAVROS_STATE_TOPIC_NAME, self.callback_mavros_state,
            QoSProfile(depth=QOS_DEPTH)
        )
        self.pub_mavros_setpoint = self.create_publisher(
            PoseStamped, MAVROS_SETPOINT_TOPIC_NAME, QoSProfile(depth=QOS_DEPTH)
        )
        self.pub_mavros_vision_pose = self.create_publisher(
            PoseStamped, MAVROS_VISION_POSE_TOPIC_NAME, QoSProfile(depth=QOS_DEPTH)
        )

        ### MAVROS clients
        self.cli_set_mode = self.create_client(SetMode,     '/mavros/set_mode')
        self.cli_arming   = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.cli_land     = self.create_client(CommandTOL,  '/mavros/cmd/land')
        self.cli_set_mode.wait_for_service()
        self.cli_arming.wait_for_service()
        self.cli_land.wait_for_service()

    # ==================================================================
    # Callbacks (thin wrappers — logic lives in HandlersMixin)
    # ==================================================================

    def callback_launch(self, request, response):
        return self.handle_launch(request, response)

    def callback_test(self, request, response):
        return self.handle_test(request, response)

    def callback_land(self, request, response):
        return self.handle_land(request, response)

    def callback_abort(self, request, response):
        return self.handle_abort(request, response)

    def callback_camera_pose(self, msg: Odometry):
        self.handle_camera_pose(msg)

    def callback_vicon_pose(self, msg: PoseStamped):
        self.handle_vicon_pose(msg)

    def callback_target_pose(self, msg: PoseStamped):
        self.handle_target_pose(msg)

    def callback_mavros_state(self, msg: State):
        self.handle_mavros_state(msg)

    # ==================================================================
    # General utilities
    # ==================================================================

    def update_pose_header(self, pose: PoseStamped) -> PoseStamped:
        pose.header.frame_id = 'map'
        pose.header.stamp    = self.get_clock().now().to_msg()
        return pose

    def at_pose(self, target: PoseStamped) -> bool:
        return distance_poses(target, self.vision_state.current_vision_pose) <= SUCCESS_RADIUS

    # ------------------------------------------------------------------
    # Continuous output for drone's navigational target
    # ------------------------------------------------------------------

    def get_current_setpoint(self) -> PoseStamped:
        """
        Determine the drone's current current target based on current state
        """
        state = self.mission_state

        if state in (MissionState.INITIALIZING,
                     MissionState.AWAITING_LAUNCH,
                     MissionState.AWAITING_TEST):
            # Hold at init hover while waiting
            return self.vision_state.get_init_hover_pose()

        elif state == MissionState.SURVEYING:
            # Always valid: returns current_vision_pose before begin() fires,
            # then the rotating hover setpoint thereafter
            return self.survey_state.get_survey_setpoint(
                self.vision_state.current_vision_pose
            )

        elif state == MissionState.TRACKING_TARGET:
            target = self.target_state.get_pose()
            if target is not None:
                return compute_tracking_pose(
                    self.vision_state.current_vision_pose,
                    target,
                    TARGET_STANDOFF_RADIUS,
                    TARGET_HOVER_ABOVE,
                )

        elif state == MissionState.GOING_HOME:
            return self.vision_state.get_init_hover_pose()

        elif state in (MissionState.LANDING, MissionState.MANUAL_OVERRIDE):
            return self.vision_state.get_init_hover_pose()

        # Should never reach here, but return to hover pose in worst case
        return self.vision_state.get_init_hover_pose()

    # ==================================================================
    # Publish utilities
    # ==================================================================

    def publish_setpoint(self, pose: PoseStamped):
        setpoint_pose = PoseStamped()
        if pose is None:
            return setpoint_pose
        setpoint_pose.pose.position.x    = pose.pose.position.x
        setpoint_pose.pose.position.y    = pose.pose.position.y
        setpoint_pose.pose.position.z    = pose.pose.position.z
        setpoint_pose.pose.orientation.x = pose.pose.orientation.x
        setpoint_pose.pose.orientation.y = pose.pose.orientation.y
        setpoint_pose.pose.orientation.z = pose.pose.orientation.z
        setpoint_pose.pose.orientation.w = pose.pose.orientation.w
        setpoint_pose = self.update_pose_header(setpoint_pose)
        self.pub_mavros_setpoint.publish(setpoint_pose)

    def publish_vision_pose(self):
        redirected_pose = PoseStamped()
        redirected_pose.pose   = self.vision_state.current_vision_pose.pose
        redirected_pose.header = self.vision_state.current_vision_pose.header
        redirected_pose = self.update_pose_header(redirected_pose)
        self.pub_mavros_vision_pose.publish(redirected_pose)

    # ==================================================================
    # MAVROS client requests
    # ==================================================================

    def request_offboard_mode(self):
        self.get_logger().info('Requesting offboard')
        req = SetMode.Request()
        req.custom_mode = OFFBOARD_MODE
        self.cli_set_mode.call_async(req)

    def request_altitude_mode(self):
        self.get_logger().info('Requesting altitude mode')
        req = SetMode.Request()
        req.custom_mode = ALTITUDE_MODE
        self.cli_set_mode.call_async(req)

    def request_arm(self):
        self.get_logger().info('Requesting arm')
        req = CommandBool.Request()
        req.value = True
        self.cli_arming.call_async(req)

    def request_land(self):
        self.get_logger().info('Requesting land')
        req = CommandTOL.Request()
        self.cli_land.call_async(req)

    # ==================================================================
    # Finite State Machine
    # ==================================================================
    '''
    State machine overview
    ----------------------
    INITIALIZING        -> AWAITING_LAUNCH      (init pose computed)
    AWAITING_LAUNCH     -> AWAITING_TEST         (/launch received)
    AWAITING_TEST       -> SURVEYING             (/test received)
    SURVEYING           -> TRACKING_TARGET       (valid target seen)
    TRACKING_TARGET     -> SURVEYING             (target lost/stale)
    TRACKING_TARGET     -> GOING_HOME            (/land received)
    SURVEYING           -> GOING_HOME            (/land received)
    GOING_HOME          -> LANDING               (at home hover pose)
    * -> LANDING                                 (/abort at any time)
    * -> MANUAL_OVERRIDE                         (RC takeover detected)
    '''

    def update_state(self):
        """
        Advance the FSM by at most ONE transition per tick.
        All guard conditions are evaluated here; side-effects happen in execute_state().
        """
        initial_state = self.mission_state

        # ---- Universal overrides (highest priority) -------------------
        if self.abort_requested:
            self.mission_state = MissionState.LANDING

        elif self.emergency_stop_requested:
            self.mission_state = MissionState.LANDING

        elif self.land_requested:
            # Graceful land: go home first (unless already landing/going home)
            if self.mission_state not in (MissionState.GOING_HOME,
                                          MissionState.LANDING):
                self.mission_state = MissionState.GOING_HOME

        elif self.manual_override_requested:
            self.mission_state = MissionState.MANUAL_OVERRIDE

        else:
            # ---- Ordinary transitions ---------------------------------

            if self.mission_state == MissionState.INITIALIZING:
                if self.vision_state.is_init_pose_computed():
                    self.mission_state = MissionState.AWAITING_LAUNCH

            elif self.mission_state == MissionState.AWAITING_LAUNCH:
                if self.launch_requested:
                    self.mission_state = MissionState.AWAITING_TEST

            elif self.mission_state == MissionState.AWAITING_TEST:
                if self.test_requested:
                    self.mission_state = MissionState.SURVEYING

            elif self.mission_state == MissionState.SURVEYING:
                # When a valid target is recieved, track it
                if self.target_state.has_valid_target():
                    self.mission_state = MissionState.TRACKING_TARGET

            elif self.mission_state == MissionState.TRACKING_TARGET:
                # Return to survey if target is lost or stale
                if not self.target_state.has_valid_target():
                    self.mission_state = MissionState.SURVEYING

            elif self.mission_state == MissionState.GOING_HOME:
                # Land once we are close enough to the home hover pose
                self.land_requested = False # clear flag
                if self.at_pose(self.vision_state.get_init_hover_pose()):
                    self.mission_state = MissionState.LANDING

        # Log any transition
        if initial_state != self.mission_state:
            self.get_logger().info(
                f'Transition! {initial_state.value} -> {self.mission_state.value}'
            )

    # ==================================================================
    # FSM — state execution
    # ==================================================================

    def execute_state(self):
        """
        Perform required state activities each step.
        """
        
        now_ns = self.get_clock().now().nanoseconds

        if self.mission_state == MissionState.LANDING:
            self.request_land()

        # Manage survey rotation
        if self.mission_state == MissionState.SURVEYING:
            if self.survey_state._hover_pose is None:
                # First tick in SURVEYING — lock in the starting pose and time
                self.survey_state.begin(
                    self.vision_state.current_vision_pose, now_ns
                )
            else:
                self.survey_state.update(now_ns)
        else:
            # Reset survey when leaving the state so it starts fresh next time
            self.survey_state.reset()

        # ---- Clear one-time command flags  ------------------------------------
        # Land / abort / emergency flags will persist to ensure drone lands
        if self.mission_state != MissionState.AWAITING_LAUNCH:
            self.launch_requested = False
        if self.mission_state != MissionState.AWAITING_TEST:
            self.test_requested = False

        # ---- Publish setpoint-------------------------------------
        current_setpoint = self.get_current_setpoint()
        self.publish_setpoint(current_setpoint)

        # ---- Publish vision pose (after init) ------------------------
        if self.mission_state != MissionState.INITIALIZING:
            self.publish_vision_pose()

            # Emergency stop check (requires valid init pose)
            self.emergency_stop_requested = self.emergency_stop_requested or (
                distance_poses(
                    self.vision_state.current_vision_pose,
                    self.vision_state.init_vision_pose
                ) >= MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT
            )

        # ---- MAVROS arming / mode management -------------------------
        _is_connected = self.current_mavros_state.connected
        _is_armed     = self.current_mavros_state.armed
        _is_offboard  = self.current_mavros_state.mode == OFFBOARD_MODE

        if not _is_connected:
            return

        # States where we actively want to be in OFFBOARD and armed
        active_flight_states = (
            MissionState.AWAITING_TEST,
            MissionState.SURVEYING,
            MissionState.TRACKING_TARGET,
            MissionState.GOING_HOME,
        )

        if self.mission_state in active_flight_states:
            if not _is_armed:
                if _is_offboard:
                    self.request_altitude_mode()
                else:
                    self.request_arm()
            else:
                if not _is_offboard:
                    self.manual_override_requested = (
                        self.has_got_to_offboard and PERMIT_MANUAL_OVERRIDE
                    )
                    if not self.manual_override_requested:
                        self.request_offboard_mode()
                else:
                    self.has_got_to_offboard = True
        else:
            self.has_got_to_offboard = False

    # ==================================================================
    # Main control loop (runs at COMMAND_RATE Hz)
    # ==================================================================

    def control_loop(self):
        self.update_state()
        self.execute_state()
