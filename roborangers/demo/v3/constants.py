###############################################
#      M I S S I O N   V A R I A B L E S      #
###############################################

from enum import Enum, auto

class MissionType(Enum):
    VICON                   = auto()  # Full reliance on Vicon
    REALSENSE               = auto()  # Full reliance on Realsense
    REALSENSE_WITH_FALLBACK = auto()  # Realsense primary, fault to Vicon on divergence

CURRENT_MISSION = MissionType.VICON

###############################################
#        T A R G E T   S O U R C E            #
###############################################

class TargetType(Enum):
    VICON           = auto()  # Target pose is already in the global Vicon/world frame
    COMPUTER_VISION = auto()  # Target pose is in the left-fisheye camera frame and must be transformed
    CV_WITH_VICON_VALIDATION = auto()  # CV primary, but each CV frame is cross-checked against the
                                       # corresponding Vicon RC car reading; frames that diverge too
                                       # far in position are rejected rather than falling back to Vicon

# Set to VICON when using forward_vicon_target_pose.py.
# Set to COMPUTER_VISION when using a CV pipeline that publishes in the camera frame.
# Set to CV_WITH_VICON_VALIDATION to sanity-check CV against Vicon without using Vicon as fallback.
CURRENT_TARGET_TYPE = TargetType.VICON

# Maximum x/y positional disagreement (m) between CV-derived world position and the
# corresponding Vicon ground-truth reading before the CV frame is rejected.
# Only used when CURRENT_TARGET_TYPE == TargetType.CV_WITH_VICON_VALIDATION.
CV_VICON_POSITION_AGREEMENT_THRESHOLD = 0.3  # m

PERMIT_MANUAL_OVERRIDE = True # for manual landing
MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT = 3.0  # m  — beyond this the geo-fence triggers

COMMAND_RATE = 50 # Hz
SUCCESS_RADIUS = 0.3 # m

HOVER_ALTITUDE = 0.5 # m

###############################################
#    V I S I O N   F A L L B A C K            #
###############################################

# Number of subsequent divergent frames to consider diverged
REALSENSE_DIVERGENCE_COUNT = 2

# Whether to attempt adding a learned static pos transform
ATTEMPT_LEARNED_VICON_ALIGN = False

# Max allowed positional distance (m) between Realsense and Vicon before faulting to Vicon
REALSENSE_VICON_POSITION_DIVERGENCE_THRESHOLD    = 0.3   # m

# Max allowed angular distance (rad) between Realsense and Vicon before faulting to Vicon
# Computed as: 2 * arccos(|q_a · q_b|), giving the geodesic angle between two orientations
REALSENSE_VICON_ORIENTATION_DIVERGENCE_THRESHOLD = 0.174 # rad (~10 degrees)

# If True, continuously publish per-axis position and orientation divergence between
# Realsense and Vicon to debug topics. Publishes zeros on any axis when only one
# source is being received.
DEBUG_VISION_DIVERGENCE = True

###############################################
#         S U R V E Y   S T A T E             #
###############################################

# How many radians the drone yaws per step while surveying
SURVEY_ANGULAR_STEP_RADIANS = 0.393  # ~22.5 degrees per step

# How long the drone holds each angular step before advancing (nanoseconds)
SURVEY_STEP_HOLD_TIME_NANOSECONDS = 2e9  # 2 seconds per step

###############################################
#          T A R G E T   T R A C K I N G      #
###############################################

# How long a received target pose is considered valid (nanoseconds)
TARGET_STALENESS_THRESHOLD_NANOSECONDS = 1e9  # 1 second

# Desired height above the target object (m)
TARGET_HOVER_ABOVE = 0.5 # m

# Desired standoff radius in the x/y plane from the target object centre (m)
TARGET_STANDOFF_RADIUS = 0.8 # m
TARGET_CLOSE_ENOUGH_RADIUS = 1.2 # m

# Maximum allowed x/y distance (m) from the drone to the target before tracking is refused.
# Prevents erroneous CV detections from sending the drone far away.
MAX_TRACKING_DISTANCE = 2.0  # m

# Rate at which the target pose simulator publishes (Hz)
TARGET_SIM_PUBLISH_RATE = 2  # Hz

# When True  — normal TRACKING_TARGET behaviour: drone moves to the standoff orbit around the target.
# When False — rotate-only mode: the drone's positional setpoint stays fixed at the init hover
#              pose and only the yaw is updated to face the detected target. Use this to sanity-
#              check CV output before allowing it to command translational motion.
TRACKING_MOVE_TO_TRACK = True

###############################################
#    N E T   F R A M E   O F F S E T          #
###############################################

# Static offset from the drone's origin (MAVROS/Vicon body frame) to the
# centre of the capture net, expressed in the drone's local body frame (m).
#   +X = drone forward, +Y = drone left, +Z = drone up
# These offsets are applied in OVERHEAD mode so the *net* is positioned
# directly above the target rather than the drone origin.
NET_OFFSET_X = 0.14
NET_OFFSET_Y = 0.14

###############################################
#              R O S   C O M M S              #
###############################################

DRONE_ID = 'rob498_drone_6'

VICON_RC_CAR_TOPIC = '/vicon/rob498_rc_car_team6/rob498_rc_car_team6'

VICON_TOPIC_NAME        = '/vicon/ROB498_Drone/ROB498_Drone'
REALSENSE_TOPIC_NAME    = '/camera/pose/sample'
TARGET_POSE_TOPIC_NAME  = f'{DRONE_ID}/target/pose'

LAND_SERVICE_NAME       = f'{DRONE_ID}/comm/land'
LAUNCH_SERVICE_NAME     = f'{DRONE_ID}/comm/launch'
ABORT_SERVICE_NAME      = f'{DRONE_ID}/comm/abort'
TEST_SERVICE_NAME       = f'{DRONE_ID}/comm/test'
OVERHEAD_SERVICE_NAME   = f'{DRONE_ID}/comm/overhead'

DEBUG_VISION_DIVERGENCE_POSITION_TOPIC_NAME    = f'{DRONE_ID}/debug/vision_divergence/position'
DEBUG_VISION_DIVERGENCE_ORIENTATION_TOPIC_NAME = f'{DRONE_ID}/debug/vision_divergence/orientation'

MAVROS_STATE_TOPIC_NAME         = '/mavros/state'
MAVROS_SETPOINT_TOPIC_NAME      = '/mavros/setpoint_position/local'
MAVROS_VISION_POSE_TOPIC_NAME   = '/mavros/vision_pose/pose'

QOS_DEPTH       = 10        # number of messages to store
OFFBOARD_MODE   = 'OFFBOARD'
ALTITUDE_MODE   = 'ALTCTL'

###############################################
#          M I S S I O N   S T A T E          #
###############################################

class MissionState(Enum):
    INITIALIZING    = 'INITIALIZING'    # Computing init pose from vision sensor
    AWAITING_LAUNCH = 'AWAITING_LAUNCH' # Waiting for /launch service call
    AWAITING_TEST   = 'AWAITING_TEST'   # Waiting for /test service call
    SURVEYING       = 'SURVEYING'       # Rotating in place, searching for target
    TRACKING_TARGET = 'TRACKING_TARGET' # Navigating to standoff pose around target
    OVERHEAD        = 'OVERHEAD'        # Positioning net directly above the current target
    GOING_HOME      = 'GOING_HOME'      # Returning to init hover pose before landing
    LANDING         = 'LANDING'         # Executing MAVROS land command
    MANUAL_OVERRIDE = 'MANUAL_OVERRIDE' # Under manual RC control
    GEO_FENCE_HOLD  = 'GEO_FENCE_HOLD' # Emergency: >MAX_EMERGENCY_LAND_DISTANCE from home; hover in place

###############################################
#            V I S I O N   S T A T E          #
###############################################

INIT_VISION_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose
