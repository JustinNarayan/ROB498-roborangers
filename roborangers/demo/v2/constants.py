###############################################
#      M I S S I O N   V A R I A B L E S      #
###############################################

from enum import Enum, auto

class MissionType(Enum):
    VICON                   = auto()  # Full reliance on Vicon
    REALSENSE               = auto()  # Full reliance on Realsense
    REALSENSE_WITH_FALLBACK = auto()  # Realsense primary, fault to Vicon on divergence

CURRENT_MISSION = MissionType.REALSENSE_WITH_FALLBACK
PERMIT_MANUAL_OVERRIDE = True # for manual landing
MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT = 10 # m

COMMAND_RATE = 50 # Hz
SUCCESS_RADIUS = 0.4 # m

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
SURVEY_ANGULAR_STEP_RADIANS = 0.785  # ~45 degrees per step

# How long the drone holds each angular step before advancing (nanoseconds)
SURVEY_STEP_HOLD_TIME_NANOSECONDS = 5e9  # 5 second per step

###############################################
#          T A R G E T   T R A C K I N G      #
###############################################

# How long a received target pose is considered valid (nanoseconds)
TARGET_STALENESS_THRESHOLD_NANOSECONDS = 1e9  # 1 second

# Desired height above the target object (m)
TARGET_HOVER_ABOVE = 0.5 # m

# Desired standoff radius in the x/y plane from the target object centre (m)
TARGET_STANDOFF_RADIUS = 0.25 # m

# Rate at which the target pose simulator publishes (Hz)
TARGET_SIM_PUBLISH_RATE = 2  # Hz

###############################################
#              R O S   C O M M S              #
###############################################

DRONE_ID = 'rob498_drone_6'

VICON_TOPIC_NAME        = '/vicon/ROB498_Drone/ROB498_Drone'
REALSENSE_TOPIC_NAME    = '/camera/pose/sample'
TARGET_POSE_TOPIC_NAME  = f'{DRONE_ID}/target/pose'

LAND_SERVICE_NAME   = f'{DRONE_ID}/comm/land'
LAUNCH_SERVICE_NAME = f'{DRONE_ID}/comm/launch'
ABORT_SERVICE_NAME  = f'{DRONE_ID}/comm/abort'
TEST_SERVICE_NAME   = f'{DRONE_ID}/comm/test'

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
    GOING_HOME      = 'GOING_HOME'      # Returning to init hover pose before landing
    LANDING         = 'LANDING'         # Executing MAVROS land command
    MANUAL_OVERRIDE = 'MANUAL_OVERRIDE' # Under manual RC control

###############################################
#            V I S I O N   S T A T E          #
###############################################

INIT_VISION_POSE_COUNT_MAX = 50  # aggregate this many poses on start to determine init pose
