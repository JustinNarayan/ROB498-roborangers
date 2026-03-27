###############################################
#      M I S S I O N   V A R I A B L E S      #
###############################################

from enum import Enum, auto

class MissionType(Enum):
    VICON = auto()
    REALSENSE = auto()

CURRENT_MISSION = MissionType.REALSENSE
PERMIT_MANUAL_OVERRIDE = True # for manual landing
MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT = 6 # m

COMMAND_RATE = 50 # Hz
SUCCESS_RADIUS = 0.20 # m, smaller than the 40 cm used during the test
LOITER_TIME_NANOSECONDS = 3e9

HOVER_ALTITUDE = 0.5

###############################################
#              D E B U G G I N G              #
###############################################

DEBUG_ALL_WAYPOINTS = True

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
    AWAITING_LAUNCH         = 'AWAITING_LAUNCH'
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
