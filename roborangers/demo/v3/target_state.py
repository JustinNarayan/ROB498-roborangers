from rclpy.clock import Clock
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped

from pose_utils import is_valid_target_pose
from constants import TARGET_STALENESS_THRESHOLD_NANOSECONDS

###############################################
#           T A R G E T   S T A T E           #
###############################################

class TargetState:
    """
    Holds the most recently received target pose and tracks whether it is
    still fresh enough to act on.

    Staleness is evaluated by comparing the timestamp embedded in the
    PoseStamped header (msg.header.stamp) against the current ROS clock time.
    This assumes the /target/pose publisher stamps messages accurately.

    A clock reference is injected at construction so that get_clock().now()
    is never written inside this class — all clock access goes through the
    single Clock object owned by CommNode.

    A pose is considered valid when:
      1. It passes `is_valid_target_pose()` (not the all-zero sentinel), AND
      2. Its header stamp is within TARGET_STALENESS_THRESHOLD_NANOSECONDS of now.

    CV_WITH_VICON_VALIDATION support
    ---------------------------------
    When operating in CV_WITH_VICON_VALIDATION mode CommNode also calls
    `update_vicon_reference()` each time a fresh Vicon RC car reading arrives.
    Handlers.handle_target_pose() reads `get_latest_vicon_target()` to
    cross-check the transformed CV position before accepting it.
    """

    def __init__(self, clock):
        self._latest_pose: PoseStamped = None
        self._latest_vicon_reference: PoseStamped = None  # ground-truth for CV validation
        self.clock = clock

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pose: PoseStamped):
        """
        Called from the target pose subscription callback (CV or Vicon path).
        The pose's header.stamp is used as the received time for staleness checks.
        """
        self._latest_pose = pose

    def update_vicon_reference(self, pose: PoseStamped):
        """
        Store the latest Vicon ground-truth reading for the RC car target.
        Only used in CV_WITH_VICON_VALIDATION mode.
        """
        self._latest_vicon_reference = pose

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has_valid_target(self) -> bool:
        """
        Returns True only when there is a non-sentinel, non-stale target pose.
        """
        if self._latest_pose is None:
            return False

        # Check sentinel value
        if not is_valid_target_pose(self._latest_pose):
            return False

        # Check staleness against msg timestamp
        msg_time_ns  = Time.from_msg(self._latest_pose.header.stamp).nanoseconds
        now_ns       = self.clock.now().nanoseconds
        age_ns       = now_ns - msg_time_ns
        if age_ns > TARGET_STALENESS_THRESHOLD_NANOSECONDS:
            return False

        return True

    def get_pose(self) -> PoseStamped:
        """Return the latest raw pose (may be None or invalid — check has_valid_target first)."""
        return self._latest_pose

    def get_latest_vicon_target(self) -> PoseStamped:
        """
        Return the most recent Vicon ground-truth reading for the RC car.
        May be None if no reading has been received yet.
        Used by handlers to validate CV frames in CV_WITH_VICON_VALIDATION mode.
        """
        return self._latest_vicon_reference
