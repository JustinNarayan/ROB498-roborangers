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
    """

    def __init__(self):
        self._latest_pose: PoseStamped | None = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pose: PoseStamped):
        """
        Called from the target pose subscription callback.
        The pose's header.stamp is used as the received time.
        """
        self._latest_pose = pose

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
        now_ns       = self.get_clock().now().nanoseconds
        age_ns       = now_ns - msg_time_ns
        if age_ns > TARGET_STALENESS_THRESHOLD_NANOSECONDS:
            return False

        return True

    def get_pose(self) -> PoseStamped | None:
        """Return the latest raw pose (may be None or invalid — check has_valid_target first)."""
        return self._latest_pose
