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

    A pose is considered valid when:
      1. It passes `is_valid_target_pose()` (not the all-zero sentinel), AND
      2. It was received within TARGET_STALENESS_THRESHOLD_NANOSECONDS nanoseconds
         of the current ROS clock time.
    """

    def __init__(self):
        self._latest_pose: PoseStamped | None = None
        self._received_time_ns: int | None = None  # ROS time (nanoseconds) of last update

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pose: PoseStamped, current_time_ns: int):
        """
        Called from the target pose subscription callback.

        `current_time_ns` should be `node.get_clock().now().nanoseconds`.
        The timestamp is recorded even for invalid poses so that staleness
        can always be evaluated correctly.
        """
        self._latest_pose = pose
        self._received_time_ns = current_time_ns

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has_valid_target(self, current_time_ns: int) -> bool:
        """
        Returns True only when there is a non-sentinel, non-stale target pose.
        """
        if self._latest_pose is None or self._received_time_ns is None:
            return False

        # Check sentinel value
        if not is_valid_target_pose(self._latest_pose):
            return False

        # Check staleness
        age_ns = current_time_ns - self._received_time_ns
        if age_ns > TARGET_STALENESS_THRESHOLD_NANOSECONDS:
            return False

        return True

    def get_pose(self) -> PoseStamped | None:
        """Return the latest raw pose (may be None or invalid — check has_valid_target first)."""
        return self._latest_pose
