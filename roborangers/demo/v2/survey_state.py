from geometry_msgs.msg import PoseStamped

from pose_utils import get_yaw_from_pose, pose_with_yaw, wrap_angle
from constants import SURVEY_ANGULAR_STEP_RADIANS, SURVEY_STEP_HOLD_TIME_NANOSECONDS

###############################################
#           S U R V E Y   S T A T E           #
###############################################

class SurveyState:
    """
    Manages the drone's survey behaviour: hovering in place while slowly
    rotating about the global +Z axis in discrete angular steps.

    Usage
    -----
    Call `begin(current_pose)` when entering SURVEYING state.
    Call `update(current_time_ns)` each control loop tick to advance the step
    when the hold time has elapsed.
    Read `get_survey_setpoint()` to get the PoseStamped the drone should target.
    """

    def __init__(self):
        self._hover_pose: PoseStamped | None = None   # Position to hold (x, y, z fixed)
        self._current_yaw: float = 0.0                # Current target yaw (radians)
        self._step_start_time_ns: int | None = None   # When the current step began

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self, current_pose: PoseStamped, current_time_ns: int):
        """
        Record the drone's current position and orientation as the survey
        origin and start the first angular step.
        """
        self._hover_pose = current_pose
        self._current_yaw = get_yaw_from_pose(current_pose)
        self._step_start_time_ns = current_time_ns

    def reset(self):
        """Clear survey state (called when leaving SURVEYING)."""
        self._hover_pose = None
        self._current_yaw = 0.0
        self._step_start_time_ns = None

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def update(self, current_time_ns: int):
        """
        Advance to the next angular step if the hold time for the current
        step has elapsed.  Safe to call every control loop tick.
        """
        if self._step_start_time_ns is None:
            return  # begin() not yet called

        elapsed_ns = current_time_ns - self._step_start_time_ns
        if elapsed_ns >= SURVEY_STEP_HOLD_TIME_NANOSECONDS:
            self._current_yaw = wrap_angle(
                self._current_yaw + SURVEY_ANGULAR_STEP_RADIANS
            )
            self._step_start_time_ns = current_time_ns  # reset hold timer

    # ------------------------------------------------------------------
    # Setpoint query
    # ------------------------------------------------------------------

    def get_survey_setpoint(self) -> PoseStamped | None:
        """
        Return the PoseStamped the drone should be commanded to.
        Returns None if begin() has not been called yet.
        """
        if self._hover_pose is None:
            return None
        return pose_with_yaw(self._hover_pose, self._current_yaw)
