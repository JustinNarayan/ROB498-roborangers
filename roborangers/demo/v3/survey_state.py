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
    Call `begin(current_pose, current_time_ns)` on the first tick of SURVEYING.
    Call `update(current_time_ns)` each subsequent tick to advance the step.
    Read `get_survey_setpoint(current_vision_pose)` for the commanded pose.

    `get_survey_setpoint` always returns a valid PoseStamped:
    - Before `begin()` is called, it returns the drone's current vision pose
      so the drone holds exactly where it is.
    - After `begin()`, it returns the rotating hover setpoint as normal.
    """

    def __init__(self):
        self._hover_pose: PoseStamped = None   # Position anchor (x, y, z fixed)
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

    def get_survey_setpoint(self, current_vision_pose: PoseStamped) -> PoseStamped:
        """
        Return the PoseStamped the drone should be commanded to.

        If begin() has not yet been called (i.e. the very first tick after
        entering SURVEYING), returns `current_vision_pose` so the drone
        holds its current position rather than rubber-banding anywhere.
        """
        if self._hover_pose is None:
            return current_vision_pose
        return pose_with_yaw(self._hover_pose, self._current_yaw)
