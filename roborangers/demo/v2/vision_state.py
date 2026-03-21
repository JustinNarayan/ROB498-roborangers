from geometry_msgs.msg import PoseStamped

from pose_utils import compute_average_pose, distance_poses
from constants import (
    INIT_VISION_POSE_COUNT_MAX,
    HOVER_ALTITUDE,
    MissionType,
    CURRENT_MISSION,
    REALSENSE_VICON_DIVERGENCE_THRESHOLD,
)

###############################################
#            V I S I O N   S T A T E          #
###############################################

class VisionState:
    def __init__(self, logger=None):
        self._logger = logger

        # Averaged pose computed at startup
        self.init_vision_pose_list = []
        self.init_vision_pose: PoseStamped | None = None

        # Latest fused pose (whichever source is currently trusted)
        self.current_vision_pose = PoseStamped()

        # Raw latest readings from each sensor (Vicon frame)
        self._latest_vicon_pose: PoseStamped | None = None
        self._latest_realsense_pose: PoseStamped | None = None  # already transformed

        # Fallback state: once True the node has faulted from Realsense -> Vicon
        self.realsense_faulted = False

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def is_init_pose_computed(self) -> bool:
        return self.init_vision_pose is not None

    def is_using_realsense(self) -> bool:
        """True when Realsense is the active (non-faulted) source."""
        if CURRENT_MISSION is MissionType.REALSENSE:
            return True
        if CURRENT_MISSION is MissionType.REALSENSE_WITH_FALLBACK:
            return not self.realsense_faulted
        return False

    # ------------------------------------------------------------------
    # Pose update entry-points (called by handlers)
    # ------------------------------------------------------------------

    def update_vicon_pose(self, pose: PoseStamped):
        """Called whenever a new Vicon reading arrives."""
        self._latest_vicon_pose = pose

        if CURRENT_MISSION is MissionType.VICON:
            # Vicon is the only source — use it directly
            self._set_current_pose(pose)

        elif CURRENT_MISSION is MissionType.REALSENSE_WITH_FALLBACK:
            # Always check for divergence when both readings are available
            self._check_divergence()

            if self.realsense_faulted:
                # Faulted: trust Vicon
                self._set_current_pose(pose)

    def update_realsense_pose(self, pose: PoseStamped):
        """Called whenever a new Realsense reading arrives (already in Vicon frame)."""
        self._latest_realsense_pose = pose

        if CURRENT_MISSION is MissionType.REALSENSE:
            # Realsense is the only source — use it directly
            self._set_current_pose(pose)

        elif CURRENT_MISSION is MissionType.REALSENSE_WITH_FALLBACK:
            self._check_divergence()

            if not self.realsense_faulted:
                # Still trusting Realsense
                self._set_current_pose(pose)
            # If faulted, Vicon updates will drive current_vision_pose instead

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_current_pose(self, pose: PoseStamped):
        """Update the live pose and accumulate init samples if still needed."""
        self.current_vision_pose = pose
        if not self.is_init_pose_computed():
            self._accumulate_init_pose(pose)

    def _accumulate_init_pose(self, pose: PoseStamped):
        self.init_vision_pose_list.append(pose)
        if len(self.init_vision_pose_list) >= INIT_VISION_POSE_COUNT_MAX:
            self.init_vision_pose = compute_average_pose(self.init_vision_pose_list)

    def _check_divergence(self):
        """
        Compare the latest Realsense and Vicon readings.
        If they diverge beyond the threshold, permanently fault to Vicon.
        Only relevant in REALSENSE_WITH_FALLBACK mode.
        """
        if self.realsense_faulted:
            return  # Already faulted — nothing more to check

        if self._latest_vicon_pose is None or self._latest_realsense_pose is None:
            return  # Need both readings before we can compare

        divergence = distance_poses(self._latest_realsense_pose, self._latest_vicon_pose)

        if divergence >= REALSENSE_VICON_DIVERGENCE_THRESHOLD:
            self.realsense_faulted = True
            if self._logger:
                self._logger.error(
                    f'[VISION FAULT] Realsense/Vicon divergence = {divergence:.3f} m '
                    f'(threshold = {REALSENSE_VICON_DIVERGENCE_THRESHOLD} m). '
                    f'Permanently switching to Vicon.'
                )

    # ------------------------------------------------------------------
    # Pose query helpers used by CommNode
    # ------------------------------------------------------------------

    def get_init_hover_pose(self) -> PoseStamped:
        """Return a pose directly above the init position at HOVER_ALTITUDE."""
        init_hover_pose = PoseStamped()
        init_hover_pose.pose.position.x    = self.init_vision_pose.pose.position.x
        init_hover_pose.pose.position.y    = self.init_vision_pose.pose.position.y
        init_hover_pose.pose.position.z    = self.init_vision_pose.pose.position.z + HOVER_ALTITUDE
        init_hover_pose.pose.orientation.x = self.init_vision_pose.pose.orientation.x
        init_hover_pose.pose.orientation.y = self.init_vision_pose.pose.orientation.y
        init_hover_pose.pose.orientation.z = self.init_vision_pose.pose.orientation.z
        init_hover_pose.pose.orientation.w = self.init_vision_pose.pose.orientation.w
        return init_hover_pose
