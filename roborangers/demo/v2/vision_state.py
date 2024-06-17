import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

from pose_utils import (
    compute_average_pose,
    distance_poses,
    orientation_divergence_angle,
    per_axis_position_divergence,
    per_axis_orientation_divergence,
)
from constants import (
    INIT_VISION_POSE_COUNT_MAX,
    HOVER_ALTITUDE,
    MissionType,
    CURRENT_MISSION,
    REALSENSE_VICON_POSITION_DIVERGENCE_THRESHOLD,
    REALSENSE_VICON_ORIENTATION_DIVERGENCE_THRESHOLD,
    DEBUG_VISION_DIVERGENCE,
)

###############################################
#            V I S I O N   S T A T E          #
###############################################

class VisionState:
    def __init__(self, logger=None, debug_pos_publisher=None, debug_ori_publisher=None):
        self._logger = logger

        # Optional publishers for per-axis divergence debug topics
        # (Float32MultiArray — set by CommNode if DEBUG_VISION_DIVERGENCE is True)
        self._debug_pos_pub = debug_pos_publisher
        self._debug_ori_pub = debug_ori_publisher

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
        Compare the latest Realsense and Vicon readings on both position and
        orientation.  If either exceeds its threshold, permanently fault to Vicon.

        Orientation comparison uses the geodesic quaternion angle:
            θ = 2 * arccos(|q_a · q_b|)
        (apparently?) This is invariant to the quaternion double-cover (q == -q) and
        gives an abs single scalar in [0, π] representing an angular difference.

        Only relevant in REALSENSE_WITH_FALLBACK mode.
        """
        if self.realsense_faulted:
            # Already faulted — still publish debug zeros if enabled
            if DEBUG_VISION_DIVERGENCE:
                self._publish_debug_divergence(None, None)
            return

        if self._latest_vicon_pose is None or self._latest_realsense_pose is None:
            # Only one source available — publish zeros on all axes
            if DEBUG_VISION_DIVERGENCE:
                self._publish_debug_divergence(None, None)
            return

        pos_divergence = distance_poses(self._latest_realsense_pose, self._latest_vicon_pose)
        ori_divergence = orientation_divergence_angle(self._latest_realsense_pose, self._latest_vicon_pose)

        if DEBUG_VISION_DIVERGENCE:
            self._publish_debug_divergence(self._latest_realsense_pose, self._latest_vicon_pose)

        pos_fault = pos_divergence >= REALSENSE_VICON_POSITION_DIVERGENCE_THRESHOLD
        ori_fault = ori_divergence >= REALSENSE_VICON_ORIENTATION_DIVERGENCE_THRESHOLD

        if pos_fault or ori_fault:
            self.realsense_faulted = True
            if self._logger:
                self._logger.error(
                    f'[VISION FAULT] Realsense/Vicon divergence exceeded threshold — '
                    f'position: {pos_divergence:.3f} m (limit {REALSENSE_VICON_POSITION_DIVERGENCE_THRESHOLD} m), '
                    f'orientation: {np.degrees(ori_divergence):.2f} deg '
                    f'(limit {np.degrees(REALSENSE_VICON_ORIENTATION_DIVERGENCE_THRESHOLD):.2f} deg). '
                    f'Permanently switching to Vicon.'
                )

    def _publish_debug_divergence(self, rs_pose, vc_pose):
        """
        Publish per-axis position and orientation divergence (Realsense - Vicon).
        Publishes zeros on all axes when either source is unavailable.
        """
        if rs_pose is not None and vc_pose is not None:
            dx, dy, dz             = per_axis_position_divergence(rs_pose, vc_pose)
            d_roll, d_pitch, d_yaw = per_axis_orientation_divergence(rs_pose, vc_pose)
        else:
            dx = dy = dz = 0.0
            d_roll = d_pitch = d_yaw = 0.0

        if self._debug_pos_pub is not None:
            msg = Float32MultiArray()
            msg.data = [float(dx), float(dy), float(dz)]
            self._debug_pos_pub.publish(msg)

        if self._debug_ori_pub is not None:
            msg = Float32MultiArray()
            msg.data = [float(d_roll), float(d_pitch), float(d_yaw)]
            self._debug_ori_pub.publish(msg)

    # ------------------------------------------------------------------
    # Pose query helpers used by CommNode
    # ------------------------------------------------------------------

    def get_init_hover_pose(self) -> PoseStamped:
        """Return a pose directly above the init position at HOVER_ALTITUDE."""
        init_hover_pose = PoseStamped()
        if self.init_vision_pose is None:
            return init_hover_pose
        init_hover_pose.pose.position.x    = self.init_vision_pose.pose.position.x
        init_hover_pose.pose.position.y    = self.init_vision_pose.pose.position.y
        init_hover_pose.pose.position.z    = self.init_vision_pose.pose.position.z + HOVER_ALTITUDE
        init_hover_pose.pose.orientation.x = self.init_vision_pose.pose.orientation.x
        init_hover_pose.pose.orientation.y = self.init_vision_pose.pose.orientation.y
        init_hover_pose.pose.orientation.z = self.init_vision_pose.pose.orientation.z
        init_hover_pose.pose.orientation.w = self.init_vision_pose.pose.orientation.w
        return init_hover_pose
