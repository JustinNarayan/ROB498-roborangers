from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from std_srvs.srv import Trigger

from pose_utils import transform_realsense_pose_to_vicon_frame, transform_cv_target_to_world_frame, xy_distance
from constants import (
    TargetType, CURRENT_TARGET_TYPE,
    CV_VICON_POSITION_AGREEMENT_THRESHOLD,
    MissionState,
)

###############################################
#               H A N D L E R S               #
###############################################

class HandlersMixin:
    """
    All ROS subscription and service callback implementations.
    Mixed into CommNode via multiple inheritance — methods have full access
    to `self` (node state, vision_state, loggers, etc.).
    """

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def handle_launch(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self.get_logger().info('Launch Requested.')
        self.launch_requested = True
        response.success = True
        return response

    def handle_test(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self.get_logger().info('Test Requested.')
        self.test_requested = True
        response.success = True
        return response

    def handle_land(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """
        Graceful land: navigate home first, then land.
        Sets land_requested so the FSM transitions to GOING_HOME -> LANDING.
        """
        self.get_logger().info('Land Requested — returning home before landing.')
        self.land_requested = True
        response.success = True
        return response

    def handle_abort(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """
        Immediate land in place — skip going home.
        """
        self.get_logger().info('Abort Requested — landing immediately.')
        self.abort_requested = True
        response.success = True
        return response

    def handle_overhead(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """
        Toggle the OVERHEAD state.

        - If a valid target is currently known, the drone transitions to OVERHEAD
          so it positions its capture net directly above the target x/y position.
        - If already in OVERHEAD, this call toggles *back* to SURVEYING.
        - If called when there is no valid target (and not already OVERHEAD),
          the request is rejected with an explanatory message.
        """
        if self.mission_state == MissionState.OVERHEAD:
            # Second call: return to survey mode
            self.get_logger().info(
                'Overhead Requested — already OVERHEAD, returning to SURVEYING.'
            )
            self.overhead_requested = True   # FSM will handle the toggle
            response.success = True
            response.message = 'Leaving OVERHEAD, returning to SURVEYING.'
        elif self.target_state.has_valid_target():
            self.get_logger().info('Overhead Requested — valid target found, entering OVERHEAD.')
            self.overhead_requested = True
            response.success = True
            response.message = 'Entering OVERHEAD mode.'
        else:
            self.get_logger().warn(
                'Overhead Requested — REJECTED: no valid target currently detected. '
                'Ensure a target pose is being published and is not stale.'
            )
            response.success = False
            response.message = (
                'No valid target detected. '
                'OVERHEAD mode requires an active, non-stale target pose.'
            )
        return response

    # ------------------------------------------------------------------
    # Vision pose handlers
    # ------------------------------------------------------------------

    def handle_camera_pose(self, msg: Odometry) -> None:
        # Extract message data for pose
        camera_pose = PoseStamped()
        camera_pose.header.stamp    = msg.header.stamp
        camera_pose.header.frame_id = msg.header.frame_id
        camera_pose.pose            = msg.pose.pose

        # Convert to Vicon frame
        transformed_camera_pose = transform_realsense_pose_to_vicon_frame(camera_pose, self.vision_state.get_static_pos_transform())

        # Feed into vision state (handles fallback logic internally)
        self.vision_state.update_realsense_pose(transformed_camera_pose)

    def handle_vicon_pose(self, msg: PoseStamped) -> None:
        # Extract message data for pose
        vicon_pose = PoseStamped()
        vicon_pose.header.stamp    = msg.header.stamp
        vicon_pose.header.frame_id = msg.header.frame_id
        vicon_pose.pose            = msg.pose

        # Feed into vision state (handles source selection internally)
        self.vision_state.update_vicon_pose(vicon_pose)

    # ------------------------------------------------------------------
    # Target pose handler
    # ------------------------------------------------------------------

    def handle_target_pose(self, msg: PoseStamped) -> None:
        if CURRENT_TARGET_TYPE is TargetType.COMPUTER_VISION:
            # The CV pipeline publishes in the left-fisheye camera frame.
            # Transform into the global world frame before handing to TargetState.
            world_pose = transform_cv_target_to_world_frame(
                msg,
                self.vision_state.current_vision_pose,
            )
            self.target_state.update(world_pose)
            self.get_logger().info(
                f'tx: {world_pose.pose.position.x}, '
                f'ty: {world_pose.pose.position.y}, '
                f'tz: {world_pose.pose.position.z}'
            )

        elif CURRENT_TARGET_TYPE is TargetType.CV_WITH_VICON_VALIDATION:
            # Transform CV detection from camera frame to world frame.
            world_pose = transform_cv_target_to_world_frame(
                msg,
                self.vision_state.current_vision_pose,
            )

            # Cross-check against the latest Vicon reading for the RC car.
            # If they disagree in x/y beyond the threshold, reject the frame.
            vicon_target = self.target_state.get_latest_vicon_target()
            if vicon_target is not None:
                disagreement = xy_distance(world_pose, vicon_target)
                if disagreement > CV_VICON_POSITION_AGREEMENT_THRESHOLD:
                    self.get_logger().warn(
                        f'[CV_VALIDATE] CV frame rejected — x/y disagreement with Vicon: '
                        f'{disagreement:.3f} m (threshold {CV_VICON_POSITION_AGREEMENT_THRESHOLD} m). '
                        f'CV: ({world_pose.pose.position.x:.2f}, {world_pose.pose.position.y:.2f}), '
                        f'Vicon: ({vicon_target.pose.position.x:.2f}, {vicon_target.pose.position.y:.2f})'
                    )
                    # Do not update target_state — keep previous valid reading or let it go stale.
                    return
                else:
                    self.get_logger().debug(
                        f'[CV_VALIDATE] CV frame accepted — disagreement: {disagreement:.3f} m'
                    )
            else:
                # No Vicon reference available yet; accept the CV frame unconditionally
                # so the drone is not permanently blind at startup.
                self.get_logger().warn(
                    '[CV_VALIDATE] No Vicon target reference available — accepting CV frame unconditionally.'
                )

            self.target_state.update(world_pose)
            self.get_logger().info(
                f'[CV_VALIDATE] tx: {world_pose.pose.position.x:.2f}, '
                f'ty: {world_pose.pose.position.y:.2f}, '
                f'tz: {world_pose.pose.position.z:.2f}'
            )

        else:
            # VICON: pose is already in the global frame — forward unchanged.
            self.target_state.update(msg)

    # ------------------------------------------------------------------
    # Vicon RC car target pose handler (used for CV validation reference)
    # ------------------------------------------------------------------

    def handle_vicon_target_pose(self, msg: PoseStamped) -> None:
        """
        Receives the raw Vicon pose of the RC car target object and stores it
        in TargetState for use as a ground-truth reference when operating in
        CV_WITH_VICON_VALIDATION mode.

        This handler is only wired up when CURRENT_TARGET_TYPE is
        CV_WITH_VICON_VALIDATION (see CommNode.__init__).
        """
        vicon_pose = PoseStamped()
        vicon_pose.header.stamp    = msg.header.stamp
        vicon_pose.header.frame_id = msg.header.frame_id
        vicon_pose.pose            = msg.pose
        self.target_state.update_vicon_reference(vicon_pose)

    # ------------------------------------------------------------------
    # MAVROS state handler
    # ------------------------------------------------------------------

    def handle_mavros_state(self, msg: State) -> None:
        self.current_mavros_state = msg
