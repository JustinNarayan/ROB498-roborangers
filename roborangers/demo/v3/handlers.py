from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from std_srvs.srv import Trigger

from pose_utils import transform_realsense_pose_to_vicon_frame, transform_cv_target_to_world_frame
from constants import TargetType, CURRENT_TARGET_TYPE

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
        else:
            # VICON: pose is already in the global frame — forward unchanged.
            self.target_state.update(msg)

    # ------------------------------------------------------------------
    # MAVROS state handler
    # ------------------------------------------------------------------

    def handle_mavros_state(self, msg: State) -> None:
        self.current_mavros_state = msg
