from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseArray
from std_srvs.srv import Trigger

from roborangers.utils.pose_utils import unpack_pose_array, transform_realsense_pose_to_vicon_frame

from constants import DEBUG_ALL_WAYPOINTS

###############################################
#               H A N D L E R S               #
###############################################

class HandlersMixin:
    def handle_waypoints(
        self,
        msg: PoseArray
    ) -> Trigger.Response:
        if self.num_waypoints == 0:
            self.waypoints = unpack_pose_array(msg)
            self.num_waypoints = len(self.waypoints)
            
            if DEBUG_ALL_WAYPOINTS:
                for waypoint in self.waypoints:
                    self.get_logger().info(f'Waypoint: ( \
                        {waypoint.pose.position.x}, \
                        {waypoint.pose.position.y}, \
                        {waypoint.pose.position.z} \
                    )')

    def handle_launch(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Launch Requested.')
        self.launch_requested = True
        response.success = True
        return response

    def handle_test(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Test Requested.')
        self.test_requested = True
        response.success = True
        return response

    def handle_land(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Land Requested.')
        self.land_requested = True
        response.success = True
        return response

    def handle_abort(
        self, 
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        self.get_logger().info('Abort Requested.')
        self.abort_requested = True
        response.success = True
        return response

    def handle_camera_pose(
        self, 
        msg: Odometry
    ) -> None:        
        # Extract message data for pose
        camera_pose = PoseStamped()
        camera_pose.header.stamp = msg.header.stamp
        camera_pose.header.frame_id = msg.header.frame_id
        camera_pose.pose = msg.pose.pose
        
        # Convert to VICON frame
        transformed_camera_pose = transform_realsense_pose_to_vicon_frame(camera_pose)
        
        # Inform the drone's current pose
        # If init pose is not yet initialized, this will do that
        self.vision_state.update_current_pose(transformed_camera_pose)
            
    def handle_vicon_pose(
        self, 
        msg: PoseStamped
    ) -> None:
        # Extract message data for pose
        vicon_pose = PoseStamped()
        vicon_pose.header.stamp = msg.header.stamp
        vicon_pose.header.frame_id = msg.header.frame_id
        vicon_pose.pose = msg.pose
        
        # No conversion to VICON frame necessary, already done
        
        # Inform the drone's current pose
        # If init pose is not yet initialized, this will do that
        self.vision_state.update_current_pose(vicon_pose)

    def handle_mavros_state(
        self, 
        msg: State
    ) -> None:
        self.current_mavros_state = msg
