from geometry_msgs.msg import PoseStamped
from roborangers.utils.pose_utils import compute_average_pose

from constants import INIT_VISION_POSE_COUNT_MAX, HOVER_ALTITUDE

###############################################
#            V I C O N   S T A T E            #
###############################################

class VisionState:
    def __init__(self):
        # Drone vision data 
        self.init_vision_pose_list = []  # List of PoseStamped for averaging
        self.init_vision_pose = None     # PoseStamped after averaging
        self.current_vision_pose = PoseStamped()
    
    def is_init_pose_computed(self):
        return self.init_vision_pose is not None
        
    def update_init_pose(self, pose: PoseStamped):
        # Still compiling init poses
        self.init_vision_pose_list.append(pose)
        
        # Compute init if all poses recieved
        if len(self.init_vision_pose_list) >= INIT_VISION_POSE_COUNT_MAX:
            self.init_vision_pose = compute_average_pose(
                self.init_vision_pose_list
            )
            
    def update_current_pose(self, pose: PoseStamped):
        self.current_vision_pose = pose
        
        # Check if init needed
        if not self.is_init_pose_computed():
            self.update_init_pose(pose)
    
    def get_init_hover_pose(self):
        # Get the init pose
        init_hover_pose = PoseStamped()
        init_hover_pose.pose.position.x    = self.init_vision_pose.pose.position.x
        init_hover_pose.pose.position.y    = self.init_vision_pose.pose.position.y
        init_hover_pose.pose.position.z    = self.init_vision_pose.pose.position.z + HOVER_ALTITUDE
        init_hover_pose.pose.orientation.x = self.init_vision_pose.pose.orientation.x
        init_hover_pose.pose.orientation.y = self.init_vision_pose.pose.orientation.y
        init_hover_pose.pose.orientation.z = self.init_vision_pose.pose.orientation.z
        init_hover_pose.pose.orientation.w = self.init_vision_pose.pose.orientation.w
        return init_hover_pose
