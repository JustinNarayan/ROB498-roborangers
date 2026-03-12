from geometry_msgs.msg import PoseStamped, PoseArray
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

def compute_average_pose(pose_list):
    """
    Get average PoseStamped from list.
    """
    n = len(pose_list)
    avg_pose = PoseStamped()

    # Positions
    x = np.mean([p.pose.position.x for p in pose_list])
    y = np.mean([p.pose.position.y for p in pose_list])
    z = np.mean([p.pose.position.z for p in pose_list])
    avg_pose.pose.position.x = x
    avg_pose.pose.position.y = y
    avg_pose.pose.position.z = z

    # Orientations
    quaternions = np.array([[p.pose.orientation.x,
                             p.pose.orientation.y,
                             p.pose.orientation.z,
                             p.pose.orientation.w] for p in pose_list])
    q_avg = np.mean(quaternions, axis=0)
    q_avg /= np.linalg.norm(q_avg)
    avg_pose.pose.orientation.x = q_avg[0]
    avg_pose.pose.orientation.y = q_avg[1]
    avg_pose.pose.orientation.z = q_avg[2]
    avg_pose.pose.orientation.w = q_avg[3]

    return avg_pose

def subtract_poses(pose_a: PoseStamped, pose_b: PoseStamped):
    """
    Compute relative pose (a - b).
    """
    result = PoseStamped()

    # Positions
    result.pose.position.x = pose_a.pose.position.x - pose_b.pose.position.x
    result.pose.position.y = pose_a.pose.position.y - pose_b.pose.position.y
    result.pose.position.z = pose_a.pose.position.z - pose_b.pose.position.z

    # Convert orientation to Euler angles
    euler_a = euler_from_quaternion([
        pose_a.pose.orientation.x,
        pose_a.pose.orientation.y,
        pose_a.pose.orientation.z,
        pose_a.pose.orientation.w
    ])
    euler_b = euler_from_quaternion([
        pose_b.pose.orientation.x,
        pose_b.pose.orientation.y,
        pose_b.pose.orientation.z,
        pose_b.pose.orientation.w
    ])
    
    # Get difference as quaternion
    euler_rel = np.array(euler_a) - np.array(euler_b)
    quat_rel = quaternion_from_euler(*euler_rel)
    result.pose.orientation.x = quat_rel[0]
    result.pose.orientation.y = quat_rel[1]
    result.pose.orientation.z = quat_rel[2]
    result.pose.orientation.w = quat_rel[3]

    return result

def distance_poses(pose_a: PoseStamped, pose_b: PoseStamped):
    """
    Compute pose distance (a - b).
    """
    
    # Positions
    delta_x = pose_a.pose.position.x - pose_b.pose.position.x
    delta_y = pose_a.pose.position.y - pose_b.pose.position.y
    delta_z = pose_a.pose.position.z - pose_b.pose.position.z
    distance = (delta_x**2 + delta_y**2 + delta_z**2)**(1/2)

    return distance

def transform_realsense_pose_to_vicon_frame(pose_realsense: PoseStamped):
    # Define transform from vicon to realsense in the map frame
    vicon_to_realsense = PoseStamped()
    vicon_to_realsense.header.frame_id = 'map'
    
    # Position
    vicon_to_realsense.pose.position.x = 0.23 # 0.23 "forward" from vicon to realsense
    vicon_to_realsense.pose.position.y = 0.0
    vicon_to_realsense.pose.position.z = -0.12 # 0.12 "down" from vicon to realsense
    
    # Default Quaternion
    vicon_to_realsense.pose.orientation.x = 0.0
    vicon_to_realsense.pose.orientation.y = 0.0
    vicon_to_realsense.pose.orientation.z = 0.0
    vicon_to_realsense.pose.orientation.w = 1.0
    
    # Subtract this transform
    # Now, (0, 0, 0) in the realsense frame is the vicon origin in the map frame
    shifted_realsense = subtract_poses(pose_realsense, vicon_to_realsense)
    shifted_realsense.header.frame_id = 'map'
    
    # Return shifted pose
    return shifted_realsense

def unpack_pose_array(pose_array: PoseArray):
    pose_stamped_list = []

    for pose in pose_array.poses:
        ps = PoseStamped()
        ps.pose = pose
        pose_stamped_list.append(ps)
    
    return pose_stamped_list