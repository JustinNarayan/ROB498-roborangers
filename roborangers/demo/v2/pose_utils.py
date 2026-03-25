from geometry_msgs.msg import PoseStamped, PoseArray
from tf_transformations import euler_from_quaternion, quaternion_from_euler, euler_matrix, quaternion_matrix, quaternion_from_matrix
import numpy as np

###############################################
#         P O S E   U T I L I T I E S         #
###############################################

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
    Compute Euclidean distance between two poses (position only).
    """
    delta_x = pose_a.pose.position.x - pose_b.pose.position.x
    delta_y = pose_a.pose.position.y - pose_b.pose.position.y
    delta_z = pose_a.pose.position.z - pose_b.pose.position.z
    distance = (delta_x**2 + delta_y**2 + delta_z**2)**(1/2)
    return distance

###############################################
#   T R A N S F O R M A T I O N   U T I L S   #
###############################################

def make_transform_matrix(x, y, z, roll, pitch, yaw):
    """
    Create a 4x4 homogeneous transform matrix from translation and RPY.
    """
    T = euler_matrix(roll, pitch, yaw)  # rotation
    T[0:3, 3] = [x, y, z]                   # translation
    return T

def pose_to_matrix(pose: PoseStamped):
    """
    Convert PoseStamped to 4x4 transform matrix.
    """
    pos = pose.pose.position
    ori = pose.pose.orientation

    T = quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
    T[0:3, 3] = [pos.x, pos.y, pos.z]
    return T

def matrix_to_pose(T, frame_id="map"):
    """
    Convert 4x4 transform matrix back to PoseStamped.
    """
    pose = PoseStamped()
    pose.header.frame_id = frame_id

    trans = T[0:3, 3]
    quat = quaternion_from_matrix(T)

    pose.pose.position.x = trans[0]
    pose.pose.position.y = trans[1]
    pose.pose.position.z = trans[2]

    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose

def transform_realsense_pose_to_vicon_frame(pose_realsense: PoseStamped):
    """
    Shift Realsense pose into the Vicon/world frame by subtracting the
    known rigid-body offset between the two sensors.
    """
    # ===================== ADJUST HERE =====================
    tx = 0 # -X
    ty = 0     # Y
    tz = 0   # +Z

    roll  = 0.0
    pitch = -3.14/4 # -pi/4
    yaw   = 0.0
    # ======================================================
    
    # We want REALSENSE → VICON
    T_rs_to_vicon = make_transform_matrix(tx, ty, tz, roll, pitch, yaw)

    # Convert input pose to matrix
    T_pose_rs = pose_to_matrix(pose_realsense)

    # Apply transform
    T_pose_vicon = T_rs_to_vicon @ T_pose_rs

    # Convert back
    pose_vicon = matrix_to_pose(T_pose_vicon, frame_id="map")

    return pose_vicon

###############################################
#     O R I E N T A T I O N   U T I L S       #
###############################################

def get_yaw_from_pose(pose: PoseStamped) -> float:
    """
    Extract the yaw (rotation about global Z axis) from a PoseStamped, in radians.
    Returns a value in [-pi, pi].
    """
    q = [
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ]
    _, _, yaw = euler_from_quaternion(q)
    return yaw

def pose_with_yaw(reference_pose: PoseStamped, yaw: float) -> PoseStamped:
    """
    Return a copy of reference_pose with its orientation replaced by a pure
    yaw rotation (roll=0, pitch=0) about the global Z axis.
    The position is preserved exactly.
    """
    new_pose = PoseStamped()
    new_pose.header = reference_pose.header

    new_pose.pose.position.x = reference_pose.pose.position.x
    new_pose.pose.position.y = reference_pose.pose.position.y
    new_pose.pose.position.z = reference_pose.pose.position.z

    q = quaternion_from_euler(0.0, 0.0, yaw)
    new_pose.pose.orientation.x = q[0]
    new_pose.pose.orientation.y = q[1]
    new_pose.pose.orientation.z = q[2]
    new_pose.pose.orientation.w = q[3]

    return new_pose

def wrap_angle(angle: float) -> float:
    """
    Wrap an angle into [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def orientation_divergence_angle(pose_a: PoseStamped, pose_b: PoseStamped) -> float:
    """
    Compute the geodesic angular distance (radians) between the orientations
    of two poses.

    Method: the angle θ between two unit quaternions q_a and q_b satisfies
        cos(θ/2) = |q_a · q_b|
    so θ = 2 * arccos(|q_a · q_b|).

    The absolute value handles the quaternion double-cover property (q and -q
    represent the same rotation), ensuring the result is always in [0, π].
    Clipping the dot product to [-1, 1] guards against floating-point noise
    that would cause arccos to return NaN.
    """
    q_a = np.array([
        pose_a.pose.orientation.x,
        pose_a.pose.orientation.y,
        pose_a.pose.orientation.z,
        pose_a.pose.orientation.w,
    ])
    q_b = np.array([
        pose_b.pose.orientation.x,
        pose_b.pose.orientation.y,
        pose_b.pose.orientation.z,
        pose_b.pose.orientation.w,
    ])
    dot = float(np.clip(np.abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)

def per_axis_position_divergence(pose_a: PoseStamped, pose_b: PoseStamped):
    """
    Return the signed per-axis position difference (a - b) as a tuple (dx, dy, dz).
    Used for debug publishing.
    """
    dx = pose_a.pose.position.x - pose_b.pose.position.x
    dy = pose_a.pose.position.y - pose_b.pose.position.y
    dz = pose_a.pose.position.z - pose_b.pose.position.z
    return dx, dy, dz

def per_axis_orientation_divergence(pose_a: PoseStamped, pose_b: PoseStamped):
    """
    Return the signed per-axis Euler angle difference (a - b) as a tuple
    (d_roll, d_pitch, d_yaw) in radians.  Used for debug publishing.
    """
    euler_a = euler_from_quaternion([
        pose_a.pose.orientation.x,
        pose_a.pose.orientation.y,
        pose_a.pose.orientation.z,
        pose_a.pose.orientation.w,
    ])
    euler_b = euler_from_quaternion([
        pose_b.pose.orientation.x,
        pose_b.pose.orientation.y,
        pose_b.pose.orientation.z,
        pose_b.pose.orientation.w,
    ])
    d_roll  = wrap_angle(euler_a[0] - euler_b[0])
    d_pitch = wrap_angle(euler_a[1] - euler_b[1])
    d_yaw   = wrap_angle(euler_a[2] - euler_b[2])
    return d_roll, d_pitch, d_yaw

###############################################
#       T A R G E T   T R A C K I N G         #
###############################################

def is_valid_target_pose(pose: PoseStamped) -> bool:
    """
    Determine whether a received target pose is valid (i.e. a real detection).

    Currently: a pose is considered invalid if all position and orientation
    components are exactly zero (the "no target found" sentinel).

    NOTE: This function is intentionally isolated so the validity check can
    be updated later without touching any other logic.
    """
    p = pose.pose.position
    o = pose.pose.orientation

    all_zero = (
        p.x == 0.0 and p.y == 0.0 and p.z == 0.0 and
        o.x == 0.0 and o.y == 0.0 and o.z == 0.0 and o.w == 0.0
    )
    return not all_zero

def compute_tracking_pose(
    drone_pose: PoseStamped,
    target_pose: PoseStamped,
    standoff_radius: float,
    hover_above: float,
) -> PoseStamped:
    """
    Given the drone's current pose and the target object's pose (both in the
    Vicon/world frame), compute a setpoint for the drone such that:

      a) The drone is `hover_above` metres above the target.
      b) The drone is `standoff_radius` metres away from the target in the x/y plane.
      c) The drone directly faces the target.

    Strategy: find the closest point on the standoff circle (radius =
    `standoff_radius`, centred on the target, at height target.z + hover_above)
    to the drone's current x/y position.
    """
    tx = target_pose.pose.position.x
    ty = target_pose.pose.position.y
    tz = target_pose.pose.position.z

    dx = drone_pose.pose.position.x - tx
    dy = drone_pose.pose.position.y - ty

    dist_xy = np.sqrt(dx**2 + dy**2)

    if dist_xy < 1e-6:
        # Drone is directly above the target — pick an arbitrary direction
        angle_to_drone = 0.0
    else:
        angle_to_drone = np.arctan2(dy, dx)

    # Closest point on the standoff circle
    setpoint_x = tx + standoff_radius * np.cos(angle_to_drone)
    setpoint_y = ty + standoff_radius * np.sin(angle_to_drone)
    setpoint_z = tz + hover_above

    # Yaw so the drone faces the target (i.e. points from setpoint toward target)
    yaw_to_target = np.arctan2(ty - setpoint_y, tx - setpoint_x)

    tracking_pose = PoseStamped()
    tracking_pose.header.frame_id = 'map'

    tracking_pose.pose.position.x = setpoint_x
    tracking_pose.pose.position.y = setpoint_y
    tracking_pose.pose.position.z = setpoint_z

    q = quaternion_from_euler(0.0, 0.0, yaw_to_target)
    tracking_pose.pose.orientation.x = q[0]
    tracking_pose.pose.orientation.y = q[1]
    tracking_pose.pose.orientation.z = q[2]
    tracking_pose.pose.orientation.w = q[3]

    return tracking_pose
