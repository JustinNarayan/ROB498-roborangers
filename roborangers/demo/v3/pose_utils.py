from geometry_msgs.msg import PoseStamped, Pose
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply, quaternion_conjugate, quaternion_from_matrix
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

def transform_realsense_pose_to_vicon_frame(pose_realsense: PoseStamped, static_pos_transform: Pose):
    """
    Shift Realsense pose into the Vicon/world frame by subtracting the
    known rigid-body offset between the two sensors.
    """
    # Mounting tilt
    # -45 deg about +Y
    q_tilt = quaternion_from_euler(0, -np.pi/4, 0)
    
    # Current pose
    q_raw = [
        pose_realsense.pose.orientation.x,
        pose_realsense.pose.orientation.y,
        pose_realsense.pose.orientation.z,
        pose_realsense.pose.orientation.w
    ]
    
    # Corrected
    q_corrected = quaternion_multiply(
        q_raw,
        q_tilt
    )

    # Apply positional offset from drone -> vicon
    vicon_to_rs = [0.23, 0, -0.12] # x,y,z
    if static_pos_transform is not None:
        vicon_to_rs[0] -= static_pos_transform.position.x
        vicon_to_rs[1] -= static_pos_transform.position.y
        vicon_to_rs[2] -= static_pos_transform.position.z

    result = PoseStamped()
    result.pose.position.x = pose_realsense.pose.position.x - vicon_to_rs[0]
    result.pose.position.y = pose_realsense.pose.position.y - vicon_to_rs[1]
    result.pose.position.z = pose_realsense.pose.position.z - vicon_to_rs[2]
    result.pose.orientation.x = q_corrected[0]
    result.pose.orientation.y = q_corrected[1]
    result.pose.orientation.z = q_corrected[2]
    result.pose.orientation.w = q_corrected[3]
    
    return result
    

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


def transform_cv_target_to_world_frame(
    target_pose_camera: PoseStamped,
    drone_pose: PoseStamped,
) -> PoseStamped:
    """
    Transform a target pose expressed in the left-fisheye camera frame into
    the global Vicon/world frame, given the drone's current world-frame pose.

    Camera-to-drone body offset (in the drone's body frame):
        translation : +0.23 m along body-X, +0.05 m along body-Y, -0.12 m along body-Z
        rotation    : -45 deg about body-Y  (camera tilted downward)

    Camera frame convention (fisheye):
        Z = forward (optical axis)
        X = left
        Y = up

    Steps:
        1. Build the rigid transform from camera frame -> drone body frame.
        2. Apply the drone's world-frame pose to get camera frame -> world frame.
        3. Express the target position in the world frame.
    """
    # ---- 1. CV -> Drone Rotation ----------------------------------------

    # Rotation: -45 deg about body-Y
    q_cam_to_drone = quaternion_from_euler(0.0, -np.pi / 4, 0.0)
    
    # Rotation: From cam to fisheye
    # The camera frame is as follows:
    # cam-X = left (body +Y)
    # cam-Y = up (body +Z)
    # cam-Z = forward (body +X)
    R_cv_to_cam = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    T_cv_to_cam = np.eye(4)
    T_cv_to_cam[:3, :3] = R_cv_to_cam
    q_cv_to_cam = quaternion_from_matrix(T_cv_to_cam)
    
    # Full rotation: Take pose in CV frame and convert to Vicon frame
    # Not yet accounting for offset of CV origin from Drone origin in Vicon frame
    q_cv_to_drone = quaternion_multiply(q_cam_to_drone, q_cv_to_cam)

    # Translation from drone origin to camera origin
    t_cam_in_drone = np.array([0.23, 0.05, -0.12])

    # ---- 2. Drone body -> world frame (from drone_pose) -----------------

    q_drone_to_world = np.array([
        drone_pose.pose.orientation.x,
        drone_pose.pose.orientation.y,
        drone_pose.pose.orientation.z,
        drone_pose.pose.orientation.w,
    ])

    t_drone_in_world = np.array([
        drone_pose.pose.position.x,
        drone_pose.pose.position.y,
        drone_pose.pose.position.z,
    ])

    # Rotate camera offset into world frame
    def rotate_vector_by_quaternion(v, q):
        """Rotate vector v by quaternion q (active rotation)."""
        q_vec = np.array([v[0], v[1], v[2], 0.0])
        q_conj = quaternion_conjugate(q)
        rotated = quaternion_multiply(quaternion_multiply(q, q_vec), q_conj)
        return rotated[:3]

    t_drone_to_cam_in_world = rotate_vector_by_quaternion(t_cam_in_drone, q_drone_to_world)

    # World position of the camera origin
    t_cam_in_world = t_drone_in_world + t_drone_to_cam_in_world

    # ---- 3. Target position in world frame ------------------------------

    target_in_cam = np.array([
        target_pose_camera.pose.position.x,
        target_pose_camera.pose.position.y,
        target_pose_camera.pose.position.z,
    ])

    # Rotation: camera -> world = body->world composed with camera->body
    q_cam_to_world = quaternion_multiply(q_drone_to_world, q_cv_to_drone)

    target_in_world = t_cam_in_world + rotate_vector_by_quaternion(target_in_cam, q_cam_to_world)

    # ---- 4. Build output PoseStamped ------------------------------------

    result = PoseStamped()
    result.header = target_pose_camera.header
    result.pose.position.x = float(target_in_world[0])
    result.pose.position.y = float(target_in_world[1])
    result.pose.position.z = float(target_in_world[2])

    # Target orientation in world frame (transform the camera-frame orientation)
    q_target_cam = np.array([
        target_pose_camera.pose.orientation.x,
        target_pose_camera.pose.orientation.y,
        target_pose_camera.pose.orientation.z,
        target_pose_camera.pose.orientation.w,
    ])
    q_target_world = quaternion_multiply(q_cam_to_world, q_target_cam)
    result.pose.orientation.x = float(q_target_world[0])
    result.pose.orientation.y = float(q_target_world[1])
    result.pose.orientation.z = float(q_target_world[2])
    result.pose.orientation.w = float(q_target_world[3])

    return result


def xy_distance(pose_a: PoseStamped, pose_b: PoseStamped) -> float:
    """
    Euclidean distance between two poses in the x/y plane only (ignores z).
    Used for MAX_TRACKING_DISTANCE enforcement.
    """
    dx = pose_a.pose.position.x - pose_b.pose.position.x
    dy = pose_a.pose.position.y - pose_b.pose.position.y
    return np.sqrt(dx ** 2 + dy ** 2)


def compute_tracking_pose(
    drone_pose: PoseStamped,
    target_pose: PoseStamped,
    standoff_radius: float,
    hover_above: float,
) -> PoseStamped:
    """
    Compute a stable setpoint for the drone to observe the target object.

    Desired geometry
    ----------------
    The drone should eventually sit on a horizontal circle of radius
    `standoff_radius` centred directly above the target at height
    ``target.z + hover_above``, facing the target.

    Stability behaviour
    -------------------
    The drone approaches the closest point on the standoff circle while it is
    outside that circle.  Once it reaches the circle (xy distance from target
    <= standoff_radius) the *position* setpoint is frozen — only the yaw
    (facing direction) is updated to keep the drone pointed at the moving
    target.  This prevents the erratic setpoint-hopping that occurs when a
    small target movement shifts the "closest circle point" discontinuously.

    The position setpoint is only updated while the drone is clearly outside
    the standoff radius, so minor target wobbles never trigger a repositioning
    manoeuvre.
    """
    tx = target_pose.pose.position.x
    ty = target_pose.pose.position.y
    tz = target_pose.pose.position.z

    dx = drone_pose.pose.position.x - tx
    dy = drone_pose.pose.position.y - ty
    dist_xy = np.sqrt(dx ** 2 + dy ** 2)

    if dist_xy <= standoff_radius:
        # ----------------------------------------------------------------
        # ON OR INSIDE the circle — hold current x/y position, only update
        # yaw so the drone keeps facing the target as it moves.
        # ----------------------------------------------------------------
        setpoint_x = drone_pose.pose.position.x
        setpoint_y = drone_pose.pose.position.y
        setpoint_z = tz + hover_above
    else:
        # ----------------------------------------------------------------
        # OUTSIDE the circle — approach the nearest point on the circle.
        # ----------------------------------------------------------------
        angle_to_drone = np.arctan2(dy, dx)
        setpoint_x = tx + standoff_radius * np.cos(angle_to_drone)
        setpoint_y = ty + standoff_radius * np.sin(angle_to_drone)
        setpoint_z = tz + hover_above

    # Yaw so the drone always faces the target
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
