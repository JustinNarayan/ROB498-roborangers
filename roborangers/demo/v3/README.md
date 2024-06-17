# Drone Node — README

## Prerequisites

Before running the drone node, start the required infrastructure depending on your vision source:

**Always required:**
```bash
ros2 launch roborangers mavros.launch.py
```

**If using Vicon:**
Enable the Vicon system externally, then ensure `CURRENT_MISSION = MissionType.VICON` in `constants.py`.

**If using Realsense:**
```bash
ros2 launch realsense2_camera rs_launch.py
```
Ensure `CURRENT_MISSION = MissionType.REALSENSE` (or `MissionType.REALSENSE_WITH_FALLBACK`) in `constants.py`.

---

## Changing Vision Source

Open `constants.py` and set `CURRENT_MISSION` to one of:

| Value | Behaviour |
|---|---|
| `MissionType.VICON` | Vicon only |
| `MissionType.REALSENSE` | Realsense only |
| `MissionType.REALSENSE_WITH_FALLBACK` | Realsense primary, auto-faults to Vicon if the two readings diverge |

---

## Changing Target Source

Open `constants.py` and set `CURRENT_TARGET_TYPE` to one of:

| Value | Behaviour |
|---|---|
| `TargetType.VICON` | Target pose is already in the global Vicon/world frame (use with `forward_vicon_target_pose.py`) |
| `TargetType.COMPUTER_VISION` | Target pose arrives in the left-fisheye camera frame and is automatically transformed to the world frame |

---

## Running the Node

```bash
ros2 run roborangers demo_v3.py
```

---

## Services

| Service | Type | Description |
|---|---|---|
| `/rob498_drone_6/comm/launch` | `std_srvs/Trigger` | Arm and enter hover |
| `/rob498_drone_6/comm/test` | `std_srvs/Trigger` | Begin survey (rotate in place searching for target) |
| `/rob498_drone_6/comm/land` | `std_srvs/Trigger` | Return home then land |
| `/rob498_drone_6/comm/abort` | `std_srvs/Trigger` | Land immediately in place |

```bash
ros2 service call /rob498_drone_6/comm/launch std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/test   std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/land   std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/abort  std_srvs/srv/Trigger {}
```

---

## Vicon Target Forwarder

When `CURRENT_TARGET_TYPE = TargetType.VICON`, run the forwarder node (in a separate terminal) to publish the RC car Vicon pose to the target topic:

```bash
ros2 run roborangers forward_vicon_target_pose.py
```

This subscribes to `/rob498_drone_6/Vicon_RC_CAR` and republishes unchanged on `/rob498_drone_6/target/pose`.

---

## Useful Topics to Monitor

```bash
# Realsense pose feed
ros2 topic echo /camera/pose/sample --truncate-length 5

# What the drone is commanding to the flight controller
ros2 topic echo /mavros/setpoint_position/local

# Vision pose being fed into MAVROS EKF
ros2 topic echo /mavros/vision_pose/pose

# Target pose (world frame, after any transform)
ros2 topic echo /rob498_drone_6/target/pose
```

---

## Target Pose Simulation

Used to simulate a detected target without a real CV pipeline.

**Run the simulator** (in a separate terminal, after the main node is running):
```bash
ros2 run roborangers simulate_target_pose.py
```

Publishes to `/rob498_drone_6/target/pose` at 2 Hz. Default payload is all-zeros, which the drone treats as "no target found".

### Example: Full workflow

```bash
# 1. Start publishing (enabled by default on launch, this re-enables after a stop)
ros2 param set /simulate_target_pose publishing_enabled true

# 2. Stop publishing (drone should fall back to SURVEYING after ~1 second)
ros2 param set /simulate_target_pose publishing_enabled false

# 3. Set a target position (identity orientation)
ros2 param set /simulate_target_pose target_x 1.5
ros2 param set /simulate_target_pose target_y 2.0
ros2 param set /simulate_target_pose target_z 0.0
ros2 param set /simulate_target_pose target_qw 1.0

# 4. Resume publishing (drone should transition from SURVEYING -> TRACKING_TARGET)
ros2 param set /simulate_target_pose publishing_enabled true

# 5. Clear the target (return to all-zero sentinel, drone returns to SURVEYING)
ros2 param set /simulate_target_pose clear true
```

To inspect the current simulator state:
```bash
ros2 param dump /simulate_target_pose
```

---

## Key Constants (constants.py)

| Constant | Default | Description |
|---|---|---|
| `CURRENT_MISSION` | `MissionType.VICON` | Vision source for drone localisation |
| `CURRENT_TARGET_TYPE` | `TargetType.VICON` | Source/frame of incoming target poses |
| `TARGET_STANDOFF_RADIUS` | `0.4 m` | Orbit radius around the target in x/y |
| `TARGET_HOVER_ABOVE` | `0.5 m` | Height above the target |
| `MAX_TRACKING_DISTANCE` | `3.0 m` | Max x/y distance at which a target is accepted (safety filter) |
| `TARGET_STALENESS_THRESHOLD_NANOSECONDS` | `1e9 ns` | How long a target pose is considered fresh |
| `SURVEY_ANGULAR_STEP_RADIANS` | `0.393 rad` | Yaw step during survey (~22.5 deg) |
| `SURVEY_STEP_HOLD_TIME_NANOSECONDS` | `2e9 ns` | Hold time per survey step |
