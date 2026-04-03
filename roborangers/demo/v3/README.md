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
| `TargetType.CV_WITH_VICON_VALIDATION` | CV primary, but every CV frame is cross-checked against the simultaneous Vicon RC car reading. Frames whose x/y position disagrees with Vicon by more than `CV_VICON_POSITION_AGREEMENT_THRESHOLD` are **rejected** (not replaced by Vicon). Use this to sanity-check your CV pipeline before trusting it for flight. |

### CV with Vicon Validation

When `CURRENT_TARGET_TYPE = TargetType.CV_WITH_VICON_VALIDATION`:

- The node automatically subscribes to `VICON_RC_CAR_TOPIC` for a ground-truth reference.
- Each incoming CV frame is transformed to the world frame, then its x/y position is compared against the latest Vicon reading.
- If they agree within `CV_VICON_POSITION_AGREEMENT_THRESHOLD` (default 0.3 m), the frame is accepted and treated as a normal target detection.
- If they disagree, the frame is silently dropped and a `WARN` is logged. The previous valid detection is retained until it goes stale.
- If no Vicon reference has been received yet, the CV frame is accepted unconditionally (so the drone is not blind at startup) and a `WARN` is logged.
- Tune `CV_VICON_POSITION_AGREEMENT_THRESHOLD` in `constants.py`.

---

## Tracking Mode: Move vs Rotate-Only

Set `TRACKING_MOVE_TO_TRACK` in `constants.py`:

| Value | Behaviour |
|---|---|
| `True` (default) | Normal tracking: drone moves to the standoff orbit around the target |
| `False` | Rotate-only: drone holds its init hover position and only rotates in yaw to face the detected target |

Rotate-only mode is useful to visually verify that CV detections are sensible before allowing the drone to chase them.

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
| `/rob498_drone_6/comm/abort` | `std_srvs/Trigger` | Land immediately in place (also exits GEO_FENCE_HOLD) |
| `/rob498_drone_6/comm/overhead` | `std_srvs/Trigger` | Enter/toggle OVERHEAD mode (see below) |

```bash
ros2 service call /rob498_drone_6/comm/launch   std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/test      std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/land      std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/abort     std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/overhead  std_srvs/srv/Trigger {}
```

---

## OVERHEAD Mode

The `/overhead` service positions the drone so that its **capture net** is directly above the current target's x/y position, rather than the drone's own origin.

**Entering OVERHEAD:**
- Can be called from `SURVEYING` or `TRACKING_TARGET`.
- Requires a valid, non-stale target to be currently detected. If no target is available the command is **rejected** (the service returns `success: false` and a descriptive message is logged).
- On entry the drone's current yaw is captured and held fixed throughout OVERHEAD, keeping the drone as level and stable as possible.

**Net frame offset:**
The net is mounted at a fixed offset from the drone origin in the drone's body frame (constants `NET_OFFSET_X` and `NET_OFFSET_Y`). The node automatically computes the world-frame drone position required to place the net over the target, accounting for the drone's current heading.

| Constant | Default | Description |
|---|---|---|
| `NET_OFFSET_X` | `0.10 m` | Net offset along drone body +X (forward) |
| `NET_OFFSET_Y` | `0.05 m` | Net offset along drone body +Y (left) |

**Leaving OVERHEAD:**
- Call `/overhead` a **second time** → toggles back to `SURVEYING`.
- Call `/land` → graceful return-home then land.
- Call `/abort` → immediate land in place.
- If the target goes stale while in OVERHEAD the drone automatically falls back to `SURVEYING`.

**State diagram additions:**
```
SURVEYING / TRACKING_TARGET  --[/overhead, valid target]-->  OVERHEAD
OVERHEAD                     --[/overhead again]          -->  SURVEYING
OVERHEAD                     --[target lost]              -->  SURVEYING
OVERHEAD                     --[/land]                    -->  GOING_HOME
OVERHEAD                     --[/abort]                   -->  LANDING
```

---

## Geo-Fence / Emergency Hold

If the drone's position ever exceeds **`MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT`** (default 3 m) from its home position, the node immediately enters `GEO_FENCE_HOLD`:

- The drone **freezes in place** by commanding its current vision pose as its setpoint. It does not land automatically — this allows time for the operator to diagnose the situation.
- The flight controller remains armed and in OFFBOARD mode so the hover setpoint is respected.
- An `ERROR`-level log message is emitted every control tick while in this state.
- To land from `GEO_FENCE_HOLD`, send `/abort`.

This is a latched condition — once triggered it cannot be cleared without sending `/abort`.

> **Rationale:** A fully automatic landing on geo-fence breach would cut power above an unexpected location. Hovering in place gives the operator the option to regain manual RC control or send a controlled abort.

---

## Vicon Target Forwarder

When `CURRENT_TARGET_TYPE = TargetType.VICON`, run the forwarder node (in a separate terminal) to publish the RC car Vicon pose to the target topic:

```bash
ros2 run roborangers forward_vicon_target_pose.py
```

This subscribes to `/vicon/rob498_rc_car_team6/rob498_rc_car_team6` and republishes unchanged on `/rob498_drone_6/target/pose`.

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

# Per-axis Realsense/Vicon divergence (if DEBUG_VISION_DIVERGENCE = True)
ros2 topic echo /rob498_drone_6/debug/vision_divergence/position
ros2 topic echo /rob498_drone_6/debug/vision_divergence/orientation
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

### Vision & Localisation

| Constant | Default | Description |
|---|---|---|
| `CURRENT_MISSION` | `MissionType.VICON` | Vision source for drone localisation |
| `REALSENSE_VICON_POSITION_DIVERGENCE_THRESHOLD` | `0.3 m` | Max position gap before faulting Realsense → Vicon |
| `REALSENSE_VICON_ORIENTATION_DIVERGENCE_THRESHOLD` | `0.174 rad` | Max orientation gap before faulting Realsense → Vicon |
| `DEBUG_VISION_DIVERGENCE` | `True` | Publish per-axis Realsense/Vicon divergence on debug topics |

### Target & Tracking

| Constant | Default | Description |
|---|---|---|
| `CURRENT_TARGET_TYPE` | `TargetType.VICON` | Source/frame of incoming target poses |
| `CV_VICON_POSITION_AGREEMENT_THRESHOLD` | `0.3 m` | Max CV/Vicon x/y disagreement before CV frame is rejected (CV_WITH_VICON_VALIDATION only) |
| `TRACKING_MOVE_TO_TRACK` | `True` | `True` = full orbit tracking; `False` = rotate-only (hold home position) |
| `TARGET_STANDOFF_RADIUS` | `0.8 m` | Orbit radius around the target in x/y |
| `TARGET_HOVER_ABOVE` | `0.5 m` | Height above the target |
| `MAX_TRACKING_DISTANCE` | `2.0 m` | Max x/y distance at which a target is accepted (safety filter) |
| `TARGET_STALENESS_THRESHOLD_NANOSECONDS` | `1e9 ns` | How long a target pose is considered fresh |

### Net Frame (OVERHEAD mode)

| Constant | Default | Description |
|---|---|---|
| `NET_OFFSET_X` | `0.10 m` | Net offset along drone body +X (forward) |
| `NET_OFFSET_Y` | `0.05 m` | Net offset along drone body +Y (left) |

### Survey

| Constant | Default | Description |
|---|---|---|
| `SURVEY_ANGULAR_STEP_RADIANS` | `0.393 rad` | Yaw step during survey (~22.5 deg) |
| `SURVEY_STEP_HOLD_TIME_NANOSECONDS` | `2e9 ns` | Hold time per survey step |

### Safety

| Constant | Default | Description |
|---|---|---|
| `MAX_EMERGENCY_LAND_DISTANCE_FROM_INIT` | `3.0 m` | Distance from home that triggers GEO_FENCE_HOLD |
