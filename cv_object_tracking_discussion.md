# ROB498 Object Tracking on Jetson Nano - Technical Discussion

**Date:** March 9, 2026

---

## Overview

Discussion on implementing object tracking for an RC car using YOLOv8 on a Jetson Nano with an IMX219 camera, combined with RealSense depth for pose estimation.

---

## Key Components

- **Drone Hardware:** Jetson Nano (inference), Cube+ flight controller
- **Detection Camera:** IMX219 (CSI interface)
- **Depth Sensor:** RealSense (for stable flight via Kalman filter)
- **Target:** RC car position estimation in camera frame, transformable to drone frame

---

## Stage 1 — Detection: YOLOv8n + TensorRT

### Export Configuration
```
model.export(format='engine', half=True, device=0)  # FP16 TensorRT for Nano
imgsz=320 or 416  # Reduced from 640 for latency optimization
Target FPS: 15–30 on Jetson Nano
```

### Fine-Tuning Strategy
- **Starting point:** COCO pretrained YOLOv8n (includes car class)
- **Issue:** COCO cars are full-sized, street-level perspective
- **Our scenario:** Small RC cars, aerial/drone perspective, motion blur, lighting variation

### Fine-Tuning Approach
```
Frozen backbone (first ~10 epochs)
  → Fine-tune on custom dataset (300–500 aerial RC car images)
  → Augmentations:
     - mosaic=True
     - degrees=45 (aerial rotation)
     - scale=0.3–2.0
     - blur_limit=7 (motion blur)
     - perspective=0.001
```

### Data Pipeline
- Use **Roboflow** for annotation + automated augmentation
- Generates motion blur and perspective transformations at export

---

## Stage 2 — Depth Estimation: RGB-D Hybrid

### Better Approach (vs. Monocular PnP)

You have a RealSense already — use it for depth rather than geometric estimation.

**Pipeline:**
1. Run YOLOv8n on IMX219 → get bounding box
2. Transform IMX219 bbox centroid to RealSense pixel coordinates
3. Sample RealSense depth at transformed location
4. Back-project using RealSense intrinsics

### Why RGB-D Hybrid is Superior

| Metric | Size-based Monocular | RGB-D Hybrid |
|--------|----------------------|-------------|
| Accuracy | ±0.3–0.5 m error | ±0.05–0.1 m error |
| Assumptions | Known object width, fronto-parallel | None (direct measurement) |
| Computational Cost | Negligible | Negligible (hardware) |
| Robustness to Pose/Angle | Poor | Excellent |

### Fallback: Size-Based Monocular PnP

If RealSense depth is invalid (occlusion, out of range):

Given:
- $W_{real}$ = known RC car width (e.g., 0.3 m)
- $f_x, f_y, c_x, c_y$ = IMX219 intrinsics
- $w_{px}$ = bounding box pixel width

Depth estimate:
$$Z = \frac{f_x \cdot W_{real}}{w_{px}}$$

Back-project centroid $(u, v)$:
$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$

---

## Key Challenge: Camera Coordinate Transformation

### Problem
Two cameras with different intrinsics and physical positions must register detections:
- IMX219: $(u_{imx}, v_{imx})$
- RealSense: $(u_{rs}, v_{rs})$

### Solution: Extrinsic Calibration

Need to compute $R_{rs}^{imx219}$ (rotation) and $\mathbf{t}_{rs}^{imx219}$ (translation).

**3D reprojection approach:**

$$\text{Normalize IMX219: } \mathbf{n}_{imx} = K_{imx}^{-1} [u_{imx}, v_{imx}, 1]^T$$

$$\text{Transform to RealSense: } \mathbf{p}_{rs} = R_{rs}^{imx219} \cdot \mathbf{n}_{imx} + \mathbf{t}_{rs}^{imx219}$$

$$\text{Project to RealSense image: } \begin{bmatrix} u_{rs} \\ v_{rs} \end{bmatrix} = K_{rs} \cdot \frac{\mathbf{p}_{rs}}{z_{rs}}$$

### Calibration Methods

1. **Checkerboard Sync Calibration** (Most Robust)
   - Capture synchronized stereo pairs
   - Use OpenCV stereo calibration or Kalibr

2. **Hand-Measure + Optimize** (Quick)
   - Physical measurement of relative positions
   - Refine with a few sync pairs

3. **Co-located Assumption** (Quick & Dirty)
   - If cameras < 5cm apart and roughly parallel
   - Set $R = I$, $\mathbf{t} \approx [0, 0, 0]$

### Simplified: Homography Mapping

If cameras are close and coplanar:
$$\mathbf{H} = K_{rs} \cdot R_{rs}^{imx219} \cdot K_{imx}^{-1}$$

$$\begin{bmatrix} u_{rs} \\ v_{rs} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u_{imx} \\ v_{imx} \\ 1 \end{bmatrix}$$

---

## Stage 3 — Frame Transformation

Apply extrinsic calibration of IMX219 mount:

$$\mathbf{p}_{drone} = R_{cam}^{drone} \cdot \mathbf{p}_{cam} + \mathbf{t}_{cam}^{drone}$$

Publish as static TF in ROS 2.

---

## Alternative: DOPE (Deep Object Pose Estimation)

### When to Use
- If you need full 6DoF object orientation (not just position)
- RC car pose matters for future control tasks

### Challenges on Jetson Nano
- Requires CAD model of RC car
- Requires synthetic data generation (Blenderproc)
- Requires training
- Computational overhead

### Viability
- **Yes, but costly.** Use only if size-based + RGB-D insufficient.
- TensorRT optimization available via `isaac_ros_dope`
- Significantly more setup than the above pipeline

### Not Recommended
- **FoundPose:** Requires large foundation model (DINOv2-scale) — not realtime on Nano

---

## Full Pipeline Architecture

```
IMX219 (CSI/v4l2)
    ↓
YOLOv8n-TensorRT (320px, FP16)
    → bbox [u1,v1,u2,v2] + confidence
    ↓
Camera extrinsic calibration (IMX219 → RealSense)
    ↓
RealSense depth at transformed coordinates
    + IMX219 intrinsics
    ↓
p_cam = [X_c, Y_c, Z_c]  (RC car in camera frame)
    ↓
Static TF: camera → drone_body
    ↓
p_drone = [X_d, Y_d, Z_d]  (RC car in drone frame)
    ↓
ROS 2 topic: /rc_car/pose (geometry_msgs/PointStamped)
    ↓
Mission planner / waypoint controller
```

**Parallel path (untouched):**
- RealSense → MAVROS → Cube+ flight controller (Kalman filtering for stable flight)

---

## Critical Prerequisites

### 1. IMX219 Intrinsic Calibration
```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  --ros-args -r image:=/camera/image_raw
```

Output: $f_x, f_y, c_x, c_y$, distortion coefficients

### 2. RealSense Intrinsics
- Available from RealSense driver (published in camera_info topic)
- Verify with factory calibration

### 3. Inter-Camera Extrinsic Calibration
- Perform checkerboard sync calibration or measure physically
- Compute $R_{rs}^{imx219}$ and $\mathbf{t}_{rs}^{imx219}$

---

## Implementation Notes

### Pseudocode
```python
# Detection
bbox = yolov8.predict(imx219_frame)
u_imx, v_imx = bbox.center_x, bbox.center_y

# Transform to RealSense coordinates
u_rs, v_rs = transform_pixel(u_imx, v_imx, H_extrinsic)

# Depth sampling
depth_value = realsense_depth[v_rs, u_rs]

# Handle invalid depth
if depth_value <= 0:
    # Fallback to size-based estimate
    depth_value = estimate_depth_from_bbox_width(bbox.width)

# Back-project to 3D
p_cam = realsense.deproject_pixel([u_rs, v_rs], depth_value)

# Transform to drone frame
p_drone = R_cam_to_drone @ p_cam + t_cam_to_drone

# Publish
publish(/rc_car/pose, p_drone)
```

---

## Summary Recommendations

| Task | Tool | Notes |
|------|------|-------|
| Detection | YOLOv8n, fine-tuned, TensorRT FP16 | 15–30 FPS on Nano |
| Depth | RealSense RGB-D hybrid | Requires extrinsic calibration |
| Fallback depth | Size-based monocular PnP | When RealSense invalid |
| 6DoF pose (if needed later) | DOPE + TensorRT | Requires CAD model; use only if necessary |
| Data pipeline | Roboflow | Automated augmentation |
| Calibration | Checkerboard sync or Kalibr | Critical first step |

---

## Open Questions / Next Steps
1. What is the physical layout of your cameras on the drone?
2. Do you have a CAD model of the RC car? (For potential DOPE use)
3. What is the expected tracking distance and FOV overlap?
4. Can you do the inter-camera extrinsic calibration before flight?


Things we need: May need to conduct training using rc cars but first attempt pretrained model(on COCO dataset which is trained on real cars). We need the transformation matrix between the imx and the drone. Also need to conduct checkerboard calibration to extract transformation matrix between imx and realsense. Check realsense topics to extract depth ma input=pixel location 