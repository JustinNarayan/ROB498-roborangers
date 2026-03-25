#!/usr/bin/env python3
"""
Extrinsic calibration between IMX219 (cam0) and T265 fisheye1 (cam1)
using a ROS2 .db3 bag and an Aprilgrid target.

Requirements:
    pip install opencv-python opencv-contrib-python rosbags numpy pyyaml

Usage:
    python3 calibrate_extrinsics.py \
        --bag /home/jetson/calibration_inputs/multicam_calib_bag_2 \
        --cameras /home/jetson/calibration_inputs/calibration_params/cameras.yaml \
        --aprilgrid /home/jetson/calibration_inputs/calibration_params/aprilgrid.yaml \
        --output extrinsics.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── rosbags imports (ROS2 .db3 reader, no ROS install needed) ───────────────
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("ERROR: rosbags not installed. Run: pip install rosbags")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMX_TOPIC      = "/imx219/image_raw"           # CompressedImage in bag
FISHEYE_TOPIC  = "/camera/fisheye1/image_raw"  # raw Image in bag
SYNC_TOLERANCE = 0.15                           # seconds — generous for 5-6 Hz IMX


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo extrinsic calibration from ROS2 bag")
    p.add_argument("--bag",       required=True, help="Path to ROS2 bag directory")
    p.add_argument("--cameras",   required=True, help="Path to cameras.yaml")
    p.add_argument("--aprilgrid", required=True, help="Path to aprilgrid.yaml")
    p.add_argument("--output",    default="extrinsics.yaml")
    p.add_argument("--max-pairs", type=int, default=80,
                   help="Max frame pairs to use (default 80, more = slower)")
    p.add_argument("--debug",     action="store_true",
                   help="Show detected corners for each pair")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera parameter loading
# ---------------------------------------------------------------------------

def load_cameras(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        data = yaml.safe_load(f)

    def parse_cam(c: dict) -> dict:
        intr = c["intrinsics"]
        return {
            "fx": intr[0], "fy": intr[1], "cx": intr[2], "cy": intr[3],
            "K": np.array([[intr[0], 0, intr[2]],
                           [0, intr[1], intr[3]],
                           [0,       0,       1]], dtype=np.float64),
            "D": np.array(c["distortion_coeffs"], dtype=np.float64),
            "distortion_model": c["distortion_model"],
            "resolution": tuple(c["resolution"]),
        }

    return parse_cam(data["cam0"]), parse_cam(data["cam1"])


# ---------------------------------------------------------------------------
# Aprilgrid setup
# ---------------------------------------------------------------------------

def build_aprilgrid_detector(cfg: dict):
    """
    Returns (detector, object_points_template).
    Object points are in the Aprilgrid coordinate frame (metres).
    """
    tag_size    = float(cfg["tagSize"])
    tag_spacing = float(cfg["tagSpacing"]) * tag_size   # spacing in metres
    cols        = int(cfg["tagCols"])
    rows        = int(cfg["tagRows"])

    # Each Apriltag has 4 corners. Build the full grid of 3D object points.
    # Corner order matches cv2.aruco detection order (top-left, top-right,
    # bottom-right, bottom-left for each tag).
    obj_pts_per_tag = []
    for row in range(rows):
        for col in range(cols):
            # Top-left corner of this tag in the grid
            x0 = col * (tag_size + tag_spacing)
            y0 = row * (tag_size + tag_spacing)
            obj_pts_per_tag.append([
                [x0,            y0,            0],
                [x0 + tag_size, y0,            0],
                [x0 + tag_size, y0 + tag_size, 0],
                [x0,            y0 + tag_size, 0],
            ])

    obj_pts_template = np.array(obj_pts_per_tag, dtype=np.float32)  # (N_tags, 4, 3)

    # Use the 36h11 dictionary (standard Aprilgrid used by Kalibr)
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    return detector, obj_pts_template, cols, rows


def detect_aprilgrid(image: np.ndarray,
                     detector,
                     obj_pts_template: np.ndarray,
                     cols: int,
                     rows: int,
                     min_tags: int = 8
                     ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detect Aprilgrid corners in image.
    Returns (image_points, object_points) or (None, None) if not enough tags found.
    Both arrays are (N, 1, 2/3) for use with OpenCV calibration functions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < min_tags:
        return None, None

    n_tags = cols * rows
    img_pts = []
    obj_pts = []

    for i, tag_id in enumerate(ids.flatten()):
        if tag_id >= n_tags:
            continue   # unexpected tag id
        img_pts.append(corners[i].reshape(4, 2))          # (4, 2)
        obj_pts.append(obj_pts_template[tag_id])           # (4, 3)

    if len(img_pts) < min_tags:
        return None, None

    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)   # (4N, 1, 2)
    obj_pts = np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3)   # (4N, 1, 3)
    return img_pts, obj_pts


# ---------------------------------------------------------------------------
# Bag reading helpers
# ---------------------------------------------------------------------------

def ns_to_sec(ns: int) -> float:
    return ns * 1e-9


def decode_compressed_image(data: bytes) -> np.ndarray | None:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def decode_raw_image(msg) -> np.ndarray | None:
    """Decode a sensor_msgs/Image (mono8 or bgr8) to a numpy array."""
    encoding = msg.encoding
    h, w     = msg.height, msg.width
    data     = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("mono8", "8UC1"):
        img = data.reshape(h, w)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif encoding in ("bgr8", "rgb8"):
        img = data.reshape(h, w, 3)
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "bgra8":
        img = data.reshape(h, w, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        print(f"  WARNING: unsupported encoding {encoding}, attempting reshape")
        img = data.reshape(h, w, -1)

    return img


def read_bag(bag_path: str, typestore):
    """
    Read all messages from both topics.
    Returns two lists of (timestamp_sec, image) sorted by time.
    """
    imx_frames     = []   # (ts, np.ndarray)
    fisheye_frames = []

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}

        if IMX_TOPIC not in connections:
            raise RuntimeError(
                f"Topic '{IMX_TOPIC}' not found in bag.\n"
                f"Available: {list(connections.keys())}")
        if FISHEYE_TOPIC not in connections:
            raise RuntimeError(
                f"Topic '{FISHEYE_TOPIC}' not found in bag.\n"
                f"Available: {list(connections.keys())}")

        imx_conn     = connections[IMX_TOPIC]
        fisheye_conn = connections[FISHEYE_TOPIC]

        for conn, ts_ns, raw in reader.messages(
                connections=[imx_conn, fisheye_conn]):
            ts = ns_to_sec(ts_ns)
            msg = typestore.deserialize_cdr(raw, conn.msgtype)

            if conn.topic == IMX_TOPIC:
                # CompressedImage — msg.data is the JPEG bytes
                img = decode_compressed_image(bytes(msg.data))
                if img is not None:
                    imx_frames.append((ts, img))

            elif conn.topic == FISHEYE_TOPIC:
                img = decode_raw_image(msg)
                if img is not None:
                    fisheye_frames.append((ts, img))

    imx_frames.sort(key=lambda x: x[0])
    fisheye_frames.sort(key=lambda x: x[0])

    print(f"  Read {len(imx_frames)} IMX219 frames")
    print(f"  Read {len(fisheye_frames)} fisheye1 frames")
    return imx_frames, fisheye_frames


# ---------------------------------------------------------------------------
# Timestamp synchronisation
# ---------------------------------------------------------------------------

def sync_frames(imx_frames: list, fisheye_frames: list,
                tolerance: float = SYNC_TOLERANCE) -> list[tuple]:
    """
    For each IMX219 frame find the nearest fisheye frame in time.
    Returns list of (imx_img, fisheye_img, dt) for pairs within tolerance.
    """
    fisheye_ts = np.array([f[0] for f in fisheye_frames])
    pairs      = []

    for ts_imx, img_imx in imx_frames:
        idx = np.argmin(np.abs(fisheye_ts - ts_imx))
        dt  = abs(fisheye_ts[idx] - ts_imx)
        if dt <= tolerance:
            pairs.append((img_imx, fisheye_frames[idx][1], dt))

    print(f"  Synchronised {len(pairs)} pairs "
          f"(tolerance={tolerance*1000:.0f}ms)")
    return pairs


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading camera parameters...")
    cam0, cam1 = load_cameras(args.cameras)
    print(f"  IMX219   K={cam0['K'].diagonal()[:2]}  res={cam0['resolution']}")
    print(f"  Fisheye1 K={cam1['K'].diagonal()[:2]}  res={cam1['resolution']}")

    with open(args.aprilgrid) as f:
        grid_cfg = yaml.safe_load(f)
    detector, obj_pts_template, cols, rows = build_aprilgrid_detector(grid_cfg)
    print(f"  Aprilgrid {cols}×{rows}  tagSize={grid_cfg['tagSize']}m")

    # ── Read bag ─────────────────────────────────────────────────────────────
    print("\n[2/5] Reading bag...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    imx_frames, fisheye_frames = read_bag(args.bag, typestore)

    # ── Sync ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Synchronising frames...")
    pairs = sync_frames(imx_frames, fisheye_frames)
    if len(pairs) < 10:
        print(f"ERROR: Only {len(pairs)} synchronised pairs found. "
              f"Try increasing SYNC_TOLERANCE (currently {SYNC_TOLERANCE}s).")
        sys.exit(1)

    # ── Detect Aprilgrid ─────────────────────────────────────────────────────
    print("\n[4/5] Detecting Aprilgrid corners...")

    all_obj_pts  = []   # 3D points (same for both cameras per pair)
    all_imx_pts  = []   # 2D points in IMX219
    all_fish_pts = []   # 2D points in fisheye1

    # Stride through pairs evenly if more than max_pairs
    stride = max(1, len(pairs) // args.max_pairs)
    candidates = pairs[::stride]

    for i, (img_imx, img_fish, dt) in enumerate(candidates):
        imx_ipts,  imx_opts  = detect_aprilgrid(img_imx,  detector,
                                                  obj_pts_template, cols, rows)
        fish_ipts, fish_opts = detect_aprilgrid(img_fish, detector,
                                                  obj_pts_template, cols, rows)

        # Only use pairs where the SAME tags were detected in both cameras
        if imx_ipts is None or fish_ipts is None:
            continue

        # Match by object point (same 3D point must be visible in both)
        # Since we indexed by tag_id, obj_pts already correspond if shapes match
        if imx_opts.shape != fish_opts.shape:
            # Different tags visible — find common subset
            # (skip for simplicity; in practice same tags usually visible)
            continue

        all_obj_pts.append(imx_opts)
        all_imx_pts.append(imx_ipts)
        all_fish_pts.append(fish_ipts)

        if args.debug:
            vis = img_imx.copy()
            cv2.aruco.drawDetectedMarkers(vis, 
                [imx_ipts.reshape(-1, 4, 2).astype(np.float32)], None)
            cv2.imshow("IMX219 detections", cv2.resize(vis, (820, 616)))
            cv2.imshow("Fisheye detections", img_fish)
            cv2.waitKey(100)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(candidates)} candidates, "
                  f"{len(all_obj_pts)} valid pairs so far...")

    cv2.destroyAllWindows()
    n_valid = len(all_obj_pts)
    print(f"  Valid pairs with grid detected in both cameras: {n_valid}")

    if n_valid < 10:
        print("ERROR: Not enough valid pairs. Check that the Aprilgrid was "
              "clearly visible in both cameras simultaneously.")
        sys.exit(1)

    # ── Stereo calibration ───────────────────────────────────────────────────
    print(f"\n[5/5] Running stereoCalibrate on {n_valid} pairs...")

    # IMX219 — standard pinhole, radtan distortion (k1,k2,p1,p2,k3)
    K0 = cam0["K"]
    D0 = cam0["D"]
    if len(D0) == 4:
        D0 = np.append(D0, 0.0)   # OpenCV wants 5 coeffs for radtan

    # Fisheye1 — Kannala-Brandt (equidistant), needs cv2.fisheye functions
    # For stereoCalibrate we undistort fisheye points first, then treat as
    # pinhole with identity distortion
    K1_raw = cam1["K"]
    D1_raw = cam1["D"].reshape(-1, 1) if cam1["D"].ndim == 1 else cam1["D"]

    # Undistort fisheye image points → normalised coords → reprojected with K1
    fish_pts_undist = []
    for pts in all_fish_pts:
        ud = cv2.fisheye.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float32),
            K1_raw, D1_raw,
            P=K1_raw)
        fish_pts_undist.append(ud)

    D1_zero = np.zeros((5, 1), dtype=np.float64)  # distortion already removed

    flags = (cv2.CALIB_FIX_INTRINSIC)   # intrinsics are fixed, solve R and T only

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objectPoints   = all_obj_pts,
        imagePoints1   = all_imx_pts,
        imagePoints2   = fish_pts_undist,
        cameraMatrix1  = K0,
        distCoeffs1    = D0,
        cameraMatrix2  = K1_raw,
        distCoeffs2    = D1_zero,
        imageSize      = cam0["resolution"],
        flags          = flags,
        criteria       = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          200, 1e-7),
    )

    print(f"\n  RMS reprojection error: {rms:.4f} px")
    if rms > 2.0:
        print("  WARNING: RMS > 2.0 px — calibration may be poor. "
              "Check that your Aprilgrid was flat and clearly visible.")
    elif rms < 1.0:
        print("  Calibration looks good.")

    print(f"\n  R (IMX219 → fisheye1):\n{R}")
    print(f"\n  T (IMX219 → fisheye1, metres):\n{T.flatten()}")
    print(f"\n  Baseline: {np.linalg.norm(T)*100:.1f} mm")

    # ── Save output ──────────────────────────────────────────────────────────
    output = {
        "source":      "imx219",
        "target":      "t265_fisheye1",
        "rms_px":      float(rms),
        "n_pairs":     int(n_valid),
        "R_imx_to_fisheye1": R.tolist(),
        "T_imx_to_fisheye1": T.flatten().tolist(),
    }

    with open(args.output, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False)

    print(f"\n  Saved extrinsics to: {args.output}")
    print("\nDone. Load into your node with:")
    print("  import yaml, numpy as np")
    print(f"  with open('{args.output}') as f: ex = yaml.safe_load(f)")
    print("  R_imx_to_t265 = np.array(ex['R_imx_to_fisheye1'])")
    print("  T_imx_to_t265 = np.array(ex['T_imx_to_fisheye1'])")


if __name__ == "__main__":
    main()