#!/usr/bin/env python3
"""
sensor_fusion.py — QCar2 Sensor Fusion: CSI Front + RealSense Depth
=====================================================================
HEADLESS MODE — no display needed, runs over SSH.

Formula:
    final_confidence = vision_score x depth_score

    vision_score  = lane detection quality from CSI Front Camera (0.0 - 1.0)
    depth_score   = road clear ahead from RealSense Depth       (0.0 - 1.0)

    If obstacle in lane → depth_score = 0.0 → final_confidence = 0.0 → STOP
    If timestamp sync bad (>100ms) → confidence = 0.0

Output:
    - Prints fusion table to terminal every frame
    - Auto-saves annotated JPG every 30 frames to scripts folder

Usage:
    # Terminal 1
    LD_PRELOAD=/opt/ros/humble/lib/aarch64-linux-gnu/librealsense2.so.2.54.1 ros2 run qcar2_nodes rgbd

    # Terminal 2
    python3 sensor_fusion.py
"""

import sys, os, time, threading

# Headless — no display
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from pal.products.qcar import QCarCameras


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SAVE_DIR         = "/home/nvidia/Desktop/SDV_workspace/scripts"
SYNC_WINDOW_MS   = 100.0
SCREENSHOT_EVERY = 30

# ── RESOLUTION ALIGNMENT ──────────────────────────────────────────
# CSI Front  : 820 x 616  (source of truth — this is what car sees)
# Depth raw  : 1280 x 720 (needs to be resized DOWN to match CSI)
# After align: both are 820 x 616 so pixel (x,y) means same location
ALIGN_W = 820   # target width  — CSI native
ALIGN_H = 616   # target height — CSI native

# Depth ROI — defined as FRACTION of ALIGNED resolution (820x616)
# These fractions now mean the same spatial region in both frames
DEPTH_ROI_TOP    = 0.4
DEPTH_ROI_BOTTOM = 0.75
DEPTH_ROI_LEFT   = 0.25
DEPTH_ROI_RIGHT  = 0.75

OBSTACLE_DIST_M       = 0.5
MIN_VALID_DEPTH       = 0.1
MAX_VALID_DEPTH       = 5.0
VISION_MIN_LANE_PIXELS = 500


# ══════════════════════════════════════════════════════════════════
#  Depth ROS2 Subscriber
# ══════════════════════════════════════════════════════════════════

class DepthSubscriber(Node):
    def __init__(self):
        super().__init__('fusion_depth_node')
        self._lock      = threading.Lock()
        self._depth     = None
        self._timestamp = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/depth_image', self._cb, qos)

    def _cb(self, msg):
        if len(msg.data) == 0:
            return
        raw   = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth = raw.astype(np.float32) * 0.001
        with self._lock:
            self._depth     = depth
            self._timestamp = time.time()

    def read(self):
        with self._lock:
            if self._depth is None:
                return None, None
            return self._depth.copy(), self._timestamp


# ══════════════════════════════════════════════════════════════════
#  Resolution Alignment
# ══════════════════════════════════════════════════════════════════

def align_depth_to_csi(depth_frame):
    """
    Resize depth frame from native (1280x720) → CSI resolution (820x616).

    Why: So pixel (x, y) in depth map corresponds to same spatial
    location as pixel (x, y) in CSI frame. Without this, ROI boxes
    and fusion comparisons are spatially incorrect.

    Uses INTER_NEAREST to avoid interpolating depth values across
    object boundaries (which would create false depth readings).

    Returns:
        aligned_depth : float32 (616, 820) in meters — same grid as CSI
        scale_x       : float — how much we scaled in x (for debug info)
        scale_y       : float — how much we scaled in y (for debug info)
    """
    orig_h, orig_w = depth_frame.shape[:2]

    aligned = cv2.resize(
        depth_frame,
        (ALIGN_W, ALIGN_H),
        interpolation=cv2.INTER_NEAREST  # preserve real depth values
    )

    scale_x = ALIGN_W / orig_w   # 820/1280 = 0.640
    scale_y = ALIGN_H / orig_h   # 616/720  = 0.856

    return aligned, scale_x, scale_y


# ══════════════════════════════════════════════════════════════════
#  Vision Score
# ══════════════════════════════════════════════════════════════════

def compute_vision_score(csi_frame):
    h, w  = csi_frame.shape[:2]
    debug = csi_frame.copy()

    road_top = int(h * 0.45)
    road_roi = csi_frame[road_top:h, :]

    gray    = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    rh, rw = edges.shape
    mask   = np.zeros_like(edges)
    trap   = np.array([[
        (int(rw * 0.05), rh),
        (int(rw * 0.95), rh),
        (int(rw * 0.65), int(rh * 0.1)),
        (int(rw * 0.35), int(rh * 0.1)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trap, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lane_pixels  = int(np.sum(masked_edges > 0))
    vision_score = float(np.clip(lane_pixels / VISION_MIN_LANE_PIXELS, 0.0, 1.0))

    # Annotate debug
    trap_draw = trap.copy()
    trap_draw[0, :, 1] += road_top
    cv2.polylines(debug, trap_draw, True, (0, 255, 0), 2)
    edge_col = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    edge_col[:, :, 0] = 0; edge_col[:, :, 2] = 0
    debug[road_top:h, :] = cv2.addWeighted(debug[road_top:h, :], 0.7, edge_col, 0.3, 0)
    cv2.putText(debug, f"Lane px:{lane_pixels}  VisionScore:{vision_score:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vision_score, lane_pixels, debug


# ══════════════════════════════════════════════════════════════════
#  Depth Score
# ══════════════════════════════════════════════════════════════════

def compute_depth_score(depth_frame):
    h, w = depth_frame.shape[:2]

    r_top   = int(h * DEPTH_ROI_TOP)
    r_bot   = int(h * DEPTH_ROI_BOTTOM)
    r_left  = int(w * DEPTH_ROI_LEFT)
    r_right = int(w * DEPTH_ROI_RIGHT)
    roi     = depth_frame[r_top:r_bot, r_left:r_right]

    valid_mask  = (roi > MIN_VALID_DEPTH) & (roi < MAX_VALID_DEPTH)
    valid_ratio = float(np.sum(valid_mask)) / max(roi.size, 1)

    obs_mask   = (roi > MIN_VALID_DEPTH) & (roi < OBSTACLE_DIST_M)
    obs_pixels = int(np.sum(obs_mask))
    obstacle   = obs_pixels > 50

    depth_score = 0.0 if obstacle else float(np.clip(valid_ratio / 0.5, 0.0, 1.0))

    # Annotate depth debug
    disp  = np.clip(depth_frame / MAX_VALID_DEPTH, 0, 1)
    debug = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    box_col = (0, 0, 255) if obstacle else (0, 255, 0)
    cv2.rectangle(debug, (r_left, r_top), (r_right, r_bot), box_col, 2)
    cv2.putText(debug, "OBSTACLE!" if obstacle else "CLEAR",
                (r_left, r_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_col, 2)
    cv2.putText(debug, f"Valid:{valid_ratio:.0%}  ObsPx:{obs_pixels}  DepthScore:{depth_score:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return depth_score, obstacle, obs_pixels, valid_ratio, debug


# ══════════════════════════════════════════════════════════════════
#  Fusion
# ══════════════════════════════════════════════════════════════════

def fuse(vision_score, depth_score, csi_ts, depth_ts):
    delta_ms   = abs(csi_ts - depth_ts) * 1000.0
    sync_ok    = delta_ms <= SYNC_WINDOW_MS
    confidence = float(vision_score * depth_score) if sync_ok else 0.0
    return confidence, sync_ok, delta_ms


# ══════════════════════════════════════════════════════════════════
#  Save annotated image
# ══════════════════════════════════════════════════════════════════

def save_fusion_image(csi_debug, depth_debug, vision_score, depth_score,
                      confidence, sync_ok, delta_ms, obstacle, frame_num):
    csi_h, csi_w = csi_debug.shape[:2]
    depth_resized = cv2.resize(depth_debug, (csi_w, csi_h))
    combined = np.hstack([csi_debug, depth_resized])

    bar = np.zeros((55, combined.shape[1], 3), dtype=np.uint8)
    conf_col = (0, 220, 0) if confidence >= 0.6 else \
               (0, 200, 255) if confidence >= 0.3 else (0, 0, 255)
    cv2.putText(bar,
        f"Frame#{frame_num}  Vision:{vision_score:.2f}  Depth:{depth_score:.2f}  "
        f"Sync:{delta_ms:.0f}ms({'OK' if sync_ok else 'BAD'})  "
        f"Obstacle:{'YES' if obstacle else 'No'}  "
        f"CONFIDENCE:{confidence:.2f}({confidence:.0%})",
        (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_col, 2, cv2.LINE_AA)

    final = np.vstack([combined, bar])
    path  = os.path.join(SAVE_DIR, f"fusion_f{frame_num}_{time.strftime('%H%M%S')}.jpg")
    cv2.imwrite(path, final)
    return path


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   QCar2 Sensor Fusion — Headless Mode               ║")
    print("║   CSI Front (2) + RealSense Depth                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  Resolution alignment : Depth (1280x720) → CSI ({ALIGN_W}x{ALIGN_H})")
    print(f"  Scale factors        : x={820/1280:.3f}  y={616/720:.3f}")
    print(f"  Interpolation        : INTER_NEAREST (preserves depth values)")
    print(f"  Screenshots saved every {SCREENSHOT_EVERY} frames → {SAVE_DIR}\n")

    rclpy.init()
    depth_node = DepthSubscriber()
    threading.Thread(target=rclpy.spin, args=(depth_node,), daemon=True).start()
    print("[ROS2] Depth subscriber started ✅")

    cameras = None
    try:
        print("[CSI] Initializing Front Camera (index=2)...")
        cameras = QCarCameras(
            frameWidth=820, frameHeight=616, frameRate=80,
            enableRight=False, enableBack=False,
            enableFront=True, enableLeft=False)
        print("[CSI] Camera ready ✅")
    except Exception as e:
        print(f"[CSI ERROR] {e}"); return

    print("[Depth] Waiting for first depth frame...")
    for _ in range(50):
        d, _ = depth_node.read()
        if d is not None: break
        time.sleep(0.1)
    print("[Depth] Ready ✅")
    print("\nPress Ctrl+C to stop.\n")

    # Table header
    print(f"{'Frame':<7} {'Vision':>7} {'Depth':>7} {'Conf':>7} "
          f"{'Sync ms':>9} {'Obstacle':>10}  Status")
    print("─" * 62)

    total_frames = 0

    try:
        while True:
            t0 = time.time()

            cameras.readAll()
            csi_frame = cameras.csi[2].imageData
            csi_ts    = time.time()

            if csi_frame is None or csi_frame.max() <= 10:
                time.sleep(0.01); continue

            total_frames += 1
            depth_frame, depth_ts = depth_node.read()

            vision_score, lane_px, csi_debug = compute_vision_score(csi_frame)

            if depth_frame is not None and depth_ts is not None:
                # ── ALIGN: resize depth (1280x720) → CSI (820x616) ──
                depth_aligned, sx, sy = align_depth_to_csi(depth_frame)

                depth_score, obstacle, obs_px, valid_ratio, depth_debug = \
                    compute_depth_score(depth_aligned)
                confidence, sync_ok, delta_ms = fuse(
                    vision_score, depth_score, csi_ts, depth_ts)
            else:
                depth_score = 0.0; obstacle = False
                obs_px = 0; valid_ratio = 0.0
                sync_ok = False; delta_ms = 9999.0; confidence = 0.0
                depth_debug = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(depth_debug, "NO DEPTH FRAME",
                            (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

            # Status
            if   confidence == 0.0 and obstacle:    status = "STOP-OBSTACLE"
            elif confidence == 0.0 and not sync_ok: status = "STOP-SYNC"
            elif confidence == 0.0:                 status = "STOP-NOLANE"
            elif confidence >= 0.6:                 status = "DRIVING ✅"
            elif confidence >= 0.3:                 status = "CAUTION ⚠️"
            else:                                   status = "LOW-CONF"

            print(f"{total_frames:<7} {vision_score:>7.2f} {depth_score:>7.2f} "
                  f"{confidence:>7.2f} {delta_ms:>9.0f} "
                  f"{'YES' if obstacle else 'No':>10}  {status}")

            # Auto-save
            if total_frames % SCREENSHOT_EVERY == 0:
                path = save_fusion_image(
                    csi_debug, depth_debug, vision_score, depth_score,
                    confidence, sync_ok, delta_ms, obstacle, total_frames)
                print(f"\n  ── [Saved] {path} ──\n")
                print(f"{'Frame':<7} {'Vision':>7} {'Depth':>7} {'Conf':>7} "
                      f"{'Sync ms':>9} {'Obstacle':>10}  Status")
                print("─" * 62)

            sleep_t = (1.0 / 30.0) - (time.time() - t0)
            if sleep_t > 0: time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\n[Stopped] Total frames: {total_frames}")

    finally:
        if cameras:
            try: cameras.terminate()
            except Exception: pass
        try:
            depth_node.destroy_node()
            rclpy.shutdown()
        except Exception: pass
        print("[Done] ✅")


if __name__ == "__main__":
    main()
