#!/usr/bin/env python3
"""
sdv_autonomous.py — QCar2 Complete Autonomous System with Sensor Fusion
========================================================================
ONE file. ONE command. Everything inside.

Sensor Fusion integrated:
  - CSI Front Camera  (vision_score  — lane edge quality)
  - RealSense Depth   (depth_score   — road clear ahead)
  - fusion_confidence = vision_score × depth_score

  confidence >= 0.6  → DRIVING  (full speed)
  confidence >= 0.3  → CAUTION  (half speed)
  confidence <  0.3  → STOP
  confidence == 0.0  → STOP (obstacle / sync lost / no lane)

LiDAR still handles hard emergency stop at distance threshold.

Usage:
    python3 sdv_autonomous.py            # full autonomous
    python3 sdv_autonomous.py --preview  # no motors, just detection
    python3 sdv_autonomous.py --stop 30  # stop at 30cm
"""

import sys, os, time, threading, argparse, subprocess, signal
import numpy as np
import cv2

# Auto-detect headless mode
HEADLESS = ("DISPLAY" not in os.environ)

if HEADLESS:
    os.environ.pop("DISPLAY", None)
    os.environ.pop("XAUTHORITY", None)
    # Prevent Qt from trying to initialize a display
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan

try:
    from pal.products.qcar import QCar, QCarLidar, QCarCameras
    HAS_QCAR = True
except ImportError:
    HAS_QCAR = False
    print("[SDV] WARNING: QCar PAL not found — preview mode only.")

from lane_detector import LaneDetector
from config import (
    BEV_WIDTH, BEV_HEIGHT, CAMERA_FPS,
    STEER_MAX_RAD, SPEED_BASE, SPEED_MIN, SPEED_MAX,
    PID_KP, PID_KI, PID_KD, PID_INTEGRAL_MAX,
    NO_LANE_STOP_FRAMES, LEDS_DEFAULT,
)


# ══════════════════════════════════════════════════════════════════
#  SENSOR FUSION CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Resolution alignment: Depth (1280x720) → CSI (820x616)
ALIGN_W = 820
ALIGN_H = 616

# Timestamp sync window
SYNC_WINDOW_MS = 100.0

# Depth ROI — fraction of aligned frame (same spatial region as CSI trapezoid)
DEPTH_ROI_TOP    = 0.4
DEPTH_ROI_BOTTOM = 0.75
DEPTH_ROI_LEFT   = 0.25
DEPTH_ROI_RIGHT  = 0.75

# Obstacle detection
OBSTACLE_DIST_M       = 0.5   # < 0.5m in ROI = obstacle = stop
MIN_VALID_DEPTH       = 0.1
MAX_VALID_DEPTH       = 5.0

# Vision score
VISION_MIN_LANE_PIXELS = 500

# Speed thresholds based on fusion confidence
SPEED_FULL    = SPEED_BASE          # confidence >= 0.6
SPEED_CAUTION = SPEED_BASE * 0.5   # confidence >= 0.3
# confidence <  0.3 → stop


# ══════════════════════════════════════════════════════════════════
#  ROS2 Node Launcher
# ══════════════════════════════════════════════════════════════════

class ROSNodeLauncher:
    def __init__(self):
        self._procs = []
        self._source = (
            "source /opt/ros/humble/setup.bash && "
            "source /home/nvidia/Documents/Quanser/5_research/sdcs/qcar2/ros2/install/setup.bash && "
            "source /home/nvidia/ros2/install/setup.bash"
        )

    def launch(self, package, executable, env_prefix=""):
        cmd  = f"{env_prefix}{self._source} && ros2 run {package} {executable}"
        proc = subprocess.Popen(
            cmd, shell=True, executable='/bin/bash',
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        self._procs.append(proc)
        print(f"[Launcher] Started: ros2 run {package} {executable} (pid={proc.pid})")
        return proc

    def stop_all(self):
        for proc in self._procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        self._procs.clear()
        subprocess.run(["pkill", "-9", "-f", "qcar2_nodes"], capture_output=True)
        print("[Launcher] ROS2 nodes stopped.")


# ══════════════════════════════════════════════════════════════════
#  Hardware Stop
# ══════════════════════════════════════════════════════════════════

def hardware_stop(car=None):
    print("[SDV] Stopping hardware via Quanser PAL...")
    try:
        if car is not None:
            try:
                car.write(0.0, 0.0, np.zeros(8))
                car.__exit__(None, None, None)
            except Exception:
                pass
        myLidar = QCarLidar()
        myCar   = QCar()
        myCar.terminate()
        myLidar.terminate()
        print("[SDV] LiDAR stopped ✅  QCar stopped ✅")
    except Exception as e:
        print(f"[SDV] Hardware stop error: {e}")


# ══════════════════════════════════════════════════════════════════
#  PID Controller
# ══════════════════════════════════════════════════════════════════

class PID:
    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD):
        self.kp=kp; self.ki=ki; self.kd=kd
        self._integral=0.0; self._prev_err=0.0; self._prev_time=None

    def reset(self):
        self._integral=0.0; self._prev_err=0.0; self._prev_time=None

    def update(self, error):
        now = time.time()
        dt  = max((now - self._prev_time) if self._prev_time else 0.033, 1e-6)
        self._prev_time = now
        self._integral  = np.clip(self._integral + error*dt,
                                  -PID_INTEGRAL_MAX, PID_INTEGRAL_MAX)
        deriv = (error - self._prev_err) / dt
        self._prev_err = error
        return float(np.clip(
            self.kp*error + self.ki*self._integral + self.kd*deriv,
            -STEER_MAX_RAD, STEER_MAX_RAD
        ))


# ══════════════════════════════════════════════════════════════════
#  Sensor Node — RealSense RGB + Depth + LiDAR via ROS2
# ══════════════════════════════════════════════════════════════════

class SensorNode(Node):
    def __init__(self):
        super().__init__('sdv_autonomous')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self._lock       = threading.Lock()
        self._rgb        = None
        self._depth      = None
        self._depth_ts   = None
        self._lidar_dist = 99.0

        self.create_subscription(Image,     '/camera/color_image', self._rgb_cb,   qos)
        self.create_subscription(Image,     '/camera/depth_image', self._depth_cb, qos)
        self.create_subscription(LaserScan, '/scan',               self._lidar_cb, qos)

    def _rgb_cb(self, msg):
        if len(msg.data) == 0: return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        with self._lock: self._rgb = frame.copy()

    def _depth_cb(self, msg):
        if len(msg.data) == 0: return
        raw   = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)
        depth = raw.astype(np.float32) * 0.001  # mm → meters
        with self._lock:
            self._depth    = depth
            self._depth_ts = time.time()

    def _lidar_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(np.degrees(msg.angle_min),
                             np.degrees(msg.angle_max), len(ranges))
        mask = (np.abs(angles) < 15) & (ranges > 0.05) & (ranges < 10.0)
        with self._lock:
            self._lidar_dist = float(np.min(ranges[mask])) if mask.any() else 99.0

    def read(self):
        with self._lock:
            return (
                self._rgb.copy()   if self._rgb   is not None else None,
                self._depth.copy() if self._depth is not None else None,
                self._depth_ts,
                self._lidar_dist
            )


# ══════════════════════════════════════════════════════════════════
#  SENSOR FUSION FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def align_depth_to_csi(depth_frame):
    """
    Resize depth (1280x720) → CSI resolution (820x616).
    INTER_NEAREST preserves real depth values — no interpolation artefacts.
    After this, pixel (x,y) in depth = same spatial location as (x,y) in CSI.
    """
    return cv2.resize(
        depth_frame,
        (ALIGN_W, ALIGN_H),
        interpolation=cv2.INTER_NEAREST
    )


def compute_vision_score(csi_frame):
    """
    Score lane detection quality from CSI front camera.
    Canny edges inside trapezoid lane mask → score 0.0-1.0.
    """
    h, w     = csi_frame.shape[:2]
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
    return vision_score


def compute_depth_score(depth_aligned):
    """
    Score road clearance from aligned depth frame (820x616).
    ROI fractions now match same spatial region as CSI trapezoid.
    Obstacle < OBSTACLE_DIST_M → score = 0.0 → STOP.
    """
    h, w = depth_aligned.shape[:2]

    r_top   = int(h * DEPTH_ROI_TOP)
    r_bot   = int(h * DEPTH_ROI_BOTTOM)
    r_left  = int(w * DEPTH_ROI_LEFT)
    r_right = int(w * DEPTH_ROI_RIGHT)
    roi     = depth_aligned[r_top:r_bot, r_left:r_right]

    valid_mask  = (roi > MIN_VALID_DEPTH) & (roi < MAX_VALID_DEPTH)
    valid_ratio = float(np.sum(valid_mask)) / max(roi.size, 1)

    obs_mask   = (roi > MIN_VALID_DEPTH) & (roi < OBSTACLE_DIST_M)
    obs_pixels = int(np.sum(obs_mask))
    obstacle   = obs_pixels > 50

    depth_score = 0.0 if obstacle else float(np.clip(valid_ratio / 0.5, 0.0, 1.0))
    return depth_score, obstacle


def compute_fusion_confidence(csi_frame, depth_frame, depth_ts, csi_ts):
    """
    Fuse CSI vision score + Depth score → final lane confidence.

      1. vision_score  = lane edge quality from CSI (0.0-1.0)
      2. depth_aligned = resize depth to CSI resolution (820x616)
      3. depth_score   = road clearance from aligned depth (0.0-1.0)
      4. sync check    = timestamps within SYNC_WINDOW_MS
      5. fusion_conf   = vision_score × depth_score  (0.0 if out of sync)

    Returns:
        fusion_conf  : float 0.0-1.0
        vision_score : float 0.0-1.0
        depth_score  : float 0.0-1.0
        obstacle     : bool
        sync_ok      : bool
        delta_ms     : float
    """
    vision_score = compute_vision_score(csi_frame)

    if depth_frame is not None and depth_ts is not None:
        depth_aligned        = align_depth_to_csi(depth_frame)
        depth_score, obstacle = compute_depth_score(depth_aligned)
        delta_ms             = abs(csi_ts - depth_ts) * 1000.0
        sync_ok              = delta_ms <= SYNC_WINDOW_MS
    else:
        depth_score = 0.0
        obstacle    = False
        delta_ms    = 9999.0
        sync_ok     = False

    fusion_conf = float(vision_score * depth_score) if sync_ok else 0.0
    return fusion_conf, vision_score, depth_score, obstacle, sync_ok, delta_ms


# ══════════════════════════════════════════════════════════════════
#  Speed from Fusion Confidence
# ══════════════════════════════════════════════════════════════════

def speed_from_confidence(confidence, steer_cmd):
    """
    confidence >= 0.6 → full speed (scaled by steering angle)
    confidence >= 0.3 → half speed (caution)
    confidence <  0.3 → 0.0 (stop)
    """
    if confidence >= 0.6:
        base = SPEED_FULL
    elif confidence >= 0.3:
        base = SPEED_CAUTION
    else:
        return 0.0

    return float(np.clip(
        base * max(0.5, np.cos(abs(steer_cmd))),
        SPEED_MIN, SPEED_MAX
    ))


# ══════════════════════════════════════════════════════════════════
#  Ego Line
# ══════════════════════════════════════════════════════════════════

def draw_ego_line(frame, detector):
    if detector.left_fit_px is None and detector.right_fit_px is None:
        return frame
    if detector._M_bev_inv is None:
        return frame

    h, w   = frame.shape[:2]
    plot_y = np.linspace(BEV_HEIGHT - 1, 0, 40).astype(int)

    if detector.left_fit_px is not None and detector.right_fit_px is not None:
        lx = np.polyval(detector.left_fit_px,  plot_y)
        rx = np.polyval(detector.right_fit_px, plot_y)
        cx = ((lx + rx) / 2.0).astype(int)
    elif detector.left_fit_px is not None:
        cx = (np.polyval(detector.left_fit_px, plot_y) + 50).astype(int)
    else:
        cx = (np.polyval(detector.right_fit_px, plot_y) - 50).astype(int)

    cx      = np.clip(cx, 0, BEV_WIDTH - 1)
    bev_pts = np.array([[c, y] for c, y in zip(cx, plot_y)],
                       dtype=np.float32).reshape(-1, 1, 2)
    cam_pts = cv2.perspectiveTransform(bev_pts, detector._M_bev_inv)
    if cam_pts is None: return frame

    cam_pts = cam_pts.reshape(-1, 2).astype(int)
    for i in range(len(cam_pts) - 1):
        p1, p2 = tuple(cam_pts[i]), tuple(cam_pts[i+1])
        if 0<=p1[0]<w and 0<=p1[1]<h and 0<=p2[0]<w and 0<=p2[1]<h:
            cv2.line(frame, p1, p2, (0, 255, 255), 5)

    valid = [(int(pt[0]), int(pt[1])) for pt in cam_pts
             if 0<=pt[0]<w and 0<=pt[1]<h]
    if len(valid) >= 4:
        sorted_pts = sorted(valid, key=lambda p: p[1], reverse=True)
        cv2.arrowedLine(frame, sorted_pts[1], sorted_pts[3],
                        (0, 255, 255), 5, tipLength=0.3)
        cv2.putText(frame, "EGO PATH",
                    (sorted_pts[3][0] + 10, sorted_pts[3][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ══════════════════════════════════════════════════════════════════
#  HUD — with fusion confidence
# ══════════════════════════════════════════════════════════════════

def draw_hud(frame, speed, steer, offset,
             fusion_conf, vision_score, depth_score,
             obstacle, sync_ok, delta_ms,
             lidar_dist, state, fps, stop_dist_m, stopped):
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (5,5), (680, 230), (15,15,15), cv2.FILLED)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    sc   = (0,255,120) if state=='DRIVING'  else \
           (0,200,255) if state=='CAUTION'  else \
           (0,0,255)   if state in ('E-STOP','STOPPED','OBSTACLE') else (0,200,255)
    lc   = (0,0,255) if (stop_dist_m>0 and lidar_dist<=stop_dist_m) else (0,255,0)
    cc   = (0,220,0) if fusion_conf>=0.6 else \
           (0,200,255) if fusion_conf>=0.3 else (0,0,255)

    lines = [
        (f"State: {state}   FPS: {fps}",                                        sc),
        (f"Speed: {speed:.3f}   Steer: {steer:+.3f} ({np.degrees(steer):+.1f}deg)", (220,220,220)),
        (f"Lane Offset: {offset:+.4f}m",                                        (220,220,220)),
        (f"Vision: {vision_score:.2f}   Depth: {depth_score:.2f}   "
         f"Fusion: {fusion_conf:.2f} ({fusion_conf:.0%})",                      cc),
        (f"Sync: {delta_ms:.0f}ms ({'OK' if sync_ok else 'BAD'})   "
         f"Obstacle: {'YES' if obstacle else 'No'}",
         (0,255,0) if sync_ok else (0,0,255)),
        (f"LiDAR: {lidar_dist*100:.0f}cm" +
         (f"  [STOP@{stop_dist_m*100:.0f}cm]" if stop_dist_m>0 else ""),        lc),
        ("ESC=quit  P=pause  R=resume  S=screenshot",                           (150,150,150)),
    ]

    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (15, 28+i*27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)

    # Fusion confidence bar
    bx, by, bw = w-215, 15, 200
    cv2.rectangle(frame, (bx,by), (bx+bw,by+14), (50,50,50), cv2.FILLED)
    cv2.rectangle(frame, (bx,by), (bx+int(bw*fusion_conf),by+14), cc, cv2.FILLED)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+14), (180,180,180), 1)
    cv2.putText(frame, f"Fusion: {fusion_conf:.0%}",
                (bx, by+32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    # STOPPED banner
    if stopped or state in ('STOPPED','OBSTACLE'):
        reason = "OBSTACLE" if obstacle else f"{lidar_dist*100:.0f}cm"
        cv2.rectangle(frame, (w//2-230,h//2-55), (w//2+230,h//2+55), (0,0,0), cv2.FILLED)
        cv2.putText(frame, f"STOPPED — {reason}",
                    (w//2-215, h//2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

    # CAUTION banner
    if state == 'CAUTION':
        cv2.rectangle(frame, (w//2-190,h//2-40), (w//2+190,h//2+40), (0,0,0), cv2.FILLED)
        cv2.putText(frame, "CAUTION — LOW CONFIDENCE",
                    (w//2-180, h//2+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    return frame


# ══════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', action='store_true')
    p.add_argument('--stop', type=float, default=50.0)
    p.add_argument('--headless', action='store_true', help='Skip all GUI visualization')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    args        = parse_args()
    preview     = args.preview or not HAS_QCAR
    stop_dist_m = args.stop / 100.0
    
    global HEADLESS
    if args.headless:
        HEADLESS = True

    print("╔══════════════════════════════════════════════════════╗")
    print("║  QCar2 SDV — Autonomous + Sensor Fusion             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Mode          : {'PREVIEW' if preview else 'AUTONOMOUS'}")
    print(f"  LiDAR stop    : {args.stop:.0f}cm (hard stop always active)")
    print(f"  Fusion        : vision_score × depth_score")
    print(f"  Speed control : Full≥0.6 | Caution≥0.3 | Stop<0.3")
    print(f"  Depth align   : (1280x720) → ({ALIGN_W}x{ALIGN_H}) INTER_NEAREST")
    print()

    # ── Launch ROS2 sensor nodes ─────────────────────────────────
    launcher = ROSNodeLauncher()
    print("[SDV] Launching RealSense (with librealsense2 fix)...")
    launcher.launch(
        'qcar2_nodes', 'rgbd',
        env_prefix="LD_PRELOAD=/opt/ros/humble/lib/aarch64-linux-gnu/"
                   "librealsense2.so.2.54.1 "
    )
    time.sleep(2)
    print("[SDV] Launching LiDAR...")
    launcher.launch('qcar2_nodes', 'lidar')
    time.sleep(2)

    # ── ROS2 ─────────────────────────────────────────────────────
    rclpy.init()
    sensors    = SensorNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(sensors,), daemon=True)
    ros_thread.start()
    print("[ROS2] Sensors ready ✅")

    # ── Lane detector + PID ──────────────────────────────────────
    detector = LaneDetector()
    pid      = PID()

    # ── CSI Front Camera ─────────────────────────────────────────
    csi_cameras = None
    try:
        print("[CSI] Initializing Front Camera (index=2)...")
        csi_cameras = QCarCameras(
            frameWidth=820, frameHeight=616, frameRate=80,
            enableRight=False, enableBack=False,
            enableFront=True, enableLeft=False
        )
        print("[CSI] Camera ready ✅")
    except Exception as e:
        print(f"[CSI] Failed: {e} — will use RealSense RGB as fallback")

    # ── QCar2 hardware ───────────────────────────────────────────
    car = None
    if not preview:
        try:
            car = QCar(readMode=1, frequency=100)
            car.__enter__()
            print("[QCar2] Hardware connected ✅")
        except Exception as e:
            print(f"[QCar2] Failed: {e} — switching to preview.")
            preview = True

    # ── Wait for RealSense RGB ───────────────────────────────────
    print("[Camera] Waiting for RealSense RGB frames...")
    for _ in range(100):
        rgb, _, _, _ = sensors.read()
        if rgb is not None: break
        time.sleep(0.1)
    if rgb is None:
        print("[ERROR] No RGB frames! Check RealSense connection.")
        hardware_stop(car)
        launcher.stop_all()
        return

    print("[Camera] Ready ✅")
    print(f"\n[SDV] LiDAR hard-stop at {stop_dist_m*100:.0f}cm ACTIVE ✅")
    print("[SDV] Fusion confidence controls speed ACTIVE ✅")
    print("Controls: ESC=quit  P=pause  R=resume  S=screenshot\n")

    if not HEADLESS:
        try:
            cv2.namedWindow("QCar2 SDV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("QCar2 SDV", 1280, 720)
        except Exception:
            pass

    paused=False; stopped=False; state='WAITING'
    fps=0; frame_count=0; fps_timer=time.time()
    speed_cmd=0.0; steer_cmd=0.0

    try:
        while True:
            t0 = time.time()

            # ── Read all sensors ──────────────────────────────────
            rgb, depth_frame, depth_ts, lidar_dist = sensors.read()
            if rgb is None: time.sleep(0.01); continue

            # ── CSI front camera (for vision score) ───────────────
            csi_frame = None
            if csi_cameras is not None:
                try:
                    csi_cameras.readAll()
                    img = csi_cameras.csi[2].imageData
                    if img is not None and img.max() > 10:
                        csi_frame = img
                except Exception:
                    pass
            if csi_frame is None:
                csi_frame = rgb   # fallback to RealSense RGB

            csi_ts = time.time()

            # ── FPS ───────────────────────────────────────────────
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps=frame_count; frame_count=0; fps_timer=now

            # ── Pause ─────────────────────────────────────────────
            if paused:
                display = rgb.copy()
                cv2.putText(display, "PAUSED — press P to resume",
                            (50, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                cv2.imshow("QCar2 SDV", display)
                if car: car.write(0.0, 0.0, np.zeros(8))
                key = cv2.waitKey(30) & 0xFF
                if key==ord('p') or key==ord('P'): paused=False
                elif key==27: break
                continue

            # ── Lane detection + ego line ─────────────────────────
            overlay = detector.process(rgb, depth_frame)
            overlay = draw_ego_line(overlay, detector)
            offset  = detector.center_offset_m
            no_lane = detector.no_lane_count

            # ── SENSOR FUSION CONFIDENCE ──────────────────────────
            fusion_conf, vision_score, depth_score, obstacle, sync_ok, delta_ms = \
                compute_fusion_confidence(csi_frame, depth_frame, depth_ts, csi_ts)

            # ── State Machine ─────────────────────────────────────
            if no_lane >= NO_LANE_STOP_FRAMES:
                state='E-STOP'; speed_cmd=0.0; steer_cmd=0.0; pid.reset()

            elif stopped:
                state='STOPPED'; speed_cmd=0.0; steer_cmd=0.0

            elif lidar_dist <= stop_dist_m:
                # LiDAR hard stop — always active regardless of fusion
                stopped=True; state='STOPPED'
                speed_cmd=0.0; steer_cmd=0.0
                print(f"\n[LIDAR STOP] Object at {lidar_dist*100:.1f}cm "
                      f"(threshold: {stop_dist_m*100:.0f}cm)")

            elif obstacle or fusion_conf == 0.0:
                # Depth obstacle OR sync lost OR no vision
                state='OBSTACLE'; speed_cmd=0.0; steer_cmd=0.0

            elif fusion_conf >= 0.3:
                # Compute steering from lane offset (PID)
                steer_cmd = pid.update(-offset / 0.008)
                # Speed controlled by fusion confidence
                speed_cmd = speed_from_confidence(fusion_conf, steer_cmd)
                state     = 'DRIVING' if fusion_conf >= 0.6 else 'CAUTION'

            else:
                state='LOST_LANE'; speed_cmd=0.0; steer_cmd=0.0

            # ── LEDs ──────────────────────────────────────────────
            leds = LEDS_DEFAULT.copy()
            if steer_cmd >  0.05: leds[0]=leds[2]=1
            if steer_cmd < -0.05: leds[1]=leds[3]=1
            if state in ('E-STOP','STOPPED','OBSTACLE'): leds[4]=leds[5]=1
            if state == 'CAUTION': leds[6]=leds[7]=1

            # ── Send to hardware ──────────────────────────────────
            if car and not preview:
                try: car.write(speed_cmd, steer_cmd, leds)
                except Exception as e: print(f"[QCar2] Write error: {e}")

            # ── HUD ───────────────────────────────────────────────
            display = draw_hud(
                overlay, speed_cmd, steer_cmd, offset,
                fusion_conf, vision_score, depth_score,
                obstacle, sync_ok, delta_ms,
                lidar_dist, state, fps, stop_dist_m, stopped
            )

            if not HEADLESS:
                try:
                    cv2.imshow("QCar2 SDV", display)
                except Exception:
                    pass
                key = cv2.waitKey(1) & 0xFF
            else:
                # In headless, just print status every 30 frames
                if frame_count % 30 == 0:
                    print(f"[SDV] State: {state} | Fusion: {fusion_conf:.2f} | Speed: {speed_cmd:.2f} | Steer: {steer_cmd:+.2f}")
                key = -1

            if key==27: break
            elif key==ord('p') or key==ord('P'): paused=True
            elif key==ord('r') or key==ord('R'):
                stopped=False; state='WAITING'; pid.reset()
                print("[Resumed]")
            elif key==ord('s') or key==ord('S'):
                path = f"{SCRIPTS_DIR}/screenshot_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(path, display)
                print(f"[Screenshot] {path}")

            elapsed = time.time() - t0
            sleep_t = (1.0/CAMERA_FPS) - elapsed
            if sleep_t > 0: time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted!")

    finally:
        print("\n[Shutdown] Stopping everything safely...")
        hardware_stop(car)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if csi_cameras is not None:
            try: csi_cameras.terminate()
            except Exception: pass
        try:
            sensors.destroy_node()
            rclpy.shutdown()
        except Exception: pass
        launcher.stop_all()
        print("[Done] ✅")


if __name__ == "__main__":
    main()
