"""
qcar2_live_avoid.py
===================
QCar2 LIVE — Single command, everything inside.

Run
---
    python3 qcar2_live_avoid.py

That is it. This file:
  1. Launches RealSense ROS2 node internally
  2. Launches RPLidar ROS2 node internally
  3. Subscribes to both sensor topics
  4. Reads IMU + encoders via HAL
  5. Runs the avoidance + re-route state machine
  6. Writes throttle + steering to the car

Stop
----
  Ctrl+C  -> all sensor processes killed, car stops safely
"""

import os, sys, math, time, threading, subprocess, logging, enum
import numpy as np

os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image

try:
    from hal import QCar
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QCar2")


# ── TUNING ───────────────────────────────────────────────────────────────────
class Config:
    LOOP_HZ              = 30
    DT                   = 1.0 / 30

    FRONT_ARC_DEG        = 60.0
    SECTOR_DEG           = 20.0
    MIN_LIDAR_DIST       = 0.05
    MAX_LIDAR_DIST       = 5.0

    D_OBSTACLE           = 1.2
    D_CLEAR              = 1.6
    D_STOP               = 0.40
    D_RESUME             = 0.55

    RS_MIN_DEPTH_M       = 0.10
    RS_MAX_DEPTH_M       = 4.0
    RS_ROI               = (0.35, 0.65, 0.30, 0.70)
    RS_MIN_VALID_RATIO   = 0.15
    RS_OBSTACLE_DEPTH_M  = 1.0

    W_LIDAR              = 0.80
    W_RS                 = 0.20
    FUSION_THRESH        = 0.60

    THROTTLE_CRUISE      = 0.12
    THROTTLE_AVOID       = 0.08
    THROTTLE_REROUTE     = 0.09
    THROTTLE_MIN         = 0.05

    MAX_STEER            = 0.50
    AVOID_STEER_FRAC     = 0.65
    YAW_P_GAIN           = 1.8
    STEER_DEADZONE       = 0.02
    STEER_ALPHA          = 0.35

    CLEAR_CYCLES_NEEDED  = 15
    PAST_OBSTACLE_DIST_M = 0.40
    YAW_TOLERANCE_RAD    = 0.08
    REROUTE_TIMEOUT_S    = 5.0
    ENCODER_M_PER_TICK   = 0.0003
    ESTOP_HOLD_S         = 1.5

    LIDAR_TOPIC          = "/scan"
    DEPTH_TOPIC          = "/camera/depth/image_rect_raw"

    REALSENSE_LAUNCH_CMD = ["ros2", "launch", "realsense2_camera", "rs_launch.py"]
    LIDAR_LAUNCH_CMD     = ["ros2", "launch", "rplidar_ros",        "rplidar_launch.py"]
    SENSOR_WARMUP_S      = 4.0


# ── SENSOR LAUNCHER ──────────────────────────────────────────────────────────
class SensorLauncher:
    def __init__(self, cfg):
        self.cfg    = cfg
        self._procs = []

    def start(self):
        log.info("Launching RealSense...")
        self._procs.append(subprocess.Popen(
            self.cfg.REALSENSE_LAUNCH_CMD,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        log.info("Launching LiDAR...")
        self._procs.append(subprocess.Popen(
            self.cfg.LIDAR_LAUNCH_CMD,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        log.info("Waiting %.0fs for sensors to warm up...", self.cfg.SENSOR_WARMUP_S)
        time.sleep(self.cfg.SENSOR_WARMUP_S)
        log.info("Sensors ready")

    def stop(self):
        for p in self._procs:
            p.terminate()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: p.kill()
        self._procs.clear()
        log.info("All sensor processes stopped")


# ── STATE MACHINE ────────────────────────────────────────────────────────────
class State(enum.Enum):
    CRUISING       = "CRUISING"
    OBSTACLE       = "OBSTACLE"
    AVOIDING       = "AVOIDING"
    REROUTING      = "REROUTING"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class AvoidDir(enum.Enum):
    NONE  =  0
    LEFT  =  1
    RIGHT = -1

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── SENSOR FUSION ────────────────────────────────────────────────────────────
class SensorFusion:
    def __init__(self, cfg):
        self.cfg = cfg

    def process_lidar(self, raw_angles, raw_dists):
        cfg  = self.cfg
        a    = (raw_angles + math.pi) % (2 * math.pi) - math.pi
        ok   = (raw_dists > cfg.MIN_LIDAR_DIST) & (raw_dists < cfg.MAX_LIDAR_DIST)
        a, d = a[ok], raw_dists[ok]
        half  = cfg.FRONT_ARC_DEG / 2
        s     = cfg.SECTOR_DEG / 2
        front = np.abs(np.degrees(a)) <= half
        lm = front & (np.degrees(a) >  s)
        cm = front & (np.abs(np.degrees(a)) <= s)
        rm = front & (np.degrees(a) < -s)
        def mn(m): return float(np.mean(d[m])) if m.any() else cfg.MAX_LIDAR_DIST
        def mi(m): return float(np.min(d[m]))  if m.any() else cfg.MAX_LIDAR_DIST
        sectors  = {"left": mn(lm), "centre": mn(cm), "right": mn(rm)}
        cmin     = mi(cm)
        best_dir = AvoidDir.LEFT if sectors["left"] >= sectors["right"] else AvoidDir.RIGHT
        return sectors, cmin, best_dir

    def process_realsense(self, depth_mm):
        cfg   = self.cfg
        depth = depth_mm.astype(np.float32) / 1000.0
        h, w  = depth.shape
        t, b, l, r = cfg.RS_ROI
        roi   = depth[int(h*t):int(h*b), int(w*l):int(w*r)]
        vm    = (roi > cfg.RS_MIN_DEPTH_M) & (roi < cfg.RS_MAX_DEPTH_M)
        ratio = vm.sum() / vm.size
        if ratio < cfg.RS_MIN_VALID_RATIO:
            return False, 0.0
        conf = min(ratio / 0.50, 1.0)
        return float(np.median(roi[vm])) < cfg.RS_OBSTACLE_DEPTH_M, conf

    def fuse(self, angles, dists, depth_mm):
        cfg = self.cfg
        sectors, cmin, best_dir = self.process_lidar(angles, dists)
        rs_obs, rs_conf         = self.process_realsense(depth_mm)
        cm = sectors["centre"]
        if   cm < cfg.D_OBSTACLE: ls = 1.0
        elif cm > cfg.D_CLEAR:    ls = 0.0
        else: ls = 1.0 - (cm - cfg.D_OBSTACLE) / (cfg.D_CLEAR - cfg.D_OBSTACLE)
        fused = ls * cfg.W_LIDAR + float(rs_obs) * rs_conf * cfg.W_RS
        return {
            "is_obstacle"  : fused >= cfg.FUSION_THRESH,
            "is_emergency" : cmin  < cfg.D_STOP,
            "fused"        : round(fused, 3),
            "lidar_score"  : round(ls, 3),
            "rs_conf"      : round(rs_conf, 3),
            "sectors"      : {k: round(v, 2) for k, v in sectors.items()},
            "centre_min"   : round(cmin, 3),
            "best_dir"     : best_dir,
        }


# ── HEADING MEMORY + ODOMETER ────────────────────────────────────────────────
class HeadingMemory:
    def __init__(self):
        self.target_yaw = None
    def save(self, yaw):
        self.target_yaw = yaw
        log.info("Heading saved: %.1f deg", math.degrees(yaw))
    def error(self, yaw):
        return 0.0 if self.target_yaw is None else wrap_angle(self.target_yaw - yaw)
    def clear(self):
        self.target_yaw = None

class DistanceTracker:
    def __init__(self, cfg):
        self.cfg   = cfg
        self._dist = 0.0
        self._last = None
    def update(self, ticks):
        if self._last is None:
            self._last = ticks; return
        self._dist += abs(ticks - self._last) * self.cfg.ENCODER_M_PER_TICK
        self._last  = ticks
    def reset(self):
        self._dist = 0.0; self._last = None
    @property
    def distance(self):
        return self._dist


# ── CONTROLLER ───────────────────────────────────────────────────────────────
class AvoidRerouteController:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.fusion   = SensorFusion(cfg)
        self.heading  = HeadingMemory()
        self.odometer = DistanceTracker(cfg)
        self.state        = State.CRUISING
        self.avoid_dir    = AvoidDir.NONE
        self.clear_cycles = 0
        self.state_ts     = time.time()
        self._estop_ts    = 0.0
        self._yaw         = 0.0
        self._ps          = 0.0
        self._pt          = 0.0

    def step(self, angles, dists, depth_mm, imu_yaw, enc_ticks):
        self.odometer.update(enc_ticks)
        f = self.fusion.fuse(angles, dists, depth_mm)
        self._transitions(f, imu_yaw)
        th, st = self._control(f, imu_yaw)
        st = self.cfg.STEER_ALPHA * st + (1 - self.cfg.STEER_ALPHA) * self._ps
        th = 0.4 * th + 0.6 * self._pt
        self._ps, self._pt = st, th
        return th, st, self.state.value, f

    def _transitions(self, f, yaw):
        cfg = self.cfg
        now = time.time()

        if f["is_emergency"]:
            if self.state != State.EMERGENCY_STOP:
                log.warning("EMERGENCY STOP  %.2fm", f["centre_min"])
                self._estop_ts = now
            self.state = State.EMERGENCY_STOP
            self.clear_cycles = 0
            return

        if self.state == State.EMERGENCY_STOP:
            if (now - self._estop_ts) >= cfg.ESTOP_HOLD_S and f["centre_min"] > cfg.D_RESUME:
                log.info("E-stop released -> AVOIDING")
                self._go(State.AVOIDING); self.odometer.reset()
            return

        if self.state == State.CRUISING:
            if f["is_obstacle"]:
                log.info("Obstacle detected  fused=%.2f", f["fused"])
                self._go(State.OBSTACLE)
            return

        if self.state == State.OBSTACLE:
            self.heading.save(yaw)
            self.avoid_dir = f["best_dir"]
            self.odometer.reset()
            log.info("Avoiding: %s", self.avoid_dir.name)
            self._go(State.AVOIDING)
            return

        if self.state == State.AVOIDING:
            self.clear_cycles = (self.clear_cycles + 1) if not f["is_obstacle"] else 0
            if (self.clear_cycles >= cfg.CLEAR_CYCLES_NEEDED and
                    self.odometer.distance >= cfg.PAST_OBSTACLE_DIST_M):
                log.info("Path clear  %.2fm past  err=%.1f deg -> REROUTING",
                         self.odometer.distance, math.degrees(self.heading.error(yaw)))
                self.clear_cycles = 0; self.odometer.reset()
                self._go(State.REROUTING)
            return

        if self.state == State.REROUTING:
            if f["is_obstacle"]:
                log.warning("New obstacle in re-route -> AVOIDING")
                self.heading.save(yaw); self.avoid_dir = f["best_dir"]
                self.odometer.reset(); self._go(State.AVOIDING); return
            err = abs(self.heading.error(yaw))
            if err < cfg.YAW_TOLERANCE_RAD:
                log.info("Heading restored (%.1f deg) -> CRUISING", math.degrees(err))
                self.avoid_dir = AvoidDir.NONE; self.heading.clear()
                self._go(State.CRUISING)
            elif (now - self.state_ts) > cfg.REROUTE_TIMEOUT_S:
                log.warning("Re-route timeout -> CRUISING")
                self.avoid_dir = AvoidDir.NONE; self.heading.clear()
                self._go(State.CRUISING)
            return

    def _control(self, f, yaw):
        cfg = self.cfg
        if self.state == State.EMERGENCY_STOP: return 0.0, 0.0
        if self.state == State.CRUISING:       return cfg.THROTTLE_CRUISE, 0.0
        if self.state == State.OBSTACLE:       return cfg.THROTTLE_AVOID,  0.0
        if self.state == State.AVOIDING:
            base = self.avoid_dir.value * cfg.MAX_STEER * cfg.AVOID_STEER_FRAC
            s    = f["sectors"]
            if self.avoid_dir == AvoidDir.LEFT  and s["left"]  < 0.8: base *= 0.4
            if self.avoid_dir == AvoidDir.RIGHT and s["right"] < 0.8: base *= 0.4
            th = cfg.THROTTLE_AVOID if s["centre"] < cfg.D_CLEAR else cfg.THROTTLE_CRUISE * 0.85
            return th, float(np.clip(base, -cfg.MAX_STEER, cfg.MAX_STEER))
        if self.state == State.REROUTING:
            st = float(np.clip(cfg.YAW_P_GAIN * self.heading.error(yaw),
                               -cfg.MAX_STEER, cfg.MAX_STEER))
            if abs(st) < cfg.STEER_DEADZONE: st = 0.0
            return cfg.THROTTLE_REROUTE, st
        return cfg.THROTTLE_MIN, 0.0

    def _go(self, s):
        self.state = s; self.state_ts = time.time()


# ── ROS2 NODE ────────────────────────────────────────────────────────────────
class QCar2Node(Node):
    def __init__(self, cfg, ctrl):
        super().__init__("qcar2_avoid_reroute")
        self.cfg  = cfg
        self.ctrl = ctrl
        self._lidar_angles = None
        self._lidar_dists  = None
        self._depth_mm     = None
        self._lock         = threading.Lock()
        self._enc_accum    = 0

        self.create_subscription(LaserScan, cfg.LIDAR_TOPIC, self._cb_lidar, 10)
        self.create_subscription(Image,     cfg.DEPTH_TOPIC, self._cb_depth, 10)

        if HAL_AVAILABLE:
            self._qcar = QCar(readMode=1, canBUS=1)
            log.info("HAL connected")
        else:
            self._qcar = None
            log.warning("HAL not found - no actuator output")

        self.create_timer(cfg.DT, self._loop)
        log.info("Node ready - waiting for sensor data...")

    def _cb_lidar(self, msg: LaserScan):
        n      = len(msg.ranges)
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(n)],
                          dtype=np.float32)
        dists  = np.array(msg.ranges, dtype=np.float32)
        dists  = np.where(np.isfinite(dists), dists, self.cfg.MAX_LIDAR_DIST)
        with self._lock:
            self._lidar_angles = angles
            self._lidar_dists  = dists

    def _cb_depth(self, msg: Image):
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        with self._lock:
            self._depth_mm = raw.reshape((msg.height, msg.width))

    def _loop(self):
        cfg = self.cfg
        if self._qcar is not None:
            self._qcar.read()
            self.ctrl._yaw  = wrap_angle(
                self.ctrl._yaw + float(self._qcar.gyroscope[2]) * cfg.DT)
            imu_yaw          = self.ctrl._yaw
            self._enc_accum += int(abs(self._qcar.motorTach[0]) * 1000)
        else:
            imu_yaw = 0.0
            self._enc_accum += 10

        with self._lock:
            angles = self._lidar_angles
            dists  = self._lidar_dists
            depth  = self._depth_mm

        if angles is None or depth is None:
            missing = []
            if angles is None: missing.append("LiDAR")
            if depth  is None: missing.append("Depth")
            log.info("Waiting for: %s", ", ".join(missing))
            return

        throttle, steering, state, f = self.ctrl.step(
            angles, dists, depth, imu_yaw, self._enc_accum)

        if self._qcar is not None:
            self._qcar.write(motor=throttle, steering=steering,
                             LEDs=np.array([0,0,0,0,0,0,1,1]))

        s = f["sectors"]
        log.info(
            "[%-14s] th=%.3f st=%+.3f | fused=%.2f cmin=%.2fm | "
            "L=%.1f C=%.1f R=%.1f | yaw=%+.1f err=%+.1f dist=%.2fm",
            state, throttle, steering,
            f["fused"], f["centre_min"],
            s["left"], s["centre"], s["right"],
            math.degrees(imu_yaw),
            math.degrees(self.ctrl.heading.error(imu_yaw)),
            self.ctrl.odometer.distance,
        )

    def safe_stop(self):
        if self._qcar is not None:
            self._qcar.write(motor=0.0, steering=0.0, LEDs=np.zeros(8))
            log.info("Actuators zeroed - car stopped")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    cfg      = Config()
    ctrl     = AvoidRerouteController(cfg)
    launcher = SensorLauncher(cfg)

    log.info("=" * 58)
    log.info("  QCar2 Obstacle Avoidance + Re-route  (single command)")
    log.info("  D_STOP=%.2fm  D_OBSTACLE=%.2fm  D_CLEAR=%.2fm",
             cfg.D_STOP, cfg.D_OBSTACLE, cfg.D_CLEAR)
    log.info("  THROTTLE=%.3f  MAX_STEER=%.2f  YAW_GAIN=%.1f",
             cfg.THROTTLE_CRUISE, cfg.MAX_STEER, cfg.YAW_P_GAIN)
    log.info("  Ctrl+C to stop everything safely")
    log.info("=" * 58)

    launcher.start()
    rclpy.init(args=sys.argv)
    node = QCar2Node(cfg, ctrl)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        log.info("Ctrl+C - shutting down...")
    finally:
        node.safe_stop()
        node.destroy_node()
        rclpy.shutdown()
        launcher.stop()
        log.info("Done")

if __name__ == "__main__":
    main()
