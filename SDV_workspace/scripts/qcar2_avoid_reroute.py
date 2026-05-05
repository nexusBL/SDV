"""
qcar2_avoid_reroute.py
======================
QCar2 SDV — Obstacle Avoidance + Deterministic Re-route to Original Path
Hardware : NVIDIA Jetson Orin | Ubuntu 20.04 | ROS2 Humble | Python 3.8
 
Sensors used
------------
  LiDAR          : 720-point scan          → obstacle detection (PRIMARY, 80%)
  RealSense D435 : depth uint16 mm         → obstacle confirmation (SECONDARY, 20%)
  IMU            : yaw (radians)           → heading memory + re-route controller
  Encoders       : ticks → metres          → distance tracking during avoidance
  CSI Camera     : BGR 820×616             → NOT used here (lane-only, future)
 
Key design decisions
--------------------
1. Sensor fusion  : LiDAR dominates. RealSense only adds confidence if its
                    valid-pixel-ratio ≥ 15 % (filters reflective-mat false positives).
2. Heading memory : The moment an obstacle is confirmed, the current IMU yaw is
                    saved as `target_yaw`. After avoidance the car steers back to
                    exactly that yaw using a P-controller.
3. Re-route arc   : Avoidance direction is locked at detection time and does NOT
                    flip mid-manoeuvre. After clearing the obstacle the car
                    counter-steers back to target_yaw smoothly.
4. State machine  : CRUISING → OBSTACLE → AVOIDING → REROUTING → CRUISING
                    (EMERGENCY_STOP can interrupt any state)
 
Headless SSH fixes applied
--------------------------
  DISPLAY / XAUTHORITY  : unset
  QT_QPA_PLATFORM       : offscreen
"""
 
import os, sys, math, time, enum, logging
import numpy as np
 
# ── Headless SSH fixes ───────────────────────────────────────────────────────
os.environ.pop("DISPLAY",     None)
os.environ.pop("XAUTHORITY",  None)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QCar2")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
 
    # Loop
    LOOP_HZ              = 30
    DT                   = 1.0 / LOOP_HZ
 
    # ── LiDAR ──────────────────────────────────────────────────────────────
    LIDAR_POINTS         = 720
    FRONT_ARC_DEG        = 60.0        # ±30° cone considered "ahead"
    SECTOR_DEG           = 20.0        # width of Left / Centre / Right sector
    MIN_LIDAR_DIST       = 0.05        # below → noise
    MAX_LIDAR_DIST       = 5.0
 
    # ── Obstacle thresholds ────────────────────────────────────────────────
    D_OBSTACLE           = 1.2         # centre sector mean < this → obstacle
    D_CLEAR              = 1.6         # centre sector mean > this → path clear
    D_STOP               = 0.40        # hard emergency stop distance
    D_RESUME             = 0.55        # hysteresis: release e-stop above this
 
    # ── RealSense false-positive filter ───────────────────────────────────
    # Shiny/reflective surfaces return depth=0 → we reject them.
    RS_MIN_DEPTH_M       = 0.10        # 0.0 → reflective artefact, ignore
    RS_MAX_DEPTH_M       = 4.0
    RS_ROI               = (0.35, 0.65, 0.30, 0.70)   # (top, bot, left, right) fractions
    RS_MIN_VALID_RATIO   = 0.15        # need ≥15 % valid pixels to trust RS
    RS_OBSTACLE_DEPTH_M  = 1.0         # RS confirms obstacle if median < this
 
    # ── Fusion weights ─────────────────────────────────────────────────────
    W_LIDAR              = 0.80
    W_RS                 = 0.20
    FUSION_THRESH        = 0.60        # fused score ≥ this → obstacle confirmed
 
    # ── Throttle ───────────────────────────────────────────────────────────
    THROTTLE_CRUISE      = 0.12
    THROTTLE_AVOID       = 0.08
    THROTTLE_REROUTE     = 0.09
    THROTTLE_MIN         = 0.05
 
    # ── Steering ───────────────────────────────────────────────────────────
    MAX_STEER            = 0.50
    AVOID_STEER_GAIN     = 1.4        # repulsive field P-gain during avoidance
    YAW_P_GAIN           = 1.8        # heading error → steer during re-route
    STEER_DEADZONE       = 0.02
    STEER_ALPHA          = 0.35       # low-pass smoothing (lower = smoother)
 
    # ── Avoidance completion ───────────────────────────────────────────────
    CLEAR_CYCLES_NEEDED  = 15         # consecutive clear LiDAR ticks to exit AVOIDING
    # How far (metres) the car must travel past the obstacle before re-routing.
    # Prevents turning back into the obstacle immediately.
    PAST_OBSTACLE_DIST_M = 0.40
 
    # ── Re-route completion ────────────────────────────────────────────────
    YAW_TOLERANCE_RAD    = 0.08       # |yaw_error| < this → heading restored
    REROUTE_TIMEOUT_S    = 5.0        # fallback: give up and cruise after this
 
    # ── IMU / Encoder ──────────────────────────────────────────────────────
    ENCODER_M_PER_TICK   = 0.0003     # calibrate for your QCar2 wheel encoder
    IMU_YAW_WRAP         = True       # wrap yaw to [-π, π]
 
    # ── Emergency stop ─────────────────────────────────────────────────────
    ESTOP_HOLD_S         = 1.5        # minimum time to hold e-stop before release
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 2.  STATES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
class State(enum.Enum):
    CRUISING        = "CRUISING"
    OBSTACLE        = "OBSTACLE"       # 1-tick: lock direction, save heading
    AVOIDING        = "AVOIDING"
    REROUTING       = "REROUTING"
    EMERGENCY_STOP  = "EMERGENCY_STOP"
 
 
class AvoidDir(enum.Enum):
    NONE  =  0
    LEFT  =  1    # positive steer
    RIGHT = -1    # negative steer
 
 
def wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SENSOR FUSION
# ═══════════════════════════════════════════════════════════════════════════════
class SensorFusion:
    """
    Fuses LiDAR + RealSense into a single obstacle score [0, 1].
 
    Rule
    ----
    fused = lidar_score * W_LIDAR  +  rs_score * W_RS * rs_confidence
 
    LiDAR score is 1.0 when centre mean < D_OBSTACLE, 0.0 when > D_CLEAR.
    RS score is only counted when enough valid (non-zero) pixels exist in the ROI.
    RS can NEVER trigger avoidance by itself — it only adds weight to LiDAR.
    """
 
    def __init__(self, cfg: Config):
        self.cfg = cfg
 
    # ── LiDAR ─────────────────────────────────────────────────────────────
    def process_lidar(self, raw_angles: np.ndarray, raw_dists: np.ndarray):
        """
        Returns
        -------
        sectors      : dict  left/centre/right → mean distance (m)
        centre_min   : float  minimum distance in the centre sector
        best_dir     : AvoidDir  side with most free space
        """
        cfg = self.cfg
        angles = (raw_angles + math.pi) % (2 * math.pi) - math.pi
 
        valid  = (raw_dists > cfg.MIN_LIDAR_DIST) & (raw_dists < cfg.MAX_LIDAR_DIST)
        a, d   = angles[valid], raw_dists[valid]
 
        half   = cfg.FRONT_ARC_DEG / 2
        front  = np.abs(np.degrees(a)) <= half
        s      = cfg.SECTOR_DEG / 2
 
        lmask  = front & (np.degrees(a) >  s)
        cmask  = front & (np.abs(np.degrees(a)) <= s)
        rmask  = front & (np.degrees(a) < -s)
 
        def smean(m): return float(np.mean(d[m])) if m.any() else cfg.MAX_LIDAR_DIST
        def smin(m):  return float(np.min(d[m]))  if m.any() else cfg.MAX_LIDAR_DIST
 
        sectors = {"left": smean(lmask), "centre": smean(cmask), "right": smean(rmask)}
        centre_min = smin(cmask)
        best_dir   = AvoidDir.LEFT if sectors["left"] >= sectors["right"] else AvoidDir.RIGHT
        return sectors, centre_min, best_dir
 
    # ── RealSense ─────────────────────────────────────────────────────────
    def process_realsense(self, depth_mm: np.ndarray):
        """
        Returns (obstacle: bool, confidence: float)
 
        False-positive guard
        --------------------
        Depth = 0 means the sensor hit a reflective surface and got no return.
        We count what fraction of ROI pixels have a physically plausible depth.
        If that fraction < RS_MIN_VALID_RATIO we declare the frame UNRELIABLE
        and return obstacle=False with confidence=0.  This stops the shiny
        Quanser mat from ever causing a false stop.
        """
        cfg   = self.cfg
        depth = depth_mm.astype(np.float32) / 1000.0   # mm → metres
        h, w  = depth.shape
        t, b, l, r = cfg.RS_ROI
        roi   = depth[int(h*t):int(h*b), int(w*l):int(w*r)]
 
        valid_mask  = (roi > cfg.RS_MIN_DEPTH_M) & (roi < cfg.RS_MAX_DEPTH_M)
        valid_ratio = valid_mask.sum() / valid_mask.size
 
        if valid_ratio < cfg.RS_MIN_VALID_RATIO:
            # Reflective / unreliable frame — ignore completely
            return False, 0.0
 
        median_d   = float(np.median(roi[valid_mask]))
        obstacle   = median_d < cfg.RS_OBSTACLE_DEPTH_M
        confidence = min(valid_ratio / 0.50, 1.0)   # saturates at 50 % valid
        return obstacle, confidence
 
    # ── Fuse ──────────────────────────────────────────────────────────────
    def fuse(self, lidar_angles, lidar_dists, depth_mm):
        cfg = self.cfg
 
        sectors, centre_min, best_dir = self.process_lidar(lidar_angles, lidar_dists)
        rs_obs, rs_conf               = self.process_realsense(depth_mm)
 
        cm = sectors["centre"]
        if   cm < cfg.D_OBSTACLE: lidar_score = 1.0
        elif cm > cfg.D_CLEAR:    lidar_score = 0.0
        else:
            lidar_score = 1.0 - (cm - cfg.D_OBSTACLE) / (cfg.D_CLEAR - cfg.D_OBSTACLE)
 
        rs_score = float(rs_obs) * rs_conf
        fused    = lidar_score * cfg.W_LIDAR + rs_score * cfg.W_RS
 
        return {
            "is_obstacle"  : fused >= cfg.FUSION_THRESH,
            "is_emergency" : centre_min < cfg.D_STOP,
            "fused"        : fused,
            "lidar_score"  : lidar_score,
            "rs_score"     : rs_score,
            "rs_conf"      : rs_conf,
            "sectors"      : sectors,
            "centre_min"   : centre_min,
            "best_dir"     : best_dir,
        }
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 4.  HEADING MEMORY  (IMU-based)
# ═══════════════════════════════════════════════════════════════════════════════
class HeadingMemory:
    """
    Saves the car's yaw the instant an obstacle is detected.
    During REROUTING the car uses a P-controller to return to that yaw.
 
    Why this solves the "continues in wrong direction" bug
    -------------------------------------------------------
    The old reactive code had no memory of where it was going.
    After steering right to avoid, it just kept going right forever.
    Now we save target_yaw = imu_yaw at detection time.
    After clearing the obstacle, yaw_error = wrap(current_yaw - target_yaw).
    We steer proportionally to yaw_error until |yaw_error| < YAW_TOLERANCE.
    The car returns to its original heading in a smooth arc.
    """
 
    def __init__(self):
        self.target_yaw  = None
        self.saved_at    = None
 
    def save(self, yaw: float):
        self.target_yaw = yaw
        self.saved_at   = time.time()
        log.info("🧭 Heading saved: %.3f rad (%.1f°)", yaw, math.degrees(yaw))
 
    def error(self, current_yaw: float) -> float:
        """Returns signed yaw error in [-π, π]. Positive = need to steer left."""
        if self.target_yaw is None:
            return 0.0
        return wrap_angle(self.target_yaw - current_yaw)
 
    def clear(self):
        self.target_yaw = None
        self.saved_at   = None
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DISTANCE TRACKER  (encoder-based)
# ═══════════════════════════════════════════════════════════════════════════════
class DistanceTracker:
    """
    Accumulates distance travelled since last reset using wheel encoder ticks.
    Used to ensure the car has physically moved past the obstacle before
    initiating the re-route arc, preventing U-turns into the obstacle.
    """
 
    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self._dist        = 0.0
        self._last_ticks  = None
 
    def update(self, encoder_ticks: int):
        if self._last_ticks is None:
            self._last_ticks = encoder_ticks
            return
        delta             = abs(encoder_ticks - self._last_ticks)
        self._dist       += delta * self.cfg.ENCODER_M_PER_TICK
        self._last_ticks  = encoder_ticks
 
    def reset(self):
        self._dist       = 0.0
        self._last_ticks  = None
 
    @property
    def distance(self) -> float:
        return self._dist
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN CONTROLLER — STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════
class AvoidRerouteController:
    """
    4-state machine:
 
    ┌──────────┐  fused≥0.6     ┌──────────┐  1 tick    ┌──────────┐
    │CRUISING  │───────────────►│OBSTACLE  │───────────►│AVOIDING  │
    └────▲─────┘                └──────────┘            └────┬─────┘
         │                                                    │
         │  |yaw_err|<tol OR timeout                         │ 15 clear cycles
         │                                                    │ + past obstacle
    ┌────┴─────┐                                        ┌────▼─────┐
    │REROUTING │◄───────────────────────────────────────│          │
    └──────────┘                                        └──────────┘
 
    EMERGENCY_STOP can interrupt any state (centre_min < D_STOP).
    """
 
    def __init__(self, cfg: Config = None):
        self.cfg      = cfg or Config()
        self.fusion   = SensorFusion(self.cfg)
        self.heading  = HeadingMemory()
        self.odometer = DistanceTracker(self.cfg)
 
        self.state        : State    = State.CRUISING
        self.avoid_dir    : AvoidDir = AvoidDir.NONE
        self.clear_cycles : int      = 0
        self.state_ts     : float    = time.time()
        self._estop_ts    : float    = 0.0
 
        self._prev_steer    = 0.0
        self._prev_throttle = 0.0
 
    # ── Public step ───────────────────────────────────────────────────────
    def step(
        self,
        lidar_angles   : np.ndarray,
        lidar_dists    : np.ndarray,
        depth_mm       : np.ndarray,
        imu_yaw        : float,
        encoder_ticks  : int,
    ):
        """
        Call once per control loop tick (30 Hz).
 
        Parameters
        ----------
        lidar_angles   : np.ndarray  shape (720,)  radians
        lidar_dists    : np.ndarray  shape (720,)  metres
        depth_mm       : np.ndarray  shape (H, W)  uint16 millimetres
        imu_yaw        : float       current yaw in radians  (from IMU)
        encoder_ticks  : int         cumulative wheel encoder ticks
 
        Returns
        -------
        throttle : float  [0, 1]
        steering : float  [-MAX_STEER, +MAX_STEER]   + = left
        state    : str
        diag     : dict   full diagnostics
        """
        cfg = self.cfg
 
        # ── Sensor reads ──────────────────────────────────────────────────
        self.odometer.update(encoder_ticks)
        f = self.fusion.fuse(lidar_angles, lidar_dists, depth_mm)
 
        # ── State transitions ─────────────────────────────────────────────
        self._transition_logic(f, imu_yaw)
 
        # ── Control output ────────────────────────────────────────────────
        throttle, steer = self._control(f, imu_yaw)
 
        # ── Smooth outputs ────────────────────────────────────────────────
        a = cfg.STEER_ALPHA
        steer    = a * steer    + (1 - a) * self._prev_steer
        throttle = 0.4 * throttle + 0.6 * self._prev_throttle
        self._prev_steer    = steer
        self._prev_throttle = throttle
 
        diag = {
            "state"        : self.state.value,
            "fused"        : round(f["fused"],       3),
            "lidar_score"  : round(f["lidar_score"],  3),
            "rs_score"     : round(f["rs_score"],     3),
            "rs_conf"      : round(f["rs_conf"],      3),
            "centre_min"   : round(f["centre_min"],   3),
            "sectors"      : {k: round(v,2) for k,v in f["sectors"].items()},
            "avoid_dir"    : self.avoid_dir.name,
            "clear_cycles" : self.clear_cycles,
            "imu_yaw_deg"  : round(math.degrees(imu_yaw), 1),
            "target_yaw_deg": round(math.degrees(self.heading.target_yaw), 1)
                              if self.heading.target_yaw is not None else None,
            "yaw_error_deg": round(math.degrees(self.heading.error(imu_yaw)), 1),
            "dist_past_obs": round(self.odometer.distance, 3),
            "throttle"     : round(throttle, 4),
            "steering"     : round(steer,    4),
        }
        return throttle, steer, self.state.value, diag
 
    # ── Transition logic ──────────────────────────────────────────────────
    def _transition_logic(self, f: dict, imu_yaw: float):
        cfg = self.cfg
        now = time.time()
 
        # ── Emergency stop: overrides everything ──────────────────────────
        if f["is_emergency"]:
            if self.state != State.EMERGENCY_STOP:
                log.warning("⛔  EMERGENCY STOP  — %.2f m", f["centre_min"])
                self._estop_ts = now
            self.state        = State.EMERGENCY_STOP
            self.clear_cycles = 0
            return
 
        # ── Release emergency stop ─────────────────────────────────────────
        if self.state == State.EMERGENCY_STOP:
            held_long_enough = (now - self._estop_ts) >= cfg.ESTOP_HOLD_S
            path_clear       = f["centre_min"] > cfg.D_RESUME
            if held_long_enough and path_clear:
                log.info("✅  E-stop released — resuming AVOIDING")
                self._set_state(State.AVOIDING)
                self.odometer.reset()
            return
 
        # ── CRUISING ──────────────────────────────────────────────────────
        if self.state == State.CRUISING:
            if f["is_obstacle"]:
                log.info("🚧  Obstacle! fused=%.2f  →  OBSTACLE", f["fused"])
                self._set_state(State.OBSTACLE)
            return
 
        # ── OBSTACLE (1 tick: save heading, lock direction) ────────────────
        if self.state == State.OBSTACLE:
            # Save heading NOW — before we start steering away
            self.heading.save(imu_yaw)
            self.avoid_dir    = f["best_dir"]
            self.odometer.reset()
            log.info("↪   Avoid direction: %s", self.avoid_dir.name)
            self._set_state(State.AVOIDING)
            return
 
        # ── AVOIDING ──────────────────────────────────────────────────────
        if self.state == State.AVOIDING:
            if not f["is_obstacle"]:
                self.clear_cycles += 1
            else:
                self.clear_cycles = 0     # obstacle still there — reset
 
            past_obstacle = self.odometer.distance >= cfg.PAST_OBSTACLE_DIST_M
            clear_enough  = self.clear_cycles >= cfg.CLEAR_CYCLES_NEEDED
 
            if clear_enough and past_obstacle:
                log.info(
                    "🛣   Obstacle cleared — %.2f m past — heading error %.1f° → REROUTING",
                    self.odometer.distance,
                    math.degrees(self.heading.error(imu_yaw)),
                )
                self.clear_cycles = 0
                self.odometer.reset()
                self._set_state(State.REROUTING)
            return
 
        # ── REROUTING ─────────────────────────────────────────────────────
        if self.state == State.REROUTING:
            yaw_err = abs(self.heading.error(imu_yaw))
            elapsed = now - self.state_ts
 
            # New obstacle while re-routing → go back to AVOIDING
            if f["is_obstacle"]:
                log.warning("⚠   New obstacle during re-route → AVOIDING")
                self.heading.save(imu_yaw)          # update heading to current
                self.avoid_dir = f["best_dir"]
                self.odometer.reset()
                self._set_state(State.AVOIDING)
                return
 
            heading_restored = yaw_err < cfg.YAW_TOLERANCE_RAD
            timed_out        = elapsed > cfg.REROUTE_TIMEOUT_S
 
            if heading_restored:
                log.info("✔   Heading restored (err=%.1f°) → CRUISING", math.degrees(yaw_err))
                self.avoid_dir = AvoidDir.NONE
                self.heading.clear()
                self._set_state(State.CRUISING)
            elif timed_out:
                log.warning("⏱   Re-route timeout → CRUISING (fallback)")
                self.avoid_dir = AvoidDir.NONE
                self.heading.clear()
                self._set_state(State.CRUISING)
            return
 
    # ── Control output per state ──────────────────────────────────────────
    def _control(self, f: dict, imu_yaw: float):
        cfg = self.cfg
 
        # ── EMERGENCY_STOP ────────────────────────────────────────────────
        if self.state == State.EMERGENCY_STOP:
            return 0.0, 0.0
 
        # ── CRUISING ──────────────────────────────────────────────────────
        if self.state == State.CRUISING:
            # Straight ahead — no lane camera here, just cruise forward
            return cfg.THROTTLE_CRUISE, 0.0
 
        # ── OBSTACLE (transition tick) ─────────────────────────────────────
        if self.state == State.OBSTACLE:
            return cfg.THROTTLE_AVOID, 0.0
 
        # ── AVOIDING ──────────────────────────────────────────────────────
        if self.state == State.AVOIDING:
            sectors = f["sectors"]
 
            # Repulsive field: weighted force from blocked sector
            # The avoid_dir is LOCKED — it never flips mid-manoeuvre.
            base = self.avoid_dir.value * cfg.MAX_STEER * 0.65
 
            # Dynamic correction: if our escape side is now also closing,
            # reduce steer so we don't drive into a wall
            if self.avoid_dir == AvoidDir.LEFT  and sectors["left"]  < 0.8:
                base *= 0.4
            elif self.avoid_dir == AvoidDir.RIGHT and sectors["right"] < 0.8:
                base *= 0.4
 
            # Slow down if centre is still blocked
            throttle = (cfg.THROTTLE_AVOID
                        if sectors["centre"] < cfg.D_CLEAR
                        else cfg.THROTTLE_CRUISE * 0.85)
 
            steer = float(np.clip(base, -cfg.MAX_STEER, cfg.MAX_STEER))
            return throttle, steer
 
        # ── REROUTING ─────────────────────────────────────────────────────
        # Proportional controller on yaw error.
        # yaw_error > 0 → need to turn LEFT  (positive steer)
        # yaw_error < 0 → need to turn RIGHT (negative steer)
        if self.state == State.REROUTING:
            yaw_err = self.heading.error(imu_yaw)
            steer   = float(np.clip(
                cfg.YAW_P_GAIN * yaw_err,
                -cfg.MAX_STEER,
                cfg.MAX_STEER,
            ))
            if abs(steer) < cfg.STEER_DEADZONE:
                steer = 0.0
            return cfg.THROTTLE_REROUTE, steer
 
        return cfg.THROTTLE_MIN, 0.0
 
    def _set_state(self, s: State):
        log.debug("%s → %s", self.state.value, s.value)
        self.state    = s
        self.state_ts = time.time()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MOCK DATA GENERATORS  (unit tests only)
# ═══════════════════════════════════════════════════════════════════════════════
def _lidar_clear(n=720):
    a = np.linspace(-math.pi, math.pi, n)
    return a, np.full(n, 3.0)
 
def _lidar_obstacle_centre(n=720, d=0.9):
    a    = np.linspace(-math.pi, math.pi, n)
    dist = np.full(n, 3.0)
    dist[np.abs(np.degrees(a)) < 15] = d
    return a, dist
 
def _lidar_obstacle_left(n=720, d=0.9):
    a    = np.linspace(-math.pi, math.pi, n)
    dist = np.full(n, 3.0)
    mask = (np.degrees(a) > 5) & (np.degrees(a) < 30)
    dist[mask] = d
    return a, dist
 
def _depth_clean(h=720, w=1280, m=2.5):
    return np.full((h, w), int(m * 1000), dtype=np.uint16)
 
def _depth_reflective(h=720, w=1280):
    return np.zeros((h, w), dtype=np.uint16)
 
def _depth_obstacle(h=720, w=1280, m=0.7):
    f = np.full((h, w), 2500, dtype=np.uint16)
    r0,r1,c0,c1 = int(h*.35),int(h*.65),int(w*.30),int(w*.70)
    f[r0:r1, c0:c1] = int(m * 1000)
    return f
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 8.  UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════
SEP = "─" * 64
 
def run_tests():
    cfg     = Config()
    results = []
 
    # ── TEST 1: Clear road → CRUISING, no obstacle ─────────────────────────
    print(f"\n{SEP}")
    print("TEST 1 — Clear road, clean sensors")
    c = AvoidRerouteController(cfg)
    th, st, state, d = c.step(*_lidar_clear(), _depth_clean(), 0.0, 0)
    ok = state == "CRUISING" and d["fused"] < cfg.FUSION_THRESH
    print(f"  State : {state}  fused={d['fused']}  th={th:.3f}  st={st:.3f}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    results.append(("Clear road", ok))
 
    # ── TEST 2: Reflective mat (depth=0) + LiDAR clear → NO false positive ──
    print(f"\n{SEP}")
    print("TEST 2 — Reflective surface false-positive suppression")
    c = AvoidRerouteController(cfg)
    th, st, state, d = c.step(*_lidar_clear(), _depth_reflective(), 0.0, 0)
    ok = state == "CRUISING" and d["rs_conf"] == 0.0
    print(f"  State : {state}  rs_conf={d['rs_conf']}  fused={d['fused']}")
    print(f"  {'✅ PASS — shiny floor ignored' if ok else '❌ FAIL'}")
    results.append(("Reflective false-positive", ok))
 
    # ── TEST 3: Real obstacle — LiDAR + RS both see it ─────────────────────
    print(f"\n{SEP}")
    print("TEST 3 — Real obstacle (LiDAR + RealSense agree)")
    c = AvoidRerouteController(cfg)
    la, ld = _lidar_obstacle_centre(d=0.8)
    th, st, state, d = c.step(la, ld, _depth_obstacle(m=0.7), 0.0, 0)
    ok = d["fused"] >= cfg.FUSION_THRESH
    print(f"  State : {state}  fused={d['fused']}  lidar={d['lidar_score']}  rs={d['rs_score']}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    results.append(("Real obstacle detected", ok))
 
    # ── TEST 4: LiDAR-only (RS sees nothing) → still triggers ──────────────
    print(f"\n{SEP}")
    print("TEST 4 — LiDAR-only trigger (RS not needed)")
    c = AvoidRerouteController(cfg)
    la, ld = _lidar_obstacle_centre(d=0.9)
    th, st, state, d = c.step(la, ld, _depth_clean(m=2.5), 0.0, 0)
    ok = d["lidar_score"] >= 0.5
    print(f"  State : {state}  lidar_score={d['lidar_score']}  fused={d['fused']}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    results.append(("LiDAR-only trigger", ok))
 
    # ── TEST 5: Emergency stop ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TEST 5 — Emergency stop (< 0.40 m)")
    c = AvoidRerouteController(cfg)
    la, ld = _lidar_obstacle_centre(d=0.3)
    th, st, state, d = c.step(la, ld, _depth_obstacle(m=0.3), 0.0, 0)
    ok = th == 0.0 and st == 0.0 and state == "EMERGENCY_STOP"
    print(f"  State : {state}  th={th}  st={st}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    results.append(("Emergency stop", ok))
 
    # ── TEST 6: Heading is saved at detection moment ────────────────────────
    print(f"\n{SEP}")
    print("TEST 6 — IMU heading saved at detection moment")
    c   = AvoidRerouteController(cfg)
    yaw = 0.45    # car is driving at 0.45 rad heading
 
    # First tick: clear — no heading saved yet
    c.step(*_lidar_clear(), _depth_clean(), yaw, 0)
    assert c.heading.target_yaw is None, "Heading should not be saved yet"
 
    # Now obstacle appears
    la, ld = _lidar_obstacle_centre(d=0.9)
    c.step(la, ld, _depth_obstacle(m=0.7), yaw, 10)   # obstacle tick
    c.step(la, ld, _depth_obstacle(m=0.7), yaw, 20)   # OBSTACLE state → saves heading
 
    ok = c.heading.target_yaw is not None and abs(c.heading.target_yaw - yaw) < 0.01
    print(f"  Saved heading : {c.heading.target_yaw}  (expected ≈ {yaw})")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    results.append(("Heading memory saved", ok))
 
    # ── TEST 7: Full cycle — CRUISING→OBSTACLE→AVOIDING→REROUTING→CRUISING ─
    print(f"\n{SEP}")
    print("TEST 7 — Full avoidance + re-route cycle")
    c    = AvoidRerouteController(cfg)
    # We fake the IMU drifting as the car steers, then coming back
    base_yaw     = 0.3       # original heading
    current_yaw  = base_yaw
    enc          = 0
    states_seen  = []
    la_cl, ld_cl = _lidar_clear()
    la_ob, ld_ob = _lidar_obstacle_centre(d=0.9)
    la_lf, ld_lf = _lidar_obstacle_left(d=0.9)
 
    for tick in range(120):
        if tick < 3:
            la, ld, dep = la_cl, ld_cl, _depth_clean()
        elif tick < 18:
            # obstacle in front
            la, ld, dep = la_ob, ld_ob, _depth_obstacle(m=0.8)
            current_yaw += 0.015         # car is turning right to avoid
            enc         += 100
        elif tick < 50:
            # obstacle now to the left, centre clearing
            la, ld, dep = la_lf, ld_lf, _depth_clean()
            current_yaw += 0.010
            enc         += 120
        else:
            # fully clear — car steers back toward base_yaw
            la, ld, dep  = la_cl, ld_cl, _depth_clean()
            # Simulate IMU heading returning toward target
            err = wrap_angle(base_yaw - current_yaw)
            current_yaw += 0.03 * err    # proportional return
            enc         += 80
 
        th, st, s, d = c.step(la, ld, dep, current_yaw, enc)
        if not states_seen or states_seen[-1] != s:
            states_seen.append(s)
            print(f"  tick={tick:3d}  {s:20s}  yaw={math.degrees(current_yaw):6.1f}°  "
                  f"err={d['yaw_error_deg']:5.1f}°  th={th:.3f}  st={st:.3f}")
 
    visited = set(states_seen)
    ok = {"CRUISING","OBSTACLE","AVOIDING","REROUTING"}.issubset(visited)
    print(f"\n  States visited : {states_seen}")
    print(f"  Final heading  : {math.degrees(current_yaw):.1f}° "
          f"(started {math.degrees(base_yaw):.1f}°)")
    print(f"  {'✅ PASS — full cycle + re-route confirmed' if ok else '❌ FAIL'}")
    results.append(("Full avoidance + re-route cycle", ok))
 
    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("SUMMARY")
    print(f"{'═'*64}")
    passed = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n  {passed}/{len(results)} tests passed")
    print(f"{'═'*64}\n")
    return passed == len(results)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 9.  ON-CAR DEPLOYMENT SKELETON
#     Replace mock lines with real QCar2 / ROS2 sensor reads
# ═══════════════════════════════════════════════════════════════════════════════
def main_loop():
    cfg  = Config()
    ctrl = AvoidRerouteController(cfg)
    log.info("QCar2 Avoid+Reroute starting at %.0f Hz", cfg.LOOP_HZ)
 
    try:
        while True:
            t0 = time.time()
 
            # ── REPLACE THESE WITH REAL SENSOR READS ──────────────────────
            # lidar_angles, lidar_dists = qcar.lidar.read()
            # depth_mm                  = realsense.get_depth_frame_mm()
            # imu_yaw                   = qcar.imu.yaw()          # radians
            # encoder_ticks             = qcar.encoder.ticks()
 
            lidar_angles, lidar_dists = _lidar_clear()       # mock
            depth_mm                  = _depth_clean()        # mock
            imu_yaw                   = 0.0                   # mock
            encoder_ticks             = int(time.time()*100)  # mock
 
            throttle, steering, state, diag = ctrl.step(
                lidar_angles, lidar_dists, depth_mm, imu_yaw, encoder_ticks
            )
 
            # ── REPLACE WITH REAL ACTUATOR WRITE ──────────────────────────
            # qcar.write(throttle=throttle, steering=steering)
 
            log.info(
                "[%s] th=%.3f st=%.3f | fused=%.2f | "
                "dist=%.2fm | yaw_err=%.1f°",
                state, throttle, steering,
                diag["fused"], diag["dist_past_obs"], diag["yaw_error_deg"],
            )
 
            time.sleep(max(0.0, cfg.DT - (time.time() - t0)))
 
    except KeyboardInterrupt:
        log.info("Shutdown.")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--run" in sys.argv:
        main_loop()
    else:
        ok = run_tests()
        sys.exit(0 if ok else 1)
