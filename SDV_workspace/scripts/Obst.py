#!/usr/bin/env python3
"""
=============================================================================
 QCAR2 INTELLIGENT OBSTACLE AVOIDANCE SYSTEM v3.0
=============================================================================
 Architecture:
   7-state machine with real LiDAR feedback at every transition

 Fixes from v2.0:
   [1] Dead reckoning: tracks TIME-based turn integral, not steering magnitude
       — avoidance records (steer * dt) separately from realign
       — realign uses same magnitude in opposite direction until zeroed
   [2] Steering ramp rate fixed: 0.08/tick → reaches full steer in ~6 ticks
   [3] Front arc narrowed: 45° → 22° to avoid wall false positives
   [4] Hysteresis gap increased: obstacle=0.65m, clear=1.10m (gap=0.45m)
   [5] Minimum avoidance arc: must turn for ≥0.4s before checking clear
   [6] QCar2 PAL API corrected: QCarLidar() no keyword args
   [7] Emergency state now tries reverse + re-evaluate after 2s
   [8] CHECK_OBSTACLE has 0.3s max — won't get stuck
   [9] Proper QCar2 shutdown sequence on every exit path
   [10] Proportional steering during avoidance based on front distance
        — closer obstacle = harder turn

 Tuning guide (all params in Config):
   OBSTACLE_THRESHOLD   — how close before triggering avoidance
   FRONT_ARC_HALF_DEG   — narrower = less false positives from walls
   AVOIDANCE_STEERING   — max steer during avoidance (0.3–0.5)
   MIN_TURN_TIME        — minimum arc duration before checking clear
   CLEAR_PATH_THRESHOLD — must be well clear before realigning
=============================================================================
"""

import os
import sys
import math
import time
import numpy as np
from enum import Enum, auto
from collections import deque

# ── EGL headless fix for QCar2 ────────────────────────────────────────────────
os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

if not IS_PHYSICAL_QCAR:
    try:
        import qlabs_setup
        qlabs_setup.setup()
    except ImportError:
        print("[WARN] qlabs_setup not found — continuing without sim setup")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — Every tunable parameter in one place
# ══════════════════════════════════════════════════════════════════════════════

class Config:

    # ── Loop timing ───────────────────────────────────────────────────────────
    SAMPLE_RATE         = 30          # Hz
    RUN_TIME            = 300.0       # Max runtime seconds
    LOOP_PERIOD         = 1.0 / 30    # Seconds per tick

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    LIDAR_MIN_VALID     = 0.05        # m — ignore closer readings (noise)
    LIDAR_MAX_VALID     = 6.0         # m — ignore farther readings
    FRONT_SMOOTH_LEN    = 6           # Frames for rolling average on front dist

    # ── Detection arcs (degrees → converted internally) ──────────────────────
    # Narrow front arc prevents wall corners from triggering avoidance
    FRONT_ARC_HALF_DEG  = 22.0        # ±22° front detection cone
    SIDE_START_DEG      = 25.0        # Side scan starts at 25°
    SIDE_END_DEG        = 110.0       # Side scan ends at 110°
    REAR_START_DEG      = 130.0       # Rear zone

    # ── Distance thresholds (meters) ─────────────────────────────────────────
    OBSTACLE_THRESHOLD  = 0.65        # Trigger avoidance at this distance
    EMERGENCY_THRESHOLD = 0.22        # Emergency stop — imminent collision
    CLEAR_PATH_THRESHOLD = 1.10       # Front must be this clear to stop avoiding
                                      # Gap vs OBSTACLE = 0.45m → prevents oscillation
    CLEARANCE_MIN       = 0.80        # Minimum side clearance to choose that side
    SIDE_WALL_THRESHOLD = 0.28        # Reduce steer if wall this close on avoidance side

    # ── Speeds ────────────────────────────────────────────────────────────────
    BASE_THROTTLE       = 0.12        # Normal forward speed
    APPROACH_THROTTLE   = 0.07        # Slow approach when obstacle detected
    AVOIDANCE_THROTTLE  = 0.09        # Speed while turning around obstacle
    REALIGN_THROTTLE    = 0.10        # Speed while counter-steering
    STABILIZE_THROTTLE  = 0.11        # Speed during settling
    REVERSE_THROTTLE    = -0.08       # Reverse speed (emergency)

    # ── Steering ──────────────────────────────────────────────────────────────
    AVOIDANCE_STEERING  = 0.42        # Peak steer during avoidance
    REALIGN_STEERING    = 0.38        # Counter-steer magnitude
    STEER_RAMP_RATE     = 0.08        # Per tick — reaches 0.42 in ~6 ticks (0.2s)
    STEER_RETURN_RATE   = 0.04        # Per tick — smoother return to straight

    # ── Dead reckoning ────────────────────────────────────────────────────────
    # Integral = sum(actual_steering * dt) during avoidance
    # Realign counter-steers until integral < REALIGN_TOLERANCE
    REALIGN_TOLERANCE   = 0.008       # rad·s — considered realigned below this
    MIN_TURN_TIME       = 0.40        # Minimum seconds turning before checking clear
    MAX_AVOID_TIME      = 7.0         # Force realign after this long
    MAX_REALIGN_TIME    = 5.0         # Force stabilize after this long
    REVERSE_TIME        = 1.5         # How long to reverse in emergency

    # ── State timing ─────────────────────────────────────────────────────────
    CHECK_OBSTACLE_MAX  = 0.30        # Max time in CHECK before deciding
    STABILIZE_DURATION  = 0.80        # Time to hold straight after realign
    COOLDOWN_PERIOD     = 1.2         # Min seconds between avoidance maneuvers

    # ── LEDs ──────────────────────────────────────────────────────────────────
    LED_OFF         = np.zeros(8,        dtype=np.float64)
    LED_HEADLIGHTS  = np.array([0,0,0,0,0,0,1,1], dtype=np.float64)
    LED_LEFT_TURN   = np.array([1,0,1,0,0,0,1,1], dtype=np.float64)
    LED_RIGHT_TURN  = np.array([0,1,0,1,0,0,1,1], dtype=np.float64)
    LED_BRAKE       = np.array([0,0,0,0,1,1,1,1], dtype=np.float64)
    LED_HAZARD      = np.array([1,1,1,1,1,1,1,1], dtype=np.float64)
    LED_REVERSE     = np.array([0,0,0,0,1,1,0,0], dtype=np.float64)

    # ── Battery ───────────────────────────────────────────────────────────────
    BATTERY_WARN     = 10.5
    BATTERY_CRITICAL = 9.5


# ══════════════════════════════════════════════════════════════════════════════
#  STATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class State(Enum):
    FORWARD         = auto()  # Normal cruise
    CHECK_OBSTACLE  = auto()  # Slow + confirm obstacle is real
    AVOID_DECISION  = auto()  # Analyze L/R clearance, pick direction
    AVOID_EXECUTE   = auto()  # Turn around obstacle
    REALIGN         = auto()  # Counter-steer to restore heading
    STABILIZE       = auto()  # Settle straight before resuming
    EMERGENCY_STOP  = auto()  # Imminent collision — reverse then re-evaluate


# ══════════════════════════════════════════════════════════════════════════════
#  LIDAR PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class LidarProcessor:
    """
    Converts raw 360° LiDAR scan into actionable zone data.
    Angle convention after normalization:
        0   = directly ahead
        +90 = left
        -90 = right
        ±180 = directly behind
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._front_buf = deque(maxlen=cfg.FRONT_SMOOTH_LEN)

        # Precompute radian boundaries
        self.front_half = math.radians(cfg.FRONT_ARC_HALF_DEG)
        self.side_start = math.radians(cfg.SIDE_START_DEG)
        self.side_end   = math.radians(cfg.SIDE_END_DEG)
        self.rear_start = math.radians(cfg.REAR_START_DEG)

    @staticmethod
    def _norm(angle):
        """Normalize angle to [-π, π]."""
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def process(self, raw_angles, raw_distances):
        """
        Returns dict with all zone distances and flags.
        All distances in meters. All angles in radians (normalized).
        """
        out = {
            'front_dist':    6.0,   # Smoothed front minimum
            'front_raw':     6.0,   # Unsmoothed front minimum
            'front_angle':   0.0,   # Angle of closest front object
            'left_min':      6.0,   # Closest on left
            'left_avg':      6.0,   # Average clearance on left
            'right_min':     6.0,   # Closest on right
            'right_avg':     6.0,   # Average clearance on right
            'rear_min':      6.0,   # Closest behind
            'obstacle':      False, # Front obstacle detected
            'emergency':     False, # Imminent collision
            'front_clear':   True,  # Path ahead is clear for realign
            'all_blocked':   False, # No safe direction
            'obs_side':      'center',  # Which side obstacle leans toward
        }

        if raw_distances is None or len(raw_distances) == 0:
            return out

        angles = np.asarray(raw_angles,    dtype=np.float64)
        dists  = np.asarray(raw_distances, dtype=np.float64)

        # Filter valid readings
        valid = (
            np.isfinite(dists) &
            (dists > self.cfg.LIDAR_MIN_VALID) &
            (dists < self.cfg.LIDAR_MAX_VALID)
        )
        if not np.any(valid):
            return out

        angles = np.array([self._norm(a) for a in angles[valid]])
        dists  = dists[valid]

        # ── Front arc ────────────────────────────────────────────────────────
        f_mask = np.abs(angles) < self.front_half
        if np.any(f_mask):
            f_dists = dists[f_mask]
            f_angs  = angles[f_mask]
            idx     = np.argmin(f_dists)
            raw_min = float(f_dists[idx])
            self._front_buf.append(raw_min)
            smoothed = float(np.mean(self._front_buf))

            out['front_raw']   = raw_min
            out['front_dist']  = smoothed
            out['front_angle'] = float(f_angs[idx])
            out['obstacle']    = smoothed < self.cfg.OBSTACLE_THRESHOLD
            out['emergency']   = smoothed < self.cfg.EMERGENCY_THRESHOLD
            out['front_clear'] = smoothed > self.cfg.CLEAR_PATH_THRESHOLD

            # Classify which side obstacle leans toward
            ang = float(f_angs[idx])
            if ang > math.radians(8):
                out['obs_side'] = 'left'
            elif ang < math.radians(-8):
                out['obs_side'] = 'right'
            else:
                out['obs_side'] = 'center'
        else:
            self._front_buf.append(6.0)

        # ── Left side ────────────────────────────────────────────────────────
        l_mask = (angles > self.side_start) & (angles < self.side_end)
        if np.any(l_mask):
            ld = dists[l_mask]
            out['left_min'] = float(np.min(ld))
            out['left_avg'] = float(np.mean(ld))

        # ── Right side ───────────────────────────────────────────────────────
        r_mask = (angles < -self.side_start) & (angles > -self.side_end)
        if np.any(r_mask):
            rd = dists[r_mask]
            out['right_min'] = float(np.min(rd))
            out['right_avg'] = float(np.mean(rd))

        # ── Rear ─────────────────────────────────────────────────────────────
        rear_mask = np.abs(angles) > self.rear_start
        if np.any(rear_mask):
            out['rear_min'] = float(np.min(dists[rear_mask]))

        # ── All blocked ──────────────────────────────────────────────────────
        out['all_blocked'] = (
            out['front_dist'] < self.cfg.OBSTACLE_THRESHOLD and
            out['left_min']   < self.cfg.CLEARANCE_MIN * 0.6 and
            out['right_min']  < self.cfg.CLEARANCE_MIN * 0.6
        )

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  DEAD RECKONING TRACKER  — Fixed version
# ══════════════════════════════════════════════════════════════════════════════

class TurnTracker:
    """
    Tracks the angular integral (steer * dt) during avoidance.
    Realignment counter-steers until that integral is zeroed.

    Key fix vs v2.0:
    - avoidance_integral accumulates ONLY during AVOID_EXECUTE
    - realign_integral accumulates ONLY during REALIGN (opposite sign)
    - Done when |realign_integral| >= |avoidance_integral|
    - No shared accumulator that confuses direction
    """

    def __init__(self):
        self.avoidance_integral = 0.0   # Accumulated during AVOID_EXECUTE
        self.realign_integral   = 0.0   # Accumulated during REALIGN
        self.direction          = 0     # +1=left, -1=right

    def reset(self):
        self.avoidance_integral = 0.0
        self.realign_integral   = 0.0
        self.direction          = 0

    def record_avoidance(self, steer: float, dt: float):
        """Call every tick during AVOID_EXECUTE."""
        self.avoidance_integral += abs(steer) * dt

    def record_realign(self, steer: float, dt: float):
        """Call every tick during REALIGN."""
        self.realign_integral += abs(steer) * dt

    def realign_needed(self) -> float:
        """
        Remaining realignment needed (positive = need more counter-steering).
        Returns 0.0 when fully realigned.
        """
        remaining = self.avoidance_integral - self.realign_integral
        return max(0.0, remaining)

    def is_realigned(self, tolerance: float) -> bool:
        return self.realign_needed() <= tolerance

    def get_realign_steer(self, cfg: Config) -> float:
        """
        Return proportional counter-steer command.
        Scales down as we approach full realignment to avoid overshoot.
        """
        remaining = self.realign_needed()
        if remaining <= cfg.REALIGN_TOLERANCE:
            return 0.0

        # Proportional: reduce steer in final 30% of realignment
        scale = min(1.0, remaining / (cfg.REALIGN_TOLERANCE * 3))
        steer = cfg.REALIGN_STEERING * scale

        # Opposite direction of avoidance
        return -self.direction * steer


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTROLLER — 7-state machine
# ══════════════════════════════════════════════════════════════════════════════

class AvoidanceController:

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.tracker    = TurnTracker()

        # State machine
        self.state       = State.FORWARD
        self.prev_state  = State.FORWARD
        self._entry_time = time.time()

        # Avoidance
        self.avoid_dir        = 0      # +1=left, -1=right
        self.avoid_start_time = 0.0
        self.last_avoid_end   = -99.0

        # Steering (smoothed output)
        self._steer_actual = 0.0

        # Telemetry
        self.loop_n    = 0
        self.info      = ""
        self._scan     = {}

    # ── State transition ──────────────────────────────────────────────────────

    def _go(self, new_state: State):
        old = self.state
        self.state       = new_state
        self.prev_state  = old
        self._entry_time = time.time()

        icons = {
            State.FORWARD:        "🟢",
            State.CHECK_OBSTACLE: "🔍",
            State.AVOID_DECISION: "📐",
            State.AVOID_EXECUTE:  "🚧",
            State.REALIGN:        "↩️ ",
            State.STABILIZE:      "⏸️ ",
            State.EMERGENCY_STOP: "🚨",
        }
        icon = icons.get(new_state, "  ")
        print(f"  {icon} {old.name:15s} → {new_state.name}")

    def _elapsed(self) -> float:
        return time.time() - self._entry_time

    # ── Steering helper ───────────────────────────────────────────────────────

    def _ramp_steer(self, target: float, rate: float) -> float:
        """Ramp actual steering toward target at given rate per tick."""
        diff = target - self._steer_actual
        step = min(abs(diff), rate)
        self._steer_actual += math.copysign(step, diff)
        return self._steer_actual

    # ── LEDs ─────────────────────────────────────────────────────────────────

    def _leds(self, throttle: float, steer: float) -> np.ndarray:
        if self.state == State.EMERGENCY_STOP:
            return self.cfg.LED_HAZARD.copy()
        if throttle < 0:
            return self.cfg.LED_REVERSE.copy()
        if steer > 0.15:
            return self.cfg.LED_LEFT_TURN.copy()
        if steer < -0.15:
            return self.cfg.LED_RIGHT_TURN.copy()
        if throttle <= 0:
            return self.cfg.LED_BRAKE.copy()
        return self.cfg.LED_HEADLIGHTS.copy()

    # ──────────────────────────────────────────────────────────────────────────
    #  STATE HANDLERS
    # ──────────────────────────────────────────────────────────────────────────

    def _forward(self, s: dict) -> tuple:
        """
        Normal cruise. Smoothly return steering to zero.
        Trigger CHECK_OBSTACLE when obstacle enters front arc.
        """
        self.info = f"CRUISE  front={s['front_dist']:.2f}m  L={s['left_min']:.2f}  R={s['right_min']:.2f}"

        # Emergency
        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        # Ramp steering back to zero naturally
        steer = self._ramp_steer(0.0, self.cfg.STEER_RETURN_RATE)

        # Obstacle check with cooldown
        since_last = time.time() - self.last_avoid_end
        if s['obstacle'] and since_last > self.cfg.COOLDOWN_PERIOD:
            self._go(State.CHECK_OBSTACLE)
            return self.cfg.APPROACH_THROTTLE, steer

        return self.cfg.BASE_THROTTLE, steer

    # ─────────────────────────────────────────────────────────────────────────

    def _check_obstacle(self, s: dict) -> tuple:
        """
        Slow down + confirm obstacle is real (not a noise spike).
        Max time: CHECK_OBSTACLE_MAX seconds.
        """
        elapsed = self._elapsed()
        self.info = f"CONFIRM front={s['front_dist']:.2f}m t={elapsed:.2f}s"

        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        # Obstacle disappeared — false positive
        if not s['obstacle']:
            self._go(State.FORWARD)
            return self.cfg.BASE_THROTTLE, self._steer_actual

        # All blocked
        if s['all_blocked']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        # Confirmed after brief confirmation window
        if elapsed >= 0.12:
            self._go(State.AVOID_DECISION)
            return self.cfg.APPROACH_THROTTLE, self._steer_actual

        # Timeout safety — don't get stuck here
        if elapsed > self.cfg.CHECK_OBSTACLE_MAX:
            self._go(State.AVOID_DECISION)
            return self.cfg.APPROACH_THROTTLE, self._steer_actual

        return self.cfg.APPROACH_THROTTLE, self._steer_actual

    # ─────────────────────────────────────────────────────────────────────────

    def _avoid_decision(self, s: dict) -> tuple:
        """
        Analyze left and right clearance with priority logic:
        1. Emergency → stop
        2. Only one side safe → choose it
        3. Both safe → choose wider side (with 1.4x ratio test)
        4. Both blocked → emergency
        5. Obstacle leaning to one side → go opposite

        Direction: +1 = steer left (positive steering), -1 = steer right
        """
        self.info = (f"DECIDE  obs={s['obs_side']}  "
                     f"L_avg={s['left_avg']:.2f}  R_avg={s['right_avg']:.2f}  "
                     f"L_min={s['left_min']:.2f}  R_min={s['right_min']:.2f}")

        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        if s['all_blocked']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        L_ok = s['left_min']  > self.cfg.CLEARANCE_MIN
        R_ok = s['right_min'] > self.cfg.CLEARANCE_MIN

        if L_ok and not R_ok:
            direction = +1
            reason = f"ONLY LEFT SAFE (L_min={s['left_min']:.2f})"

        elif R_ok and not L_ok:
            direction = -1
            reason = f"ONLY RIGHT SAFE (R_min={s['right_min']:.2f})"

        elif not L_ok and not R_ok:
            self._go(State.EMERGENCY_STOP)
            print(f"    🚨 Both sides blocked! L={s['left_min']:.2f} R={s['right_min']:.2f}")
            return 0.0, 0.0

        else:
            # Both sides safe — pick better one
            ratio = 1.40  # One side must be 40% clearer to prefer it

            if s['left_avg'] > s['right_avg'] * ratio:
                direction = +1
                reason = f"LEFT WIDER (avg L={s['left_avg']:.2f} > R={s['right_avg']:.2f})"

            elif s['right_avg'] > s['left_avg'] * ratio:
                direction = -1
                reason = f"RIGHT WIDER (avg R={s['right_avg']:.2f} > L={s['left_avg']:.2f})"

            else:
                # Similar clearance — use obstacle lean
                if s['obs_side'] == 'left':
                    direction = -1
                    reason = f"OBS LEANS LEFT → GO RIGHT"
                elif s['obs_side'] == 'right':
                    direction = +1
                    reason = f"OBS LEANS RIGHT → GO LEFT"
                else:
                    # Center obstacle — prefer right (safer for most corridors)
                    direction = -1
                    reason = f"CENTER OBS → DEFAULT RIGHT"

        print(f"    📐 {reason}")

        self.avoid_dir        = direction
        self.avoid_start_time = time.time()
        self.tracker.reset()
        self.tracker.direction = direction

        self._go(State.AVOID_EXECUTE)
        return self.cfg.AVOIDANCE_THROTTLE, self._steer_actual

    # ─────────────────────────────────────────────────────────────────────────

    def _avoid_execute(self, s: dict) -> tuple:
        """
        Steer around obstacle using proportional control:
        - Harder steer when obstacle is close (urgent)
        - Softer steer when obstacle is farther (smoother)
        - Reduce steer if wall detected on avoidance side
        - Transition to REALIGN when front arc is clear AND min turn time elapsed
        """
        elapsed     = self._elapsed()
        total_avoid = time.time() - self.avoid_start_time
        dt          = self.cfg.LOOP_PERIOD

        self.info = (f"AVOID {'L' if self.avoid_dir>0 else 'R'}  "
                     f"front={s['front_dist']:.2f}m  "
                     f"integral={self.tracker.avoidance_integral:.3f}  "
                     f"t={elapsed:.1f}s")

        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        # ── Proportional steer magnitude ──────────────────────────────────────
        # Closer to obstacle = harder steer (range: 0.6 to 1.0 scale)
        dist = max(s['front_dist'], 0.15)
        proximity_scale = np.clip(
            1.0 - (dist - self.cfg.EMERGENCY_THRESHOLD) /
                  (self.cfg.OBSTACLE_THRESHOLD - self.cfg.EMERGENCY_THRESHOLD),
            0.60, 1.00
        )
        target_steer = self.avoid_dir * self.cfg.AVOIDANCE_STEERING * proximity_scale

        # ── Wall avoidance on the turning side ───────────────────────────────
        if self.avoid_dir > 0:   # Turning left — watch left wall
            wall_dist = s['left_min']
        else:                    # Turning right — watch right wall
            wall_dist = s['right_min']

        if wall_dist < self.cfg.SIDE_WALL_THRESHOLD:
            # Reduce steer proportionally as we approach wall
            wall_scale = max(0.15, wall_dist / self.cfg.SIDE_WALL_THRESHOLD)
            target_steer *= wall_scale
            self.info += f"  WALL={wall_dist:.2f}→scale={wall_scale:.2f}"

        # ── Ramp steering ────────────────────────────────────────────────────
        steer = self._ramp_steer(target_steer, self.cfg.STEER_RAMP_RATE)

        # ── Record actual steer for dead reckoning ───────────────────────────
        self.tracker.record_avoidance(steer, dt)

        # ── Check if obstacle cleared ─────────────────────────────────────────
        # Must have turned for minimum time AND front must be truly clear
        if elapsed >= self.cfg.MIN_TURN_TIME and s['front_clear']:
            print(f"    ✅ Obstacle cleared at {elapsed:.2f}s  "
                  f"front={s['front_dist']:.2f}m  "
                  f"avoidance integral={self.tracker.avoidance_integral:.3f}")
            self._go(State.REALIGN)
            return self.cfg.REALIGN_THROTTLE, steer

        # ── Timeout → force realign ──────────────────────────────────────────
        if total_avoid > self.cfg.MAX_AVOID_TIME:
            print(f"    ⏰ Avoidance timeout at {total_avoid:.1f}s — forcing REALIGN")
            self._go(State.REALIGN)
            return self.cfg.REALIGN_THROTTLE, steer

        return self.cfg.AVOIDANCE_THROTTLE, steer

    # ─────────────────────────────────────────────────────────────────────────

    def _realign(self, s: dict) -> tuple:
        """
        Counter-steer to restore original heading.
        Uses the exact integral accumulated during AVOID_EXECUTE.
        Proportional: reduces counter-steer as integral nears zero.
        """
        elapsed = self._elapsed()
        dt      = self.cfg.LOOP_PERIOD
        remaining = self.tracker.realign_needed()

        self.info = (f"REALIGN  remaining={remaining:.3f}  "
                     f"steer={self._steer_actual:.2f}  t={elapsed:.1f}s")

        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        # New obstacle during realignment — re-evaluate
        if s['obstacle']:
            print(f"    ⚠️  New obstacle during realign! front={s['front_dist']:.2f}m")
            self._go(State.CHECK_OBSTACLE)
            return self.cfg.APPROACH_THROTTLE, self._steer_actual

        # Get proportional counter-steer
        target_steer = self.tracker.get_realign_steer(self.cfg)
        steer = self._ramp_steer(target_steer, self.cfg.STEER_RAMP_RATE)

        # Record counter-steer
        self.tracker.record_realign(steer, dt)

        # Done?
        if self.tracker.is_realigned(self.cfg.REALIGN_TOLERANCE):
            print(f"    ↩️  Heading restored at {elapsed:.2f}s  "
                  f"remaining={self.tracker.realign_needed():.4f}")
            self._go(State.STABILIZE)
            return self.cfg.STABILIZE_THROTTLE, steer

        # Timeout
        if elapsed > self.cfg.MAX_REALIGN_TIME:
            print(f"    ⏰ Realign timeout — moving to STABILIZE")
            self._go(State.STABILIZE)
            return self.cfg.STABILIZE_THROTTLE, steer

        return self.cfg.REALIGN_THROTTLE, steer

    # ─────────────────────────────────────────────────────────────────────────

    def _stabilize(self, s: dict) -> tuple:
        """
        Hold straight for STABILIZE_DURATION seconds.
        Ramp steering smoothly to zero.
        """
        elapsed = self._elapsed()
        self.info = (f"STABILIZE  steer={self._steer_actual:.3f}  "
                     f"t={elapsed:.1f}/{self.cfg.STABILIZE_DURATION:.1f}s")

        if s['emergency']:
            self._go(State.EMERGENCY_STOP)
            return 0.0, 0.0

        steer = self._ramp_steer(0.0, self.cfg.STEER_RETURN_RATE)

        done = (elapsed > self.cfg.STABILIZE_DURATION and abs(steer) < 0.015)
        if done:
            self.last_avoid_end = time.time()
            self._steer_actual  = 0.0
            self._go(State.FORWARD)

        return self.cfg.STABILIZE_THROTTLE, steer

    # ─────────────────────────────────────────────────────────────────────────

    def _emergency_stop(self, s: dict) -> tuple:
        """
        Emergency stop → brief reverse → re-evaluate.
        If path clears, return to FORWARD.
        If still blocked after reverse, wait.
        """
        elapsed = self._elapsed()
        self.info = (f"🚨 EMERGENCY  front={s['front_dist']:.2f}m  "
                     f"L={s['left_min']:.2f}  R={s['right_min']:.2f}  "
                     f"t={elapsed:.1f}s")

        # Phase 1: Full stop (first 0.4s)
        if elapsed < 0.4:
            return 0.0, 0.0

        # Phase 2: Reverse (0.4s to REVERSE_TIME)
        if elapsed < self.cfg.REVERSE_TIME:
            return self.cfg.REVERSE_THROTTLE, 0.0

        # Phase 3: Re-evaluate after reversing
        if s['front_clear'] and not s['all_blocked']:
            print(f"    🟢 Path cleared after {elapsed:.1f}s — resuming")
            self.last_avoid_end = time.time()
            self._steer_actual  = 0.0
            self._go(State.FORWARD)
            return self.cfg.BASE_THROTTLE, 0.0

        # Still blocked — check if one side opened
        if not s['all_blocked']:
            print(f"    📐 Re-evaluating after emergency reverse...")
            self._go(State.AVOID_DECISION)
            return 0.0, 0.0

        # Truly stuck — wait
        return 0.0, 0.0

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, scan: dict) -> tuple:
        """
        Route scan to correct state handler.
        Returns (throttle, steering, leds) — all clamped.
        """
        self._scan   = scan
        self.loop_n += 1

        handlers = {
            State.FORWARD:        self._forward,
            State.CHECK_OBSTACLE: self._check_obstacle,
            State.AVOID_DECISION: self._avoid_decision,
            State.AVOID_EXECUTE:  self._avoid_execute,
            State.REALIGN:        self._realign,
            State.STABILIZE:      self._stabilize,
            State.EMERGENCY_STOP: self._emergency_stop,
        }

        throttle, steer = handlers[self.state](scan)

        # Safety clamps
        throttle = float(np.clip(throttle, -0.15, 0.25))
        steer    = float(np.clip(steer, -self.cfg.AVOIDANCE_STEERING,
                                         self.cfg.AVOIDANCE_STEERING))

        leds = self._leds(throttle, steer)
        return throttle, steer, leds

    def telemetry(self, battery: float, t: float) -> str:
        return (
            f"t={t:6.1f}s | {self.state.name:15s} | "
            f"{self.info:60s} | "
            f"🔋{battery:.1f}V"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg          = Config()
    lidar_proc   = LidarProcessor(cfg)
    controller   = AvoidanceController(cfg)

    print("=" * 72)
    print("  QCAR2 INTELLIGENT OBSTACLE AVOIDANCE  v3.0")
    print("=" * 72)
    print(f"  Obstacle threshold  : {cfg.OBSTACLE_THRESHOLD}m")
    print(f"  Clear threshold     : {cfg.CLEAR_PATH_THRESHOLD}m")
    print(f"  Emergency threshold : {cfg.EMERGENCY_THRESHOLD}m")
    print(f"  Front arc           : ±{cfg.FRONT_ARC_HALF_DEG}°")
    print(f"  Base speed          : {cfg.BASE_THROTTLE}")
    print(f"  Avoidance steering  : {cfg.AVOIDANCE_STEERING}")
    print(f"  Min turn time       : {cfg.MIN_TURN_TIME}s")
    print(f"  Sample rate         : {cfg.SAMPLE_RATE}Hz")
    print(f"  Physical QCar       : {IS_PHYSICAL_QCAR}")
    print("=" * 72)
    print("  Press Ctrl+C to stop safely")
    print("=" * 72)
    print()

    myLidar = None
    t0      = time.time()

    try:
        # ── Init LiDAR ────────────────────────────────────────────────────────
        print("Initializing LiDAR ... ", end="", flush=True)
        myLidar = QCarLidar()          # QCar2 PAL — no keyword args
        time.sleep(1.2)               # Let motor spin up

        # Flush initial bad readings
        for _ in range(5):
            myLidar.read()
            time.sleep(0.1)
        print("✅")

        # ── Init QCar ─────────────────────────────────────────────────────────
        print("Initializing QCar  ... ", end="", flush=True)
        with QCar(readMode=1, frequency=cfg.SAMPLE_RATE) as myCar:
            print("✅")

            myCar.read()
            batt = myCar.batteryVoltage
            print(f"Battery: {batt:.1f}V")

            if batt < cfg.BATTERY_CRITICAL:
                print("❌ Battery critical — aborting!")
                return

            print()
            print("🚀 Autonomous drive active!")
            print("-" * 72)

            t0             = time.time()
            last_telemetry = t0

            # ── Control loop ──────────────────────────────────────────────────
            while True:
                loop_start = time.time()
                elapsed    = loop_start - t0

                if elapsed > cfg.RUN_TIME:
                    print(f"\n⏰ Run time limit ({cfg.RUN_TIME}s) reached")
                    break

                # Read sensors
                myCar.read()
                myLidar.read()

                # Battery safety
                batt = myCar.batteryVoltage
                if batt < cfg.BATTERY_CRITICAL:
                    print(f"\n🔋❌ Battery critical ({batt:.1f}V) — stopping!")
                    myCar.write(0.0, 0.0, cfg.LED_HAZARD)
                    break
                if batt < cfg.BATTERY_WARN:
                    print(f"🔋⚠️  Low battery: {batt:.1f}V")

                # Process LiDAR
                scan = lidar_proc.process(myLidar.angles, myLidar.distances)

                # State machine
                throttle, steer, leds = controller.update(scan)

                # Send to car
                myCar.write(throttle, steer, leds)

                # Telemetry
                if loop_start - last_telemetry >= 0.5:
                    print(controller.telemetry(batt, elapsed))
                    last_telemetry = loop_start

                # Maintain loop rate
                loop_dt = time.time() - loop_start
                slack   = cfg.LOOP_PERIOD - loop_dt
                if slack > 0:
                    time.sleep(slack)

    except KeyboardInterrupt:
        print("\n\n  🛑 Stopped by user")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ── Safe shutdown ─────────────────────────────────────────────────────
        print("\nShutting down...")

        # Stop LiDAR motor properly
        if myLidar is not None:
            try:
                myLidar.terminate()
                print("  ✅ LiDAR terminated")
            except Exception as e:
                print(f"  ⚠️  LiDAR stop error: {e}")

        # Belt-and-suspenders: QCar2 hardware stop
        try:
            from pal.products.qcar import QCarLidar as _QL
            _ql = _QL()
            _ql.terminate()
        except Exception:
            pass

        total = time.time() - t0
        print(f"  Session: {total:.1f}s  |  Loops: {controller.loop_n}  "
              f"|  Avg: {controller.loop_n/max(total,1):.1f} Hz")
        print("  ✅ Shutdown complete\n")


if __name__ == "__main__":
    main()
