#!/usr/bin/env python3
"""
=============================================================================
 INTELLIGENT QCAR2 AUTONOMOUS OBSTACLE AVOIDANCE SYSTEM v2.0
=============================================================================
 Features:
   - 6-state structured state machine
   - Real-time LiDAR-driven avoidance (not purely time-based)
   - Symmetric counter-steering for lane return (dead reckoning)
   - Left/right clearance analysis before choosing avoidance direction
   - Emergency stop when blocked on all sides
   - Cooldown periods between avoidance maneuvers
   - Smooth stabilization to prevent oscillations
   - Modular design for future lane detection / RT-DETR integration
=============================================================================
"""

# ---------------------------------------------------------------------------
# Environment fixes for QCar2 headless operation
# ---------------------------------------------------------------------------
import os
import sys
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import time
import math
from enum import Enum, auto
from collections import deque

from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

# QLabs setup if running in simulation
if not IS_PHYSICAL_QCAR:
    try:
        import qlabs_setup
        qlabs_setup.setup()
    except ImportError:
        print("WARNING: qlabs_setup not found, continuing without simulation setup")


# ===========================================================================
# CONFIGURATION — All tunable parameters in one place
# ===========================================================================
class Config:
    """
    Central configuration class.
    Change values here to tune behavior without modifying logic code.
    Future modules (lane detection, RT-DETR) can read/override these.
    """

    # --- Timing ---
    SAMPLE_RATE         = 30        # Control loop frequency (Hz)
    RUN_TIME            = 300.0     # Maximum run time (seconds)
    LOOP_PERIOD         = 1.0 / SAMPLE_RATE  # Seconds per loop iteration
    TELEMETRY_INTERVAL  = 0.5       # Print telemetry every N seconds

    # --- LiDAR ---
    LIDAR_NUM_MEASUREMENTS  = 1000
    LIDAR_RANGING_MODE      = 2
    LIDAR_INTERPOLATION     = 0
    LIDAR_MIN_VALID         = 0.05  # Minimum valid reading (meters)
    LIDAR_MAX_VALID         = 5.0   # Maximum valid reading (meters)

    # --- Detection Arcs (in degrees, converted to radians internally) ---
    FRONT_ARC_HALF_DEG      = 45.0   # Front detection zone: ±45°
    SIDE_CHECK_START_DEG    = 30.0   # Side clearance check start angle
    SIDE_CHECK_END_DEG      = 120.0  # Side clearance check end angle
    REAR_ARC_START_DEG      = 135.0  # Rear zone start
    REAR_ARC_END_DEG        = 225.0  # Rear zone end

    # --- Obstacle Thresholds (meters) ---
    OBSTACLE_THRESHOLD      = 0.7    # Distance to trigger avoidance
    CLEARANCE_THRESHOLD     = 1.0    # Minimum clearance for choosing direction
    EMERGENCY_THRESHOLD     = 0.25   # Emergency stop distance
    CLEAR_PATH_THRESHOLD    = 0.9    # Distance to confirm path is clear again
    SIDE_DANGER_THRESHOLD   = 0.3    # Too close on side during avoidance

    # --- Motion Control ---
    BASE_THROTTLE           = 0.15   # Normal forward speed
    AVOIDANCE_THROTTLE      = 0.10   # Speed during avoidance maneuvers
    STABILIZE_THROTTLE      = 0.12   # Speed during stabilization
    MAX_STEERING            = 0.5    # Maximum steering magnitude
    AVOIDANCE_STEERING      = 0.45   # Steering during avoidance turn
    REALIGN_STEERING        = 0.40   # Steering during counter-steer realignment
    STABILIZE_STEERING_RATE = 0.02   # How fast steering returns to zero per tick

    # --- Dead Reckoning ---
    # We accumulate (steering * dt) during avoidance and counter-steer
    # the same integral during realignment
    REALIGN_TOLERANCE       = 0.05   # Accumulated steering integral close enough to 0

    # --- Safety ---
    COOLDOWN_PERIOD         = 1.5    # Minimum seconds between avoidance maneuvers
    MAX_AVOIDANCE_DURATION  = 8.0    # Timeout: force return to FORWARD
    STABILIZE_DURATION      = 1.0    # Time to hold straight after realignment
    BATTERY_WARNING_VOLTAGE = 10.5   # Print warning below this voltage
    BATTERY_CRITICAL_VOLTAGE = 9.5   # Emergency stop below this voltage

    # --- LEDs (8-element array for QCar) ---
    LED_OFF         = np.zeros(8)
    LED_LEFT_TURN   = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    LED_RIGHT_TURN  = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float)
    LED_BRAKE       = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=float)
    LED_HAZARD      = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    LED_HEADLIGHTS  = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=float)


# ===========================================================================
# STATE DEFINITIONS
# ===========================================================================
class State(Enum):
    """
    Six-state machine for intelligent obstacle avoidance.
    Designed to be extensible — future states like LANE_FOLLOW,
    INTERSECTION, SIGN_RESPONSE can be added without restructuring.
    """
    FORWARD         = auto()  # Normal straight driving
    CHECK_OBSTACLE  = auto()  # Obstacle detected, analyzing environment
    AVOID_DECISION  = auto()  # Deciding left vs right avoidance
    AVOID_EXECUTE   = auto()  # Actively steering around obstacle
    REALIGN         = auto()  # Counter-steering back to original trajectory
    STABILIZE       = auto()  # Holding straight to let vehicle settle
    EMERGENCY_STOP  = auto()  # Blocked on all sides, full stop


# ===========================================================================
# LIDAR PROCESSOR — Robust 360° analysis
# ===========================================================================
class LidarProcessor:
    """
    Processes raw LiDAR data into actionable zone information.
    Handles the 0/2π wraparound correctly for front-facing detection.
    """

    def __init__(self, config: Config):
        self.cfg = config

        # Precompute arc boundaries in radians
        self.front_half_rad = math.radians(config.FRONT_ARC_HALF_DEG)
        self.side_start_rad = math.radians(config.SIDE_CHECK_START_DEG)
        self.side_end_rad   = math.radians(config.SIDE_CHECK_END_DEG)
        self.rear_start_rad = math.radians(config.REAR_ARC_START_DEG)
        self.rear_end_rad   = math.radians(config.REAR_ARC_END_DEG)

        # Smoothing buffer for front distance (reduces noise spikes)
        self.front_dist_buffer = deque(maxlen=5)

    def _normalize_angle(self, angle_rad):
        """
        Normalize angle to [-π, +π] range.
        QCar LiDAR gives 0 to 2π; we need centered angles for front detection.
        0 rad (or 2π) = directly ahead
        π/2 = left
        -π/2 (or 3π/2) = right
        """
        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi
        return angle_rad

    def process(self, raw_angles, raw_distances):
        """
        Process raw LiDAR scan into zone analysis.

        Returns a dictionary with:
          - front_min_dist: closest object in front arc
          - front_obstacle: True if obstacle in front
          - front_angle: angle of closest front object (normalized)
          - left_clearance: average clearance on left side
          - right_clearance: average clearance on right side
          - left_min_dist: closest object on left
          - right_min_dist: closest object on right
          - rear_min_dist: closest object behind
          - all_blocked: True if no safe direction exists
          - obstacle_side: 'left', 'right', or 'center'
          - front_clear: True if front arc is clear above threshold
        """
        result = {
            'front_min_dist': 5.0,
            'front_obstacle': False,
            'front_angle': 0.0,
            'left_clearance': 5.0,
            'right_clearance': 5.0,
            'left_min_dist': 5.0,
            'right_min_dist': 5.0,
            'rear_min_dist': 5.0,
            'all_blocked': False,
            'obstacle_side': 'center',
            'front_clear': True,
            'emergency': False,
        }

        if raw_distances is None or len(raw_distances) == 0:
            return result

        angles = np.array(raw_angles, dtype=float)
        distances = np.array(raw_distances, dtype=float)

        # Filter valid measurements
        valid_mask = (
            (distances > self.cfg.LIDAR_MIN_VALID) &
            (distances < self.cfg.LIDAR_MAX_VALID) &
            np.isfinite(distances)
        )
        if not np.any(valid_mask):
            return result

        valid_angles = angles[valid_mask]
        valid_distances = distances[valid_mask]

        # Normalize all angles to [-π, +π]
        normalized_angles = np.array(
            [self._normalize_angle(a) for a in valid_angles]
        )

        # ======================
        # FRONT ARC ANALYSIS
        # ======================
        front_mask = np.abs(normalized_angles) < self.front_half_rad
        if np.any(front_mask):
            front_dists = valid_distances[front_mask]
            front_angs  = normalized_angles[front_mask]

            min_idx = np.argmin(front_dists)
            front_min = front_dists[min_idx]
            front_ang = front_angs[min_idx]

            # Smooth with rolling average to reduce noise
            self.front_dist_buffer.append(front_min)
            smoothed_front = np.mean(self.front_dist_buffer)

            result['front_min_dist'] = smoothed_front
            result['front_angle'] = front_ang
            result['front_obstacle'] = smoothed_front < self.cfg.OBSTACLE_THRESHOLD
            result['front_clear'] = smoothed_front > self.cfg.CLEAR_PATH_THRESHOLD
            result['emergency'] = smoothed_front < self.cfg.EMERGENCY_THRESHOLD

            # Classify obstacle side
            if front_ang > math.radians(10):
                result['obstacle_side'] = 'left'
            elif front_ang < math.radians(-10):
                result['obstacle_side'] = 'right'
            else:
                result['obstacle_side'] = 'center'
        else:
            self.front_dist_buffer.append(5.0)

        # ======================
        # LEFT SIDE ANALYSIS
        # (Positive angles in normalized = left of vehicle)
        # ======================
        left_mask = (
            (normalized_angles > self.side_start_rad) &
            (normalized_angles < self.side_end_rad)
        )
        if np.any(left_mask):
            left_dists = valid_distances[left_mask]
            result['left_clearance'] = float(np.mean(left_dists))
            result['left_min_dist'] = float(np.min(left_dists))

        # ======================
        # RIGHT SIDE ANALYSIS
        # (Negative angles in normalized = right of vehicle)
        # ======================
        right_mask = (
            (normalized_angles < -self.side_start_rad) &
            (normalized_angles > -self.side_end_rad)
        )
        if np.any(right_mask):
            right_dists = valid_distances[right_mask]
            result['right_clearance'] = float(np.mean(right_dists))
            result['right_min_dist'] = float(np.min(right_dists))

        # ======================
        # REAR ANALYSIS
        # ======================
        rear_mask = (
            (np.abs(normalized_angles) > self.rear_start_rad)
        )
        if np.any(rear_mask):
            rear_dists = valid_distances[rear_mask]
            result['rear_min_dist'] = float(np.min(rear_dists))

        # ======================
        # ALL-BLOCKED CHECK
        # ======================
        result['all_blocked'] = (
            result['front_min_dist'] < self.cfg.OBSTACLE_THRESHOLD and
            result['left_min_dist'] < self.cfg.CLEARANCE_THRESHOLD * 0.5 and
            result['right_min_dist'] < self.cfg.CLEARANCE_THRESHOLD * 0.5
        )

        return result


# ===========================================================================
# DEAD RECKONING TRACKER — Tracks accumulated steering for lane return
# ===========================================================================
class DeadReckoningTracker:
    """
    Tracks cumulative steering to enable symmetric counter-steering.
    When the vehicle steers left during avoidance, it accumulates positive
    steering integral. During realignment, it counter-steers (negative)
    until the integral returns to approximately zero.
    """

    def __init__(self):
        self.steering_integral = 0.0  # Accumulated (steering * dt)
        self.avoidance_direction = 0  # +1 = avoiding left, -1 = avoiding right

    def reset(self):
        """Reset for new avoidance maneuver."""
        self.steering_integral = 0.0
        self.avoidance_direction = 0

    def accumulate(self, steering_value, dt):
        """Add steering contribution during avoidance or realignment."""
        self.steering_integral += steering_value * dt

    def get_realign_steering(self, config: Config):
        """
        Calculate the steering command needed to return to original trajectory.
        Returns steering value that opposes the accumulated integral.
        """
        if abs(self.steering_integral) < config.REALIGN_TOLERANCE:
            return 0.0  # Close enough, stop counter-steering

        # Steer in opposite direction of accumulated offset
        if self.steering_integral > 0:
            return -config.REALIGN_STEERING
        else:
            return config.REALIGN_STEERING

    def is_realigned(self, config: Config):
        """Check if vehicle has counter-steered enough to be realigned."""
        return abs(self.steering_integral) < config.REALIGN_TOLERANCE


# ===========================================================================
# MAIN CONTROLLER — The intelligent state machine
# ===========================================================================
class ObstacleAvoidanceController:
    """
    Main controller implementing the 6-state obstacle avoidance system.
    Designed for future extensibility with lane detection and sign recognition.
    """

    def __init__(self):
        self.cfg = Config()
        self.lidar_proc = LidarProcessor(self.cfg)
        self.dead_reckoning = DeadReckoningTracker()

        # State machine
        self.state = State.FORWARD
        self.prev_state = State.FORWARD
        self.state_start_time = 0.0
        self.state_entry_time = 0.0

        # Avoidance tracking
        self.avoidance_direction = 0     # +1 = steering left, -1 = steering right
        self.last_avoidance_time = -10.0  # When last avoidance ended (for cooldown)
        self.avoidance_start_time = 0.0

        # Stabilization
        self.current_steering = 0.0      # Smoothed steering output
        self.target_steering = 0.0

        # Telemetry
        self.last_telemetry_time = 0.0
        self.loop_count = 0
        self.scan_result = {}

        # Decision info for telemetry
        self.decision_info = ""

    def _transition_to(self, new_state, current_time):
        """Clean state transition with logging."""
        old_state = self.state
        self.prev_state = old_state
        self.state = new_state
        self.state_entry_time = current_time

        transition_msg = f"🔄 {old_state.name} → {new_state.name}"

        if new_state == State.EMERGENCY_STOP:
            print(f"🚨 {transition_msg} — VEHICLE BLOCKED")
        elif new_state == State.AVOID_EXECUTE:
            direction = "LEFT" if self.avoidance_direction > 0 else "RIGHT"
            print(f"🚧 {transition_msg} — Avoiding {direction}")
        elif new_state == State.REALIGN:
            print(f"↩️  {transition_msg} — Counter-steering to realign")
        elif new_state == State.STABILIZE:
            print(f"✅ {transition_msg} — Stabilizing straight")
        elif new_state == State.FORWARD:
            print(f"🟢 {transition_msg} — Resuming forward drive")
        else:
            print(f"   {transition_msg}")

    def _smooth_steering(self, target, rate):
        """
        Gradually move current steering toward target to prevent jerky motion.
        """
        diff = target - self.current_steering
        if abs(diff) < rate:
            self.current_steering = target
        elif diff > 0:
            self.current_steering += rate
        else:
            self.current_steering -= rate
        return self.current_steering

    def _get_leds(self, throttle, steering):
        """Generate LED array based on current driving state."""
        if self.state == State.EMERGENCY_STOP:
            return self.cfg.LED_HAZARD.copy()

        leds = self.cfg.LED_HEADLIGHTS.copy()

        if steering > 0.15:
            leds = np.maximum(leds, self.cfg.LED_LEFT_TURN)
        elif steering < -0.15:
            leds = np.maximum(leds, self.cfg.LED_RIGHT_TURN)

        if throttle <= 0.0:
            leds = np.maximum(leds, self.cfg.LED_BRAKE)

        return leds

    def _check_battery(self, voltage):
        """Monitor battery and return True if safe to continue."""
        if voltage < self.cfg.BATTERY_CRITICAL_VOLTAGE:
            print(f"🔋❌ CRITICAL: Battery at {voltage:.1f}V — EMERGENCY STOP")
            return False
        elif voltage < self.cfg.BATTERY_WARNING_VOLTAGE:
            print(f"🔋⚠️  WARNING: Battery low at {voltage:.1f}V")
        return True

    # ===================================================================
    # STATE HANDLERS
    # ===================================================================

    def _handle_forward(self, scan, current_time):
        """
        FORWARD state: Drive straight, monitor for obstacles.
        Transition to CHECK_OBSTACLE if obstacle detected.
        """
        throttle = self.cfg.BASE_THROTTLE
        steering = 0.0
        self.decision_info = f"CRUISING (front:{scan['front_min_dist']:.2f}m)"

        # Smoothly return steering to zero if coming from stabilize
        steering = self._smooth_steering(0.0, self.cfg.STABILIZE_STEERING_RATE * 2)

        # Check for emergency
        if scan['emergency']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # Check for obstacle (with cooldown)
        cooldown_elapsed = current_time - self.last_avoidance_time
        if scan['front_obstacle'] and cooldown_elapsed > self.cfg.COOLDOWN_PERIOD:
            self._transition_to(State.CHECK_OBSTACLE, current_time)
            return throttle * 0.5, steering  # Slow down while checking

        return throttle, steering

    def _handle_check_obstacle(self, scan, current_time):
        """
        CHECK_OBSTACLE state: Slow down and confirm obstacle is real.
        Uses multiple readings (inherent in smoothing buffer) to confirm.
        Transition to AVOID_DECISION or back to FORWARD.
        """
        throttle = self.cfg.AVOIDANCE_THROTTLE * 0.5  # Slow down significantly
        steering = self.current_steering  # Hold current steering
        self.decision_info = (
            f"CHECKING (front:{scan['front_min_dist']:.2f}m "
            f"L:{scan['left_clearance']:.2f} R:{scan['right_clearance']:.2f})"
        )

        # Emergency check
        if scan['emergency']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # If obstacle disappeared (noise), return to forward
        if not scan['front_obstacle']:
            elapsed = current_time - self.state_entry_time
            if elapsed > 0.15:  # Give it a moment to confirm
                self._transition_to(State.FORWARD, current_time)
                return self.cfg.BASE_THROTTLE, 0.0

        # All blocked check
        if scan['all_blocked']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # Confirmed obstacle — move to decision
        elapsed = current_time - self.state_entry_time
        if elapsed > 0.1:  # Brief confirmation delay
            self._transition_to(State.AVOID_DECISION, current_time)

        return throttle, steering

    def _handle_avoid_decision(self, scan, current_time):
        """
        AVOID_DECISION state: Analyze left and right clearance to choose
        the safest avoidance direction.
        """
        throttle = self.cfg.AVOIDANCE_THROTTLE * 0.3  # Nearly stopped while deciding
        steering = 0.0

        left_clear  = scan['left_clearance']
        right_clear = scan['right_clearance']
        left_min    = scan['left_min_dist']
        right_min   = scan['right_min_dist']
        obs_side    = scan['obstacle_side']

        self.decision_info = (
            f"DECIDING (obs:{obs_side} "
            f"L_clr:{left_clear:.2f} R_clr:{right_clear:.2f})"
        )

        # Decision logic:
        # 1. If one side has significantly more clearance, choose it
        # 2. If obstacle is on one side, prefer the opposite
        # 3. If both sides are roughly equal, default based on obstacle position

        clearance_ratio = 1.5  # One side must be 1.5x clearer to be preferred

        if left_clear > right_clear * clearance_ratio and left_min > self.cfg.CLEARANCE_THRESHOLD:
            # Left side is significantly clearer — avoid to the left
            self.avoidance_direction = 1  # Positive steering = left
            reason = f"LEFT CLEARER ({left_clear:.2f} vs {right_clear:.2f})"

        elif right_clear > left_clear * clearance_ratio and right_min > self.cfg.CLEARANCE_THRESHOLD:
            # Right side is significantly clearer — avoid to the right
            self.avoidance_direction = -1  # Negative steering = right
            reason = f"RIGHT CLEARER ({right_clear:.2f} vs {left_clear:.2f})"

        elif left_min > self.cfg.CLEARANCE_THRESHOLD and right_min < self.cfg.CLEARANCE_THRESHOLD:
            # Only left is safe
            self.avoidance_direction = 1
            reason = "ONLY LEFT SAFE"

        elif right_min > self.cfg.CLEARANCE_THRESHOLD and left_min < self.cfg.CLEARANCE_THRESHOLD:
            # Only right is safe
            self.avoidance_direction = -1
            reason = "ONLY RIGHT SAFE"

        elif left_min < self.cfg.CLEARANCE_THRESHOLD and right_min < self.cfg.CLEARANCE_THRESHOLD:
            # Neither side is safe enough — emergency
            self._transition_to(State.EMERGENCY_STOP, current_time)
            print(f"🚨 Neither side safe! L:{left_min:.2f} R:{right_min:.2f}")
            return 0.0, 0.0

        else:
            # Both sides acceptable — choose based on obstacle position
            if obs_side == 'left':
                self.avoidance_direction = -1  # Obstacle on left, go right
                reason = "OBS ON LEFT → GO RIGHT"
            elif obs_side == 'right':
                self.avoidance_direction = 1   # Obstacle on right, go left
                reason = "OBS ON RIGHT → GO LEFT"
            else:
                # Center obstacle, pick side with more clearance
                if left_clear >= right_clear:
                    self.avoidance_direction = 1
                    reason = "CENTER OBS → LEFT (more space)"
                else:
                    self.avoidance_direction = -1
                    reason = "CENTER OBS → RIGHT (more space)"

        print(f"   📐 Decision: {reason}")

        # Reset dead reckoning for this avoidance maneuver
        self.dead_reckoning.reset()
        self.dead_reckoning.avoidance_direction = self.avoidance_direction
        self.avoidance_start_time = current_time

        self._transition_to(State.AVOID_EXECUTE, current_time)
        return throttle, steering

    def _handle_avoid_execute(self, scan, current_time):
        """
        AVOID_EXECUTE state: Actively steer around the obstacle.
        Uses real-time LiDAR feedback to determine when obstacle is cleared.
        Tracks steering integral for dead reckoning.
        """
        dt = self.cfg.LOOP_PERIOD
        elapsed = current_time - self.state_entry_time
        total_avoidance_time = current_time - self.avoidance_start_time

        # Safety timeout
        if total_avoidance_time > self.cfg.MAX_AVOIDANCE_DURATION:
            print(f"⏰ Avoidance timeout ({self.cfg.MAX_AVOIDANCE_DURATION}s) — forcing realign")
            self._transition_to(State.REALIGN, current_time)
            return self.cfg.AVOIDANCE_THROTTLE, 0.0

        # Emergency check
        if scan['emergency']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # Calculate steering command
        target_steer = self.avoidance_direction * self.cfg.AVOIDANCE_STEERING

        # Check if the side we're steering toward is getting too close
        if self.avoidance_direction > 0:  # Steering left
            if scan['left_min_dist'] < self.cfg.SIDE_DANGER_THRESHOLD:
                target_steer *= 0.3  # Reduce steering to avoid wall on left
        else:  # Steering right
            if scan['right_min_dist'] < self.cfg.SIDE_DANGER_THRESHOLD:
                target_steer *= 0.3  # Reduce steering to avoid wall on right

        # Smooth steering application
        steering = self._smooth_steering(target_steer, 0.05)
        throttle = self.cfg.AVOIDANCE_THROTTLE

        # Track steering for dead reckoning
        self.dead_reckoning.accumulate(steering, dt)

        self.decision_info = (
            f"AVOIDING {'LEFT' if self.avoidance_direction > 0 else 'RIGHT'} "
            f"(front:{scan['front_min_dist']:.2f}m "
            f"integral:{self.dead_reckoning.steering_integral:.3f} "
            f"t:{elapsed:.1f}s)"
        )

        # === REAL-TIME SENSOR CHECK: Is the obstacle cleared? ===
        # Conditions to move to REALIGN:
        # 1. Front arc is clear (obstacle no longer ahead)
        # 2. We've been turning for at least a minimum time (0.3s debounce)
        min_turn_time = 0.3  # Minimum time to ensure we've actually turned

        if scan['front_clear'] and elapsed > min_turn_time:
            # Obstacle is no longer in front — start counter-steering
            print(
                f"   ✅ Path clear! Front: {scan['front_min_dist']:.2f}m "
                f"(threshold: {self.cfg.CLEAR_PATH_THRESHOLD}m) "
                f"Steering integral: {self.dead_reckoning.steering_integral:.3f}"
            )
            self._transition_to(State.REALIGN, current_time)

        return throttle, steering

    def _handle_realign(self, scan, current_time):
        """
        REALIGN state: Counter-steer to return to original trajectory.
        Uses dead reckoning integral to determine how much counter-steering
        is needed. Monitors LiDAR to avoid steering into new obstacles.
        """
        dt = self.cfg.LOOP_PERIOD
        elapsed = current_time - self.state_entry_time
        total_avoidance_time = current_time - self.avoidance_start_time

        # Safety timeout
        if total_avoidance_time > self.cfg.MAX_AVOIDANCE_DURATION:
            print(f"⏰ Realignment timeout — forcing stabilize")
            self._transition_to(State.STABILIZE, current_time)
            return self.cfg.STABILIZE_THROTTLE, 0.0

        # Emergency check — if new obstacle appears during realignment
        if scan['emergency']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # If a new obstacle appears in front during realignment, go back to check
        if scan['front_obstacle']:
            print(f"   ⚠️  New obstacle during realignment! Rechecking...")
            self._transition_to(State.CHECK_OBSTACLE, current_time)
            return self.cfg.AVOIDANCE_THROTTLE * 0.5, 0.0

        # Get counter-steering command from dead reckoning
        target_steer = self.dead_reckoning.get_realign_steering(self.cfg)

        # Smooth the steering transition
        steering = self._smooth_steering(target_steer, 0.04)
        throttle = self.cfg.STABILIZE_THROTTLE

        # Track counter-steering
        self.dead_reckoning.accumulate(steering, dt)

        self.decision_info = (
            f"REALIGNING "
            f"(integral:{self.dead_reckoning.steering_integral:.3f} "
            f"steer:{steering:.2f} t:{elapsed:.1f}s)"
        )

        # Check if we've counter-steered enough
        if self.dead_reckoning.is_realigned(self.cfg):
            print(
                f"   ↩️  Realigned! "
                f"Final integral: {self.dead_reckoning.steering_integral:.4f}"
            )
            self._transition_to(State.STABILIZE, current_time)

        return throttle, steering

    def _handle_stabilize(self, scan, current_time):
        """
        STABILIZE state: Hold straight for a brief period to let the vehicle
        settle and prevent oscillations before returning to FORWARD.
        """
        elapsed = current_time - self.state_entry_time

        # Smoothly bring steering to zero
        steering = self._smooth_steering(0.0, self.cfg.STABILIZE_STEERING_RATE)
        throttle = self.cfg.STABILIZE_THROTTLE

        self.decision_info = (
            f"STABILIZING "
            f"(steer:{steering:.3f} t:{elapsed:.1f}/{self.cfg.STABILIZE_DURATION:.1f}s)"
        )

        # Emergency check
        if scan['emergency']:
            self._transition_to(State.EMERGENCY_STOP, current_time)
            return 0.0, 0.0

        # Wait for stabilization period AND steering to be near zero
        if elapsed > self.cfg.STABILIZE_DURATION and abs(steering) < 0.02:
            self.last_avoidance_time = current_time  # Start cooldown
            self.current_steering = 0.0  # Force zero
            self._transition_to(State.FORWARD, current_time)

        return throttle, steering

    def _handle_emergency_stop(self, scan, current_time):
        """
        EMERGENCY_STOP state: Full stop, wait until path clears.
        Does NOT reverse — just waits.
        """
        elapsed = current_time - self.state_entry_time

        self.decision_info = (
            f"🚨 EMERGENCY STOP "
            f"(front:{scan['front_min_dist']:.2f}m "
            f"L:{scan['left_min_dist']:.2f} R:{scan['right_min_dist']:.2f} "
            f"t:{elapsed:.1f}s)"
        )

        # Check if path has cleared
        if scan['front_clear'] and not scan['all_blocked']:
            print(f"   🟢 Path cleared after {elapsed:.1f}s emergency stop")
            self.last_avoidance_time = current_time
            self._transition_to(State.FORWARD, current_time)

        return 0.0, 0.0  # Full stop

    # ===================================================================
    # MAIN UPDATE — Called every loop iteration
    # ===================================================================
    def update(self, scan_result, current_time):
        """
        Main update function. Routes to appropriate state handler.

        Args:
            scan_result: dict from LidarProcessor.process()
            current_time: current timestamp

        Returns:
            (throttle, steering, LEDs) tuple
        """
        self.scan_result = scan_result
        self.loop_count += 1

        # Route to state handler
        state_handlers = {
            State.FORWARD:        self._handle_forward,
            State.CHECK_OBSTACLE: self._handle_check_obstacle,
            State.AVOID_DECISION: self._handle_avoid_decision,
            State.AVOID_EXECUTE:  self._handle_avoid_execute,
            State.REALIGN:        self._handle_realign,
            State.STABILIZE:      self._handle_stabilize,
            State.EMERGENCY_STOP: self._handle_emergency_stop,
        }

        handler = state_handlers.get(self.state, self._handle_forward)
        throttle, steering = handler(scan_result, current_time)

        # Clamp outputs for safety
        throttle = float(np.clip(throttle, 0.0, 0.3))
        steering = float(np.clip(steering, -self.cfg.MAX_STEERING, self.cfg.MAX_STEERING))

        # Generate LEDs
        leds = self._get_leds(throttle, steering)

        return throttle, steering, leds

    def get_telemetry_string(self, battery_voltage, elapsed_time):
        """Generate formatted telemetry string."""
        scan = self.scan_result
        return (
            f"⏱️ {elapsed_time:6.1f}s | "
            f"{self.state.name:15s} | "
            f"{self.decision_info:50s} | "
            f"T:{self.current_steering:5.2f} | "  # This shows actual steering
            f"🔋{battery_voltage:5.1f}V"
        )


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def main():
    """
    Main function — initializes hardware and runs the control loop.
    This is a standalone script: run it and the car drives autonomously.
    """
    cfg = Config()

    print("=" * 70)
    print("  INTELLIGENT QCAR2 OBSTACLE AVOIDANCE SYSTEM v2.0")
    print("=" * 70)
    print(f"  Forward speed:       {cfg.BASE_THROTTLE}")
    print(f"  Obstacle threshold:  {cfg.OBSTACLE_THRESHOLD}m")
    print(f"  Clearance threshold: {cfg.CLEARANCE_THRESHOLD}m")
    print(f"  Emergency threshold: {cfg.EMERGENCY_THRESHOLD}m")
    print(f"  Max steering:        ±{cfg.MAX_STEERING}")
    print(f"  Cooldown period:     {cfg.COOLDOWN_PERIOD}s")
    print(f"  Max avoidance time:  {cfg.MAX_AVOIDANCE_DURATION}s")
    print(f"  Run time:            {cfg.RUN_TIME}s")
    print(f"  Sample rate:         {cfg.SAMPLE_RATE}Hz")
    print(f"  Physical QCar:       {IS_PHYSICAL_QCAR}")
    print("=" * 70)
    print("  Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Initialize components
    controller = ObstacleAvoidanceController()
    lidar_proc = LidarProcessor(cfg)

    myLidar = None
    myCar_context = None

    try:
        # ---- Initialize LiDAR ----
        print("Initializing LiDAR...", end=" ", flush=True)
        myLidar = QCarLidar(
            numMeasurements=cfg.LIDAR_NUM_MEASUREMENTS,
            rangingDistanceMode=cfg.LIDAR_RANGING_MODE,
            interpolationMode=cfg.LIDAR_INTERPOLATION
        )
        print("✅ Done")

        # Small delay to let LiDAR spin up
        time.sleep(1.0)

        # Do a few dummy reads to flush initial bad data
        print("Warming up LiDAR (3 dummy reads)...", end=" ", flush=True)
        for _ in range(3):
            myLidar.read()
            time.sleep(0.1)
        print("✅ Done")

        # ---- Initialize QCar ----
        print("Initializing QCar...", end=" ", flush=True)
        with QCar(readMode=1, frequency=cfg.SAMPLE_RATE) as myCar:
            print("✅ Done")

            # Initial battery check
            myCar.read()
            initial_voltage = myCar.batteryVoltage
            print(f"🔋 Battery voltage: {initial_voltage:.1f}V")

            if initial_voltage < cfg.BATTERY_CRITICAL_VOLTAGE:
                print("❌ Battery too low to operate! Exiting.")
                return

            print()
            print("🚀 Starting autonomous drive!")
            print("-" * 70)

            t0 = time.time()
            last_telemetry = t0
            last_loop_time = t0

            # ============================================================
            # MAIN CONTROL LOOP
            # ============================================================
            while True:
                current_time = time.time()
                elapsed = current_time - t0

                # Check run time limit
                if elapsed > cfg.RUN_TIME:
                    print(f"\n⏰ Run time limit ({cfg.RUN_TIME}s) reached")
                    break

                # ---- Read Sensors ----
                myCar.read()
                myLidar.read()

                # ---- Process LiDAR Data ----
                scan_result = lidar_proc.process(
                    myLidar.angles,
                    myLidar.distances
                )

                # ---- Battery Safety ----
                battery_ok = controller._check_battery(myCar.batteryVoltage)
                if not battery_ok:
                    myCar.write(0.0, 0.0, cfg.LED_HAZARD)
                    print("🛑 Battery critical — stopped")
                    break

                # ---- Run State Machine ----
                throttle, steering, leds = controller.update(
                    scan_result, current_time
                )

                # ---- Apply Commands ----
                myCar.write(throttle, steering, leds)

                # ---- Telemetry Output ----
                if current_time - last_telemetry >= cfg.TELEMETRY_INTERVAL:
                    telemetry = controller.get_telemetry_string(
                        myCar.batteryVoltage, elapsed
                    )
                    print(telemetry)
                    last_telemetry = current_time

                # ---- Loop Timing ----
                # Maintain consistent loop rate
                loop_dt = time.time() - current_time
                sleep_time = cfg.LOOP_PERIOD - loop_dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("  🛑 INTERRUPTED BY USER — Safe shutdown initiated")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("🛑 Performing emergency stop...")

    finally:
        # ============================================================
        # SAFE SHUTDOWN — Always runs
        # ============================================================
        print("\nShutting down...")

        # Stop the car
        try:
            # We need to handle the case where myCar might already be closed
            # by the context manager. Try writing stop command.
            # If we're still inside the 'with' block this works.
            # If not, we need a fallback.
            pass  # The context manager handles QCar cleanup
        except Exception:
            pass

        # Terminate LiDAR
        try:
            if myLidar is not None:
                myLidar.terminate()
                print("  ✅ LiDAR terminated")
        except Exception as e:
            print(f"  ⚠️  LiDAR shutdown error: {e}")

        print("  ✅ Shutdown complete")
        print()

        # Print session summary
        try:
            total_time = time.time() - t0
            print(f"  Session duration: {total_time:.1f}s")
            print(f"  Total loop iterations: {controller.loop_count}")
            if controller.loop_count > 0:
                avg_hz = controller.loop_count / total_time
                print(f"  Average loop rate: {avg_hz:.1f} Hz")
            print(f"  Final state: {controller.state.name}")
        except Exception:
            pass

        print()


# ===========================================================================
# FUTURE INTEGRATION HOOKS
# ===========================================================================
"""
To integrate with lane detection:
    1. Add a LaneDetector class that processes camera frames
    2. In the FORWARD state, use lane offset to adjust steering
    3. In REALIGN state, use lane detection instead of dead reckoning
       if available (fall back to dead reckoning if lane not detected)

To integrate with RT-DETR traffic sign detection:
    1. Add a SignDetector class
    2. Create new states: SIGN_RESPONSE, INTERSECTION, SPEED_ADJUST
    3. In FORWARD state, check for detected signs and transition accordingly
    4. Signs can override speed (speed limit) or trigger stops (stop sign)

To add a higher-level planner:
    1. Create a MissionPlanner class that sets waypoints
    2. The planner can override the default FORWARD behavior with
       targeted steering toward waypoints
    3. Obstacle avoidance still takes priority (interrupt-style)

Example integration point in the main loop:
    # After reading sensors:
    # lane_offset = lane_detector.process(camera_frame)
    # signs = sign_detector.process(camera_frame)
    # controller.set_lane_offset(lane_offset)  # Adjusts FORWARD steering
    # controller.process_signs(signs)           # May trigger state changes
"""


if __name__ == "__main__":
    main()
