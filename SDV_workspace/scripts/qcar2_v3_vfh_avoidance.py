#!/usr/bin/env python3
"""
=============================================================================
 QUANSER QCAR2 INDUSTRIAL OBSTACLE AVOIDANCE SYSTEM v3.0 (VFH+)
=============================================================================
 Paradigm: Vector Field Histogram (VFH) with Kinematic Path Projection
 Features:
   - Dynamic Gap Discovery (Wide-Valley Search)
   - Ackermann Kinematic Odometry (X, Y, Theta tracking)
   - PD-Targeted Steering (Smooth, no binary angles)
   - Car-Width Safety Masking (Accounting for physical dimensions)
   - Multi-state Robust FSM with priority overrides
=============================================================================
"""

import os
import sys
import numpy as np
import time
import math
from enum import IntEnum
from collections import deque

# Hardware Compatibility Layer
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

# QLabs simulation setup
if not IS_PHYSICAL_QCAR:
    try:
        import qlabs_setup
        qlabs_setup.setup()
    except ImportError:
        pass

# ===========================================================================
# VEHICLE & SYSTEM CONFIGURATION
# ===========================================================================
class Config:
    # --- Physical Dimensions (Meters) ---
    CAR_WIDTH               = 0.192
    WHEELBASE               = 0.256
    SAFETY_MARGIN           = 0.15   # Increased for better clearance
    TOTAL_WIDTH_REQD        = CAR_WIDTH + (2 * SAFETY_MARGIN)

    # --- Dynamics Approximation ---
    MAX_SPEED_MPS           = 2.0    # Approx speed at throttle=1.0
    STEER_RATIO             = 0.5    # Max physical steering (radians)

    # --- Timing & Control ---
    LOOP_HZ                 = 30
    DT                      = 1.0 / LOOP_HZ
    TELEMETRY_HZ            = 2.0

    # --- LiDAR Parameters ---
    NUM_BINS                = 72     # 5° per bin
    MAX_RANGE               = 4.5    # Max reliable distance
    MIN_RANGE               = 0.18   # Ignore closer reflections
    OBSTACLE_THRESHOLD      = 0.8    # Distance to trigger reactive mode
    CLEAR_THRESHOLD         = 1.2    # Distance to consider path "Wide Open"
    EMERGENCY_STOP_DIST     = 0.28   # Hard stop
    LIDAR_ANGLE_OFFSET      = 180.0  # Rotation in DEGREES
    LIDAR_REVERSE           = True   # Flip angles if left/right are swapped

    # --- Control Gains ---
    K_P_STEER               = 1.5    # Increased gain for sharper dodging
    K_P_RETURN              = 1.5    # Gain for returning to lane (Lateral)
    K_P_YAW                 = 1.0    # Gain for returning to lane (Heading)

    # --- Throttle Profiles ---
    THROTTLE_CRUISE         = 0.12
    THROTTLE_AVOID          = 0.08
    THROTTLE_STABILIZE      = 0.10

    # --- LED Mappings ---
    LED_HEADLIGHTS          = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
    LED_HAZARD              = np.ones(8, dtype=float)
    LED_LEFT                = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    LED_RIGHT               = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float)
    LED_BRAKE               = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=float)

# ===========================================================================
# KINEMATIC ODOMETRY TRACKER
# ===========================================================================
class KinematicTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.distance_traveled = 0.0

    def update(self, throttle, steering, dt):
        v = throttle * self.cfg.MAX_SPEED_MPS
        delta = steering * self.cfg.STEER_RATIO
        
        # Ackermann update
        if abs(delta) > 0.001:
            omega = (v / self.cfg.WHEELBASE) * math.tan(delta)
        else:
            omega = 0.0

        self.yaw += omega * dt
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.distance_traveled += abs(v * dt)

# ===========================================================================
# VFH+ LIDAR PROCESSOR
# ===========================================================================
class VectorFieldHistogram:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bin_size = 360 / cfg.NUM_BINS
        self.debug_closest = (5.0, 0.0)
        
    def process_lidar(self, raw_angles, raw_distances):
        if raw_distances is None or len(raw_distances) == 0:
            return 0.0, 5.0, False

        angles = np.array(raw_angles)
        if self.cfg.LIDAR_REVERSE:
            angles = (2 * math.pi - angles) % (2 * math.pi)
        
        # Apply rotation to align front
        angles = (angles + math.radians(self.cfg.LIDAR_ANGLE_OFFSET)) % (2 * math.pi)
        
        distances = np.array(raw_distances)

        mask = (distances > self.cfg.MIN_RANGE) & (distances < self.cfg.MAX_RANGE)
        
        # Chassis Mask: Ignore points very close to the car at the back/sides
        # QCar2 pillars/cables are typically at degrees [110-250]
        v_ang_raw = angles[mask]
        v_dist_raw = distances[mask]
        
        chassis_mask = ~((v_dist_raw < 0.4) & (np.abs(np.degrees((v_ang_raw + math.pi) % (2*math.pi) - math.pi)) > 110))
        v_dist = v_dist_raw[chassis_mask]
        v_ang = v_ang_raw[chassis_mask]

        if len(v_dist) == 0:
            return 0.0, 5.0, False

        for d, a in zip(v_dist, v_ang):
            expansion = math.atan2(self.cfg.TOTAL_WIDTH_REQD / 2.0, d)
            start_ang = (math.degrees(a) - math.degrees(expansion)) % 360
            end_ang   = (math.degrees(a) + math.degrees(expansion)) % 360
            
            s_bin = int(start_ang // self.bin_size)
            e_bin = int(end_ang // self.bin_size)
            
            if s_bin <= e_bin:
                hist[s_bin:e_bin+1] = np.minimum(hist[s_bin:e_bin+1], d)
            else:
                hist[s_bin:] = np.minimum(hist[s_bin:], d)
                hist[:e_bin+1] = np.minimum(hist[:e_bin+1], d)

        # 4. Find the 'Best Front Gap'
        best_score = -1.0
        best_angle = 0.0
        
        # Front arc: -100 to 100 degrees
        for i in range(self.cfg.NUM_BINS):
            angle = i * self.bin_size
            if angle > 180: angle -= 360
            if abs(angle) > 100: continue
            
            dist = hist[i]
            if dist < self.cfg.OBSTACLE_THRESHOLD: continue
            
            score = dist * (math.cos(math.radians(angle)) + 1.2)
            if score > best_score:
                best_score = score
                best_angle = math.radians(angle)

        # Minimum Front distance for emergency stop (using filtered data)
        # Center angles around 0 for front mask
        v_ang_centered = (v_ang + math.pi) % (2 * math.pi) - math.pi
        front_mask = (np.abs(np.degrees(v_ang_centered)) < 40) # Slightly tighter front arc
        min_front = np.min(v_dist[front_mask]) if np.any(front_mask) else 5.0

        # Debug: Show closest overall object location
        if len(v_dist) > 0:
            idx = np.argmin(v_dist)
            self.debug_closest = (v_dist[idx], math.degrees(v_ang_centered[idx]))
        else:
            self.debug_closest = (5.0, 0.0)

        is_blocked = (best_score <= 0)
        return best_angle, min_front, is_blocked

# ===========================================================================
# FSM CONTROLLER
# ===========================================================================
class CarState(IntEnum):
    CRUISE = 0; AVOID = 1; RETURN = 2; STABILIZE = 3; EMERGENCY = 4

class Controller:
    def __init__(self):
        self.cfg = Config(); self.tracker = KinematicTracker(self.cfg); self.vfh = VectorFieldHistogram(self.cfg)
        self.state = CarState.CRUISE; self.state_time = time.time(); self.steer = 0.0; self.msg = ""
        self.closest_info = ""

    def step(self, angles, distances, current_time):
        gap_ang, min_f, blocked = self.vfh.process_lidar(angles, distances)
        throttle = 0.0; steer = 0.0; leds = self.cfg.LED_HEADLIGHTS.copy()
        
        c_dist, c_ang = self.vfh.debug_closest
        self.closest_info = f"Closest:{c_dist:.2f}m @ {c_ang:.0f}°"
        
        if min_f < self.cfg.EMERGENCY_STOP_DIST: self.state = CarState.EMERGENCY

        if self.state == CarState.CRUISE:
            self.msg = f"CRUISING | Front:{min_f:.2f}m"; throttle = self.cfg.THROTTLE_CRUISE
            if min_f < self.cfg.OBSTACLE_THRESHOLD:
                if blocked: self.state = CarState.EMERGENCY
                else: self.tracker.reset(); self.state = CarState.AVOID; print("🚧 Obstacle!")

        elif self.state == CarState.AVOID:
            self.msg = f"AVOIDING | Gap:{math.degrees(gap_ang):.1f}°"; throttle = self.cfg.THROTTLE_AVOID
            steer = gap_ang * self.cfg.K_P_STEER
            if min_f > self.cfg.CLEAR_THRESHOLD: self.state = CarState.RETURN; print("✅ Clear!")

        elif self.state == CarState.RETURN:
            self.msg = f"RETURNING | Y:{self.tracker.y:.2f}m ψ:{math.degrees(self.tracker.yaw):.0f}°"
            throttle = self.cfg.THROTTLE_STABILIZE
            # Return logic: steer toward the lane (y=0) while aligning yaw (0)
            # Increased K_P for stronger correction
            steer = (-2.0 * self.tracker.y) - (1.5 * self.tracker.yaw)
            
            if abs(self.tracker.y) < 0.05 and abs(self.tracker.yaw) < 0.1:
                self.state = CarState.STABILIZE; self.state_time = time.time()

        elif self.state == CarState.STABILIZE:
            self.msg = "SETTLING..."; throttle = self.cfg.THROTTLE_CRUISE
            if time.time() - self.state_time > 1.0: self.state = CarState.CRUISE; print("🟢 Resume")

        elif self.state == CarState.EMERGENCY:
            self.msg = "🚨 EMERGENCY"; leds = self.cfg.LED_HAZARD
            if min_f > self.cfg.OBSTACLE_THRESHOLD: self.state = CarState.CRUISE

        self.steer = 0.7 * self.steer + 0.3 * np.clip(steer, -1.0, 1.0)
        self.tracker.update(throttle, self.steer, self.cfg.DT)
        return throttle, self.steer, leds

def main():
    ctrl = Controller(); cfg = Config()
    try:
        myLidar = QCarLidar(numMeasurements=1000, rangingDistanceMode=2)
        with QCar(readMode=1, frequency=cfg.LOOP_HZ) as myCar:
            t0 = time.time(); last_tele = 0
            while time.time() - t0 < 600:
                myCar.read(); myLidar.read()
                th, st, leds = ctrl.step(myLidar.angles, myLidar.distances, time.time())
                myCar.write(th, st, leds)
                if time.time() - last_tele > 0.5: 
                    print(f"[{time.time()-t0:5.1f}s] {ctrl.msg} | {ctrl.closest_info}")
                    last_tele = time.time()
                time.sleep(max(0, cfg.DT - (time.time() - t0 % cfg.DT)))
    except KeyboardInterrupt: pass
    finally:
        if 'myLidar' in locals(): myLidar.terminate()

if __name__ == "__main__": main()
