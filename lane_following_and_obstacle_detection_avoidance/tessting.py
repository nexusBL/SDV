#!/usr/bin/env python3
"""
testing.py - QCar2 Autonomous Lane Following & Obstacle Avoidance Orchestrator
============================================================================
Update based on user feedback: Implements a robust state machine with dedicated
lane re-entry logic to ensure the car returns to the path after an avoidance maneuver.

State Machine:
  IDLE ──[press 'a']──> DRIVING ──> OBSTACLE_STOP ──> SEARCHING ──> DRIVING
    ^                      |                                        ^
    |                      |                                        |
    +──[press 's']─────────+                                        |
    |                                                               v
    +──────────────────────────────────────────────────────────────[Lost Lane]
    
Usage:
  python3 testing.py              # Full autonomous mode (motors enabled)
  python3 testing.py --preview    # Preview mode (motors disabled, vision only)

Controls (focus the OpenCV window):
  [a] - Start autonomous driving
  [s] - Stop (manual override)
  [q] - Quit and shutdown all hardware safely
"""

import os
import sys

# ── CRITICAL: Fix nvarguscamerasrc EGL authorization ──
# Must happen BEFORE any GStreamer or camera imports
# Save DISPLAY so we can restore it for cv2.imshow after camera init
_saved_display = os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

import time
import math
import cv2
import traceback
import numpy as np

# --- Re-import necessary modules here if they belong to separate files ---
# (Assuming config, hardware, perception, control are external modules as per original prompt)

# --- FIX: Update import to a singular AppConfig class ---
from config import AppConfig # Import the main configuration class

# --- FIX: Add explicit imports for external classes ---
from hardware.camera_manager import CameraManager
from hardware.safety_monitor import SafetyMonitor
from hardware.depth_monitor import DepthMonitor
from hardware.car_controller import CarController
from perception.lane_cv import LaneDetector
from perception.side_clearance import side_preference_from_pair, get_side_clearance_m, is_side_yellow_visible
from control.pid_controller import PIDController

# Reactive controller lives in the parent scripts folder (no PAL deps).
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)
from control.reactive_avoidance_controller import ReactiveController as ReactiveAvoidanceController
from control.reactive_avoidance_controller import ReactiveAvoidanceParams

# ══════════════════════════════════════════════════════════════════
# Supervisor Code (Re-implemented with Re-entry Logic)
# ══════════════════════════════════════════════════════════════════

class Supervisor:
    """
    Main application supervisor that orchestrates all QCar2 subsystems.
    """

    # --- State constants (Updated) ---
    STATE_IDLE          = "IDLE"
    STATE_DRIVING       = "DRIVING"
    STATE_OBSTACLE_STOP = "OBSTACLE_STOP"
    STATE_AVOIDING      = "AVOIDING"
    STATE_BACKING       = "BACKING"
    STATE_SEARCHING     = "SEARCHING"

    def __init__(self, preview_mode=False):
        """
        Args:
            preview_mode: bool - if True, motors are disabled (vision-only mode)
        """
        self.cfg = AppConfig() # <--- Load AppConfig here
        self.preview_mode = preview_mode
        self.state = self.STATE_IDLE
        self.running = True

        # FPS tracking
        self._fps_counter = 0
        self._fps_start = time.time()
        self._fps_display = 0.0

        # Fusion debounce counters
        self._depth_obstacle_frames = 0
        self._fused_clear_frames = 0
        self._lane_good_frames = 0
        self._avoid_stop_frames = 0
        self._drop_frames = 0
        self._avoid_state_frames = 0

        print("\n" + "=" * 50)
        print("  🚗  QCar2 Autonomous Lane Following System  🚗")
        print("=" * 50)
        mode_str = "PREVIEW (motors disabled)" if preview_mode else "LIVE (motors enabled)"
        print(f"  Mode: {mode_str}")
        print("=" * 50 + "\n")

        print("[Supervisor] Initializing subsystem drivers...")
        # --- FIX: Initialize classes from imported modules ---
        self.car = CarController(self.cfg)
        self.perception = LaneDetector(self.cfg)
        self.controller = PIDController(self.cfg)
        self.camera = CameraManager(self.cfg)
        self.safety = SafetyMonitor(self.cfg)
        self.depth = DepthMonitor(self.cfg)
        self.reactive_cfg = ReactiveAvoidanceParams(
            max_steer=float(self.cfg.control.max_steering),
            lidar_angle_offset_deg=float(self.cfg.safety.lidar_front_angle_deg),
            front_arc_deg=float(self.cfg.reactive.front_arc_deg),
            lidar_reverse=bool(self.cfg.reactive.lidar_reverse),
            car_width_m=float(self.cfg.safety.car_width_m),
            gap_max_angle_deg=float(self.cfg.safety.gap_max_angle_deg),
        )
        self.reactive = ReactiveAvoidanceController(self.reactive_cfg)

        self.last_steer = 0.0 # Store steering for re-entry
        self.avoid_direction = 0.0 # 1.0 left, -1.0 right
        self.side_clearance = 2.0 # Current side clearance in meters

    def run(self):
        """
        Main execution loop. Initializes hardware, runs the perception-control
        loop, and guarantees clean shutdown on exit.
        """
        print("[Supervisor] Initializing subsystem drivers...")
        try:
            self.camera.initialize()
            self.safety.initialize()
            self.depth.initialize()
            self.car.initialize()
        except Exception as e:
            print(f"\n[FATAL] Hardware initialization failed: {e}")
            traceback.print_exc()
            self._shutdown()
            return

        # Restore DISPLAY for cv2.imshow (stripped earlier for GStreamer/EGL)
        if _saved_display:
            os.environ["DISPLAY"] = _saved_display
            print(f"[Supervisor] Restored DISPLAY={_saved_display} for OpenCV.")

        print("\n[Supervisor] All systems nominal. Entering main loop.")
        print("[Supervisor] Controls: [a]=Start  [s]=Stop  [q]=Quit\n")

        # ── Phase 2: Main Loop ──
        try:
            while self.running:
                start_time = time.time()
                
                # --- 1. SENSOR POLLING ---
                # Sync hardware watchdog. Camera readAll happens inside get_frame().
                self.car.read()
                
                # ──── 1. CAPTURE FRAME ────
                frame = self.camera.get_frame()
                if frame is None:
                    self._drop_frames += 1
                    if self._drop_frames % 10 == 0:
                        print("[Supervisor] Warning: Dropped frame.")

                    # Always show a window so it doesn't look "stuck"
                    hud = np.zeros(
                        (self.cfg.camera.capture_height, self.cfg.camera.capture_width, 3),
                        dtype=np.uint8,
                    )
                    cv2.putText(
                        hud,
                        f"NO CAMERA FRAME (drops:{self._drop_frames})",
                        (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                    cv2.putText(
                        hud,
                        "Press 'q' to quit cleanly",
                        (40, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    self._update_fps()
                    self._render_hud(hud)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_input(key)

                    # If we have sustained drops, try restarting the camera
                    if self._drop_frames == 60:
                        print("[Supervisor] Restarting camera after sustained drops...")
                        try:
                            self.camera.terminate()
                        except Exception:
                            pass
                        time.sleep(0.5)
                        try:
                            self.camera.initialize()
                            self._drop_frames = 0
                        except Exception as e:
                            print(f"[Supervisor] Camera restart failed: {e}")
                    time.sleep(0.01)
                    continue
                else:
                    self._drop_frames = 0

                # ──── 2. PERCEPTION (Lane Detection) ────
                # --- FIX: Correct unpacking based on "expected 3, got 2" error ---
                # We assume self.perception.process_frame(frame) returns (error, hud).
                # If vision_score is needed, we calculate it now.
                error, hud = self.perception.process_frame(frame)
                vision_score = self._calculate_vision_score(error) # Calculate vision score based on error

                left_bgr, right_bgr = self.camera.get_side_frames()
                side_vis = side_preference_from_pair(left_bgr, right_bgr)
                
                # Update side clearance based on previous avoidance direction
                if self.avoid_direction > 0.1: # previously steering left
                    self.side_clearance = get_side_clearance_m(left_bgr)
                elif self.avoid_direction < -0.1: # previously steering right
                    self.side_clearance = get_side_clearance_m(right_bgr)
                else:
                    self.side_clearance = 2.0

                # ──── 3. SAFETY CHECK (LiDAR) ────
                lidar_clear, lidar_dist = self.safety.is_path_clear()
                depth_reading = self.depth.get_obstacle()

                # Early avoidance distance (use SAME processing as production code)
                raw_angles, raw_distances = self.safety.get_last_scan()
                tmp_angles, tmp_distances = self.reactive.process_lidar(raw_angles, raw_distances)
                avoid_min_dist = float(np.min(tmp_distances)) if (tmp_distances is not None and len(tmp_distances) > 0) else float("inf")
                avoid_points = int(len(tmp_distances)) if (tmp_distances is not None) else 0
                trigger_min_dist = float("inf")
                trigger_points = 0
                center_min_dist = float("inf")
                center_points = 0
                if (tmp_angles is not None) and (tmp_distances is not None) and (len(tmp_distances) > 0):
                    trigger_mask = np.abs(np.degrees(tmp_angles)) <= float(self.cfg.safety.avoid_trigger_arc_deg)
                    if np.any(trigger_mask):
                        trigger_d = np.asarray(tmp_distances)[trigger_mask]
                        trigger_points = int(trigger_d.size)
                        trigger_min_dist = float(np.min(trigger_d))
                    center_mask = np.abs(np.degrees(tmp_angles)) <= float(self.cfg.safety.lane_block_arc_deg)
                    if np.any(center_mask):
                        center_d = np.asarray(tmp_distances)[center_mask]
                        center_points = int(center_d.size)
                        center_min_dist = float(np.min(center_d))

                # Auto-correct LiDAR reverse if SafetyMonitor sees obstacle but reactive front-arc sees nothing.
                if (not lidar_clear) and (avoid_points < 3):
                    self.reactive_cfg.lidar_reverse = not bool(self.reactive_cfg.lidar_reverse)
                    self.reactive = ReactiveAvoidanceController(self.reactive_cfg)
                    tmp_angles, tmp_distances = self.reactive.process_lidar(raw_angles, raw_distances)
                    avoid_min_dist = float(np.min(tmp_distances)) if (tmp_distances is not None and len(tmp_distances) > 0) else float("inf")
                    avoid_points = int(len(tmp_distances)) if (tmp_distances is not None) else 0

                fused_clear, fused_dist = self._fuse_obstacles(
                    lidar_clear=lidar_clear,
                    lidar_dist=lidar_dist,
                    depth_reading=depth_reading,
                )

                # ──── 4. STATE MACHINE + CONTROL ────
                dt = max(time.time() - start_time, 0.01)
                self._update_state(
                    fused_clear,
                    fused_dist,
                    error,
                    vision_score,
                    dt,
                    hud,
                    depth_reading=depth_reading,
                    lidar_clear=lidar_clear,
                    avoid_min_dist=avoid_min_dist,
                    trigger_min_dist=trigger_min_dist,
                    trigger_points=trigger_points,
                    center_min_dist=center_min_dist,
                    center_points=center_points,
                    side_vis=side_vis,
                )

                # Always show obstacle debug
                cv2.putText(
                    hud,
                    f"min_front:{(avoid_min_dist if avoid_min_dist != float('inf') else -1):.2f}m pts:{avoid_points} rev:{int(self.reactive_cfg.lidar_reverse)}",
                    (10, hud.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    hud,
                    f"trig_min:{(trigger_min_dist if trigger_min_dist != float('inf') else -1):.2f} trig_pts:{trigger_points}",
                    (10, hud.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                cv2.putText(
                    hud,
                    f"center_min:{(center_min_dist if center_min_dist != float('inf') else -1):.2f} center_pts:{center_points}",
                    (10, hud.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                # ──── 5. FPS TRACKING ────
                self._update_fps()

                # ──── 6. DISPLAY HUD ────
                self._render_hud(hud)

                # ──── 7. INPUT HANDLING ────
                key = cv2.waitKey(1) & 0xFF
                self._handle_input(key)

        except KeyboardInterrupt:
            print("\n[Supervisor] Ctrl+C caught. Shutting down...")
        except Exception as e:
            print(f"\n[Supervisor] FATAL loop error: {e}")
            traceback.print_exc()
        finally:
            self._shutdown()

    # --- Re-entry logic and other methods (unchanged from previous update) ---

    def _fuse_obstacles(self, lidar_clear, lidar_dist, depth_reading):
        """
        LiDAR-primary fusion (Updated):
          - If LiDAR reports obstacle -> fused obstacle (LiDAR wins).
          - If LiDAR is clear, the path is considered CLEAR regardless of Depth
            to prevent false positives from stopping the vehicle.
          - RealSense depth is treated as an informative secondary signal but 
            does not trigger hard-stops if LiDAR confirms a clear path.
        """
        if not lidar_clear:
            # LiDAR detects obstacle -> return blocked
            self._depth_obstacle_frames = 0
            return False, float(lidar_dist)

        # Trust Depth if LiDAR is clear but Depth is confident and seeing something close
        if (not depth_reading.is_clear) and (depth_reading.confidence > 0.4):
            self._depth_obstacle_frames += 1
            if self._depth_obstacle_frames >= 3:
                return False, float(depth_reading.min_depth_m)
        else:
            self._depth_obstacle_frames = 0

        return True, -1.0

    def _update_state(
        self,
        fused_is_clear,
        fused_obstacle_dist,
        error,
        vision_score,
        dt,
        hud,
        depth_reading=None,
        lidar_clear=True,
        avoid_min_dist=float("inf"),
        trigger_min_dist=float("inf"),
        trigger_points=0,
        center_min_dist=float("inf"),
        center_points=0,
        side_vis=0.0,
    ):
        """
        Consolidated State Machine: Transitions + Actuation.
        States: IDLE, DRIVING, AVOIDING, SEARCHING.
        """
        h, w = hud.shape[:2]
        steering = 0.0
        throttle = 0.0
        status_info = ""

        # --- 1. GLOBAL SAFETY TRIGGERS ---
        # If we see something very close in center, force OBSTACLE_STOP regardless of current state.
        limit = float(self.cfg.safety.avoid_trigger_distance_m)
        should_emergency_stop = (
            (center_points >= 1 and center_min_dist < limit)
            or (not fused_is_clear and fused_obstacle_dist < limit)
        )
        if should_emergency_stop and self.state not in (self.STATE_OBSTACLE_STOP, self.STATE_AVOIDING, self.STATE_BACKING, self.STATE_IDLE):
            print(f"[Supervisor] ⚠️ EMERGENCY STOP: object at {min(center_min_dist, fused_obstacle_dist):.2f}m")
            self.state = self.STATE_OBSTACLE_STOP
            self._avoid_state_frames = 0
            self._avoid_direction_latch = 0

        # --- 2. STATE LOGIC ---
        if self.state == self.STATE_DRIVING:
            # A. Transitions
            should_avoid = (center_points >= self.cfg.safety.lane_block_points_min and center_min_dist <= float(self.cfg.safety.avoid_trigger_distance_m))
            if should_avoid or not fused_is_clear:
                self.state = self.STATE_OBSTACLE_STOP
                self._avoid_state_frames = 0
                self._avoid_direction_latch = 0
                print(f"[Supervisor] ⚠️ Obstacle detected ({fused_obstacle_dist:.2f}m). STOPPING to scan...")
            elif vision_score < self.cfg.safety.re_entry_vision_threshold:
                self.state = self.STATE_SEARCHING
                self._avoid_state_frames = 0
                print(f"[Supervisor] ⚠️ Lane lost (score={vision_score:.2f}). Searching...")
            else:
                # B. Actuation
                raw_steering = self.controller.compute(error, dt)
                steering = self.controller.saturate(raw_steering, self.cfg.control.max_steering, -self.cfg.control.max_steering)
                throttle = self.cfg.control.base_speed
                status_info = f"AUTO | spd:{throttle:.2f} str:{steering:+.2f} score:{vision_score:.2f}"

        elif self.state == self.STATE_OBSTACLE_STOP:
            self._avoid_state_frames += 1
            throttle = 0.0
            steering = 0.0
            status_info = f"SCANNING... ({self._avoid_state_frames}/30)"
            
            # Hazard lights / brakes
            if self._avoid_state_frames == 1:
                print(f"[Supervisor] ⚠️  OBSTACLE HALT! Distance: {fused_obstacle_dist:.2f}m")
                self.car.hazard_stop()
            
            # Use this time to decide direction based on LiDAR gaps
            angles, distances = self.safety.get_last_scan()
            adj_a, adj_d = self.reactive.process_lidar(angles, distances)
            best_angle, passable = self.reactive.find_best_gap(adj_a, adj_d)
            
            if best_angle is not None:
                # 0.5 rad threshold for "clear" side preference
                self._avoid_direction_latch = 1.0 if best_angle > 0 else -1.0
            else:
                # Fallback to side cameras if no clear LiDAR gap
                if side_vis > 0: self._avoid_direction_latch = 1.0 # Prefer Left
                else: self._avoid_direction_latch = -1.0 # Prefer Right
            
            if self._avoid_state_frames >= 30: # ~1s stop
                # Check final distance to decide if a backup is needed
                current_dist = fused_obstacle_dist if fused_obstacle_dist > 0 else 2.0
                if current_dist < 0.65:
                    print(f"[Supervisor] ⚠️ Too close ({current_dist:.2f}m). BACKING UP to create space.")
                    self.state = self.STATE_BACKING
                else:
                    print(f"[Supervisor] ✓ Scan complete. Best angle: {math.degrees(best_angle if best_angle else 0):.1f}°. Choosing {'LEFT' if self._avoid_direction_latch > 0 else 'RIGHT'} path.")
                    self.state = self.STATE_AVOIDING
                
                self._avoid_state_frames = 0
                self.avoid_direction = self._avoid_direction_latch

        elif self.state == self.STATE_BACKING:
            self._avoid_state_frames += 1
            throttle = -0.06 # Slow reverse
            steering = 0.0
            status_info = f"BACKING UP... ({self._avoid_state_frames}/15)"
            
            if self._avoid_state_frames >= 15: # 0.5s reverse
                print("[Supervisor] ✓ Space created. Now avoiding.")
                self.state = self.STATE_AVOIDING
                self._avoid_state_frames = 0

        elif self.state == self.STATE_AVOIDING:
            self._avoid_state_frames += 1
            
            # Use reactive controller for base commands
            angles, distances = self.safety.get_last_scan()
            adj_a, adj_d = self.reactive.process_lidar(angles, distances)
            react_th, react_st, react_stat = self.reactive.compute_control(adj_a, adj_d)
            arc_min = float(np.min(adj_d)) if (adj_d is not None and len(adj_d) > 0) else float("inf")
            
            # Actuation: Reactive + Side Clearance Nudge
            steering = 0.55 * self.avoid_direction # Aggressive initial turn
            
            # Side Nudge
            target_m = float(self.cfg.safety.side_clearance_target_m)
            if self.side_clearance < target_m:
                nudge = (target_m - self.side_clearance) * float(self.cfg.safety.side_clearance_gain)
                steering += np.clip(nudge, 0.0, 0.2) * self.avoid_direction
            
            # Safe Throttle Scaling
            prox_factor = np.clip((self.side_clearance - 0.2) / 0.8, 0.40, 1.0)
            throttle = 0.05 + (0.02 * prox_factor) # Range [0.05, 0.07] - slightly slower for safety
            
            # Transitions
            if fused_is_clear: self._fused_clear_frames += 1
            else: self._fused_clear_frames = 0
            
            can_search = (
                self._avoid_state_frames > 20
                and arc_min > 1.0
                and self._fused_clear_frames > 3
            )
            if can_search:
                print("[Supervisor] ✓ Way clear. Switching to SEARCHING.")
                self.state = self.STATE_SEARCHING
                self._avoid_state_frames = 0
            
            status_info = f"AVOID | dist:{arc_min:.2f}m side:{self.side_clearance:.2f}m"
            self._draw_gap_viz(hud, adj_a, adj_d)

        elif self.state == self.STATE_SEARCHING:
            self._avoid_state_frames += 1
            
            # Steering: steer back toward the center strongly
            base_search_steer = 0.25 if self._avoid_state_frames < 40 else 0.35
            
            # If we don't have a latch, use the last direction we were turning
            search_dir = self.avoid_direction
            if abs(search_dir) < 0.01:
                search_dir = -1.0 if self.last_steer > 0 else 1.0 

            # Vision Pull
            side_yellow = False
            left_bgr, right_bgr = self.camera.get_side_frames()
            if search_dir > 0: # Avoided Left, look Right
                side_yellow = is_side_yellow_visible(right_bgr, self.cfg.cv.hsv_lower, self.cfg.cv.hsv_upper)
            else:
                side_yellow = is_side_yellow_visible(left_bgr, self.cfg.cv.hsv_lower, self.cfg.cv.hsv_upper)
            
            v_pull = 0.20 if side_yellow else (0.12 if vision_score > 0.1 else 0.0)
            raw_steering = -(base_search_steer + v_pull) * search_dir
            steering = self.controller.saturate(raw_steering, self.cfg.control.max_steering, -self.cfg.control.max_steering)
            throttle = self.cfg.control.search_speed
            
            # Safety Timeout: if we search for too long (~6s @ 20fps), just stop.
            if self._avoid_state_frames > 120:
                print("[Supervisor] ⚠️ SEARCH TIMEOUT: Lane not found. Returning to IDLE for safety.")
                self.state = self.STATE_IDLE
                self.car.stop()
                return

            # Transitions
            if (not fused_is_clear) or (center_points >= self.cfg.safety.lane_block_points_min and center_min_dist < 0.8):
                self.state = self.STATE_AVOIDING
                self._avoid_state_frames = 0
            elif vision_score >= self.cfg.safety.re_entry_vision_threshold:
                print(f"[Supervisor] ✓ Lane found ({vision_score:.2f}). Resuming DRIVING.")
                self.state = self.STATE_DRIVING
                self._avoid_direction_latch = 0
            
            cv2.putText(
                hud,
                f"SEARCH | Thr:{throttle:.2f} Str:{steering:+.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        else:
            self.car.stop()
            status_info = "IDLE | Press 'a' to engage motors"

        # --- 3. FINAL ACTUATION ---
        # IMPORTANT: This block sends the power to the motors. MUST be present.
        if self.state != self.STATE_IDLE and not self.preview_mode:
            self.car.drive(float(throttle), float(steering))
        
        # --- 4. RENDER HUD ---
        color = (0, 255, 0) if self.state == self.STATE_DRIVING else (0, 255, 255)
        if self.state == self.STATE_AVOIDING: color = (0, 165, 255)
        # Shift text down to avoid overlap with FPS
        cv2.putText(hud, f"[{self.state}] {status_info}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _calculate_vision_score(self, error):
        """
        Calculates a confidence score based on lane detection quality.
        Needs tuning based on actual perception output.
        (MOCK version: assuming smaller errors mean higher confidence)
        """
        error_conf = float(np.clip(1.0 - abs(error) / 0.5, 0.0, 1.0))

        # Lane visibility heuristic from the lane detector pipeline.
        left_ok = getattr(self.perception, "last_left_found", False)
        right_ok = getattr(self.perception, "last_right_found", False)
        left_pts = getattr(self.perception, "last_left_points", 0)
        right_pts = getattr(self.perception, "last_right_points", 0)

        if left_ok and right_ok:
            lines_conf = 1.0
        elif left_ok or right_ok:
            lines_conf = 0.65
        else:
            lines_conf = 0.2

        points_conf = float(np.clip((left_pts + right_pts) / 300.0, 0.0, 1.0))
        score = 0.55 * lines_conf + 0.25 * points_conf + 0.20 * error_conf
        return float(np.clip(score, 0.0, 1.0))

    def _handle_input(self, key):
        if key == ord('q'): self.running = False
        elif key == ord('a'):
            if self.state == self.STATE_IDLE:
                print("[Supervisor] ✓ Autonomous driving ENGAGED!")
                self.state = self.STATE_DRIVING
                self.controller.reset_state()
        elif key == ord('s'):
            if self.state in (self.STATE_DRIVING, self.STATE_SEARCHING, self.STATE_AVOIDING, self.STATE_OBSTACLE):
                print("[Supervisor] ■ Manual stop. Returning to IDLE.")
                self.state = self.STATE_IDLE
                self.car.stop()

    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = time.time()

    def _render_hud(self, hud):
        cv2.putText(hud, f"FPS: {self._fps_display:.1f}",
                    (20, hud.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        state_colors = {
            self.STATE_IDLE: (200, 200, 200),
            self.STATE_DRIVING: (0, 255, 0),
            self.STATE_OBSTACLE_STOP: (0, 0, 255),
            self.STATE_AVOIDING: (0, 165, 255),
            self.STATE_SEARCHING: (255, 165, 0),
        }
        cv2.putText(hud, f"[{self.state}]", (hud.shape[1] - 250, hud.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors.get(self.state, (255, 255, 255)), 2)
        cv2.imshow("QCar2 Perception & Control", hud)

    def _draw_gap_viz(self, hud, angles, distances):
        """
        Visualizes the best gap on the HUD.
        """
        if angles is None or len(angles) < 2:
            return
            
        best_angle, passable = self.reactive.find_best_gap(angles, distances)
        if best_angle is not None:
            h, w = hud.shape[:2]
            center_x = w // 2
            # Map angle to pixel (rough approximation for HUD)
            # FOV is approx 120 deg total
            px_per_rad = w / math.radians(120)
            target_x = int(center_x + best_angle * px_per_rad)
            
            color = (0, 255, 0) if passable else (0, 0, 255)
            cv2.line(hud, (center_x, h-50), (target_x, h-100), color, 3)
            cv2.circle(hud, (target_x, h-100), 5, color, -1)
            cv2.putText(hud, f"GAP:{'OK' if passable else 'NO'}", (target_x - 30, h-110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _shutdown(self):
        print("\n[Supervisor] Shutting down all subsystems...")
        try: cv2.destroyAllWindows()
        except Exception: pass
        self.car.terminate()
        self.safety.terminate()
        self.depth.terminate()
        self.camera.terminate()
        print("[Supervisor] ✓ Shutdown complete. Goodbye!\n")

if __name__ == "__main__":
    is_preview = "--preview" in sys.argv
    app = Supervisor(preview_mode=is_preview)
    app.run()
