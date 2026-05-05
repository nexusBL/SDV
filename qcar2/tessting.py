#!/usr/bin/env python3
"""
tessting.py - QCar2 Autonomous Lane Following & Obstacle Avoidance Orchestrator
============================================================================
Integrated Version: Stable Backup + Optimized Maneuver (Backup -> Rotate -> Bypass)
User Requirements: 35cm stop, 60s thinking, 3cm side gap, 45° rotation.
"""

import os
import sys

# ── CRITICAL: Fix nvarguscamerasrc EGL authorization ──
_saved_display = os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

import time
import math
import cv2
import traceback
import numpy as np

from config import AppConfig
from hardware.camera_manager import CameraManager
from hardware.safety_monitor import SafetyMonitor
from hardware.depth_monitor import DepthMonitor
from hardware.car_controller import CarController
from perception.lane_cv import LaneDetector
from perception.side_clearance import side_preference_from_pair, get_side_clearance_m, is_side_yellow_visible
from control.pid_controller import PIDController

# Reactive controller from parent scripts folder
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)
from control.reactive_avoidance_controller import ReactiveController as ReactiveAvoidanceController
from control.reactive_avoidance_controller import ReactiveAvoidanceParams

class Supervisor:
    """Main application supervisor that orchestrates all QCar2 subsystems."""

    # --- State constants ---
    STATE_IDLE          = "IDLE"
    STATE_DRIVING       = "DRIVING"
    STATE_OBSTACLE_STOP = "OBSTACLE_STOP"
    STATE_BACKING       = "BACKING"
    STATE_ROTATE        = "ROTATE"
    STATE_AVOIDING      = "AVOIDING"
    STATE_SEARCHING     = "SEARCHING"

    def __init__(self, preview_mode=False):
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.state = self.STATE_IDLE
        self.running = True

        self._fps_counter = 0
        self._fps_start = time.time()
        self._fps_display = 0.0

        self._depth_obstacle_frames = 0
        self._fused_clear_frames = 0
        self._drop_frames = 0
        self._avoid_state_frames = 0

        print("\n" + "=" * 50)
        print("  🚗  QCar2 Autonomous Lane Following System  🚗")
        print("=" * 50)
        mode_str = "PREVIEW (motors disabled)" if preview_mode else "LIVE (motors enabled)"
        print(f"  Mode: {mode_str}")
        print("=" * 50 + "\n")

        print("[Supervisor] Initializing subsystem drivers...")
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

        # --- FIX: Initial LiDAR reverse sync ---
        if getattr(self.reactive_cfg, 'lidar_reverse', False):
             self.safety.cfg.lidar_front_angle_deg = 0.0
        else:
             self.safety.cfg.lidar_front_angle_deg = 180.0

        self.last_steer = 0.0
        self.avoid_direction = 0.0
        self.side_clearance = 2.0
        self._scan_start_time = 0.0
        self.last_depth = None # Cache for HUD

    def run(self):
        try:
            self.camera.initialize()
            self.safety.initialize()
            self.depth.initialize()
            self.car.initialize()
        except Exception as e:
            print(f"[FATAL] Hardware initialization failed: {e}")
            traceback.print_exc()
            self._shutdown()
            return

        if _saved_display: os.environ["DISPLAY"] = _saved_display
        print("\n[Supervisor] All systems nominal. Entering main loop.")
        print("[Supervisor] [a]=Start  [s]=Stop  [q]=Quit\n")

        try:
            while self.running:
                start_time = time.time()
                self.car.read()
                # ──── 1. CAPTURE FRAME ────
                frame = self.camera.get_frame()
                if frame is None:
                    self._drop_frames += 1
                    if self._drop_frames % 10 == 0:
                        print(f"[Supervisor] Warning: {self._drop_frames} dropped frames.")

                    # Always show a window so it doesn't look "stuck"
                    hud = np.zeros((self.cfg.camera.capture_height, self.cfg.camera.capture_width, 3), dtype=np.uint8)
                    cv2.putText(hud, f"NO CAMERA FRAME (drops:{self._drop_frames})", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(hud, "Attempting Camera Restart...", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    self._update_fps()
                    self._render_hud(hud)
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Sustained drops - try restarting the camera
                    if self._drop_frames >= 40:
                        print("[Supervisor] Sustained drops. Restarting camera...")
                        try:
                            self.camera.terminate()
                            time.sleep(1.0)
                            self.camera.initialize()
                            self._drop_frames = 0
                        except Exception as e:
                            print(f"[Supervisor] Camera restart failed: {e}")
                    
                    time.sleep(0.01)
                    continue
                else:
                    self._drop_frames = 0

                error, hud = self.perception.process_frame(frame)
                vision_score = self._calculate_vision_score(error)

                left_bgr, right_bgr = self.camera.get_side_frames()
                side_vis = side_preference_from_pair(left_bgr, right_bgr)
                
                # LiDAR points
                raw_angles, raw_distances = self.safety.get_last_scan()
                tmp_angles, tmp_distances = self.reactive.process_lidar(raw_angles, raw_distances)
                avoid_min_dist = float(np.min(tmp_distances)) if (tmp_distances is not None and len(tmp_distances) > 0) else float("inf")
                avoid_points = int(len(tmp_distances)) if (tmp_distances is not None) else 0
                
                center_min_dist = float("inf")
                center_points = 0
                if (tmp_angles is not None) and (tmp_distances is not None) and (len(tmp_distances) > 0):
                    center_mask = np.abs(np.degrees(tmp_angles)) <= float(self.cfg.safety.lane_block_arc_deg)
                    if np.any(center_mask):
                        center_d = np.asarray(tmp_distances)[center_mask]
                        center_points = int(center_d.size)
                        center_min_dist = float(np.min(center_d))

                # LiDAR orientation sync
                lidar_clear, lidar_dist = self.safety.is_path_clear()

                depth_reading = self.depth.get_obstacle()
                self.last_depth = depth_reading # Store for HUD
                fused_clear, fused_dist = self._fuse_obstacles(lidar_clear, lidar_dist, depth_reading)

                dt = max(time.time() - start_time, 0.01)
                self._update_state(fused_clear, fused_dist, error, vision_score, dt, hud, 
                                 depth_reading, center_min_dist, center_points, side_vis, left_bgr, right_bgr)

                # Obstacle HUD
                cv2.putText(hud, f"min_front:{(avoid_min_dist if avoid_min_dist != float('inf') else -1):.2f}m pts:{avoid_points} rev:{int(self.reactive_cfg.lidar_reverse)}",
                           (10, hud.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(hud, f"center_min:{(center_min_dist if center_min_dist != float('inf') else -1):.2f} center_pts:{center_points}",
                           (10, hud.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(hud, f"depth_min:{(depth_reading.min_depth_m if depth_reading.min_depth_m > 0 else -1):.2f}m conf:{depth_reading.confidence:.2f}",
                           (10, hud.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

                self._update_fps()
                self._render_hud(hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): self.running = False
                elif key == ord('a'):
                    if self.state == self.STATE_IDLE:
                        print("[Supervisor] Autonomous driving ENGAGED!")
                        self.state = self.STATE_DRIVING
                        self.controller.reset_state()
                elif key == ord('s'):
                    print("[Supervisor] Manual stop.")
                    self.state = self.STATE_IDLE
                    self.car.stop()

        except KeyboardInterrupt: pass
        except Exception as e:
            traceback.print_exc()
        finally: self._shutdown()

    def _fuse_obstacles(self, lidar_clear, lidar_dist, depth_reading):
        """LiDAR + Depth Fusion (Balanced)."""
        lidar_d = float(lidar_dist) if lidar_dist != float('inf') else float('inf')
        if not lidar_clear:
            self._depth_obstacle_frames = 0
            return False, lidar_d

        depth_limit = 0.6
        if (not depth_reading.is_clear) and (depth_reading.min_depth_m < depth_limit) and (depth_reading.confidence > 0.4):
            self._depth_obstacle_frames += 1
            if self._depth_obstacle_frames >= 2: return False, float(depth_reading.min_depth_m)
        else: self._depth_obstacle_frames = 0
        return True, lidar_d

    def _update_state(self, fused_is_clear, fused_obstacle_dist, error, vision_score, dt, hud, 
                      depth_reading, center_min_dist, center_points, side_vis, left_bgr, right_bgr):
        h, w = hud.shape[:2]
        steering = 0.0
        throttle = 0.0
        status_info = ""

        limit = float(self.cfg.safety.avoid_trigger_distance_cm) / 100.0 # 0.35m

        # Emergency Stop
        should_stop = (
            (center_points >= 1 and center_min_dist <= limit)
            or (not fused_is_clear and fused_obstacle_dist <= limit)
            or (not depth_reading.is_clear and depth_reading.min_depth_m <= limit and depth_reading.confidence > self.cfg.safety.depth_min_confidence)
        )
        if should_stop and self.state not in (self.STATE_OBSTACLE_STOP, self.STATE_AVOIDING, self.STATE_BACKING, self.STATE_ROTATE, self.STATE_IDLE):
            print("[Supervisor] ⚠️ EMERGENCY STOP!")
            self.state = self.STATE_OBSTACLE_STOP
            self._avoid_state_frames = 0
            self._scan_start_time = time.time()

        if self.state == self.STATE_DRIVING:
            raw_steering = self.controller.compute(error, dt)
            steering = self.controller.saturate(raw_steering, self.cfg.control.max_steering, -self.cfg.control.max_steering)
            # Slow down if near obstacle
            throttle = self.cfg.control.crawl_speed if (fused_obstacle_dist < 0.6 and fused_obstacle_dist > 0.0) else self.cfg.control.base_speed
            status_info = f"AUTO | spd:{throttle:.2f} str:{steering:+.2f} score:{vision_score:.2f}"

        elif self.state == self.STATE_OBSTACLE_STOP:
            self._avoid_state_frames += 1
            elapsed = time.time() - self._scan_start_time
            scan_duration = 60.0 # 1 minute
            status_info = f"THINKING... ({elapsed:.1f}/{scan_duration}s)"
            if self._avoid_state_frames == 1: self.car.hazard_stop()
            
            # Decision
            if side_vis > 0: self.avoid_direction = 1.0 # Prefer Left
            else: self.avoid_direction = -1.0 # Prefer Right
            
            if elapsed >= scan_duration:
                self.state = self.STATE_BACKING
                self._avoid_state_frames = 0

        elif self.state == self.STATE_BACKING:
            self._avoid_state_frames += 1
            throttle = float(self.cfg.avoidance.backup_throttle)
            status_info = f"BACKING UP... ({self._avoid_state_frames}/{self.cfg.avoidance.backup_duration_frames})"
            if self._avoid_state_frames >= self.cfg.avoidance.backup_duration_frames:
                self.state = self.STATE_ROTATE
                self._avoid_state_frames = 0

        elif self.state == self.STATE_ROTATE:
            self._avoid_state_frames += 1
            steering = float(self.cfg.avoidance.rotate_steering) * (-self.avoid_direction)
            throttle = float(self.cfg.avoidance.rotate_throttle)
            status_info = f"ROTATING 45° ({self._avoid_state_frames}/{self.cfg.avoidance.rotate_duration_frames})"
            if self._avoid_state_frames >= self.cfg.avoidance.rotate_duration_frames:
                self.state = self.STATE_AVOIDING
                self._avoid_state_frames = 0

        elif self.state == self.STATE_AVOIDING:
            self._avoid_state_frames += 1
            steering = -0.35 * self.avoid_direction
            
            # Nudge logic (3cm goal) using LiDAR
            raw_angles, raw_distances = self.safety.get_last_scan()
            if raw_angles is not None and len(raw_angles) > 0:
                angle_deg = (np.degrees(raw_angles) % 360.0).astype(float)
                angular_diff = (angle_deg - float(self.cfg.safety.lidar_front_angle_deg) + 180.0) % 360.0 - 180.0
                
                if self.avoid_direction > 0:  # Prefer Left -> Obstacle is on Right
                    side_mask = (angular_diff < -20.0) & (angular_diff > -160.0)
                else:  # Prefer Right -> Obstacle is on Left
                    side_mask = (angular_diff > 20.0) & (angular_diff < 160.0)
                    
                side_dists = raw_distances[side_mask & (raw_distances > 0.05) & (raw_distances < 3.5)]
                if len(side_dists) > 0:
                    self.side_clearance = float(np.min(side_dists))
                else:
                    self.side_clearance = 2.0  # Clear
            else:
                self.side_clearance = 2.0
            
            target_m = float(self.cfg.safety.side_clearance_target_cm) / 100.0 # 0.03m
            if self.side_clearance < target_m:
                steering += (target_m - self.side_clearance) * float(self.cfg.safety.side_clearance_gain) * self.avoid_direction
            
            throttle = float(self.cfg.avoidance.avoidance_throttle)
            status_info = f"AVOID | side:{self.side_clearance:.3f}m"
            
            # Transition via Side CSI
            target_bgr = right_bgr if self.avoid_direction > 0 else left_bgr
            found_lane, _ = self.perception.detect_yellow_lane(target_bgr)
            if found_lane and self._avoid_state_frames > 20:
                print("[Supervisor] ✓ Side YELLOW detected! Switching to SEARCHING.")
                self.state = self.STATE_SEARCHING
                self._avoid_state_frames = 0

        elif self.state == self.STATE_SEARCHING:
            self._avoid_state_frames += 1
            steering = 0.3 * (-self.avoid_direction)
            throttle = self.cfg.control.search_speed
            if vision_score >= self.cfg.safety.re_entry_vision_threshold:
                print("[Supervisor] ✓ Lane found. Resuming DRIVING.")
                self.state = self.STATE_DRIVING

        if self.state != self.STATE_IDLE and not self.preview_mode:
            self.car.drive(float(throttle), float(steering))
        
        color = (0, 255, 0) if self.state == self.STATE_DRIVING else (0, 255, 255)
        cv2.putText(hud, f"[{self.state}] {status_info}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _calculate_vision_score(self, error):
        l_ok = getattr(self.perception, "last_left_found", False)
        r_ok = getattr(self.perception, "last_right_found", False)
        score = 0.8 if (l_ok and r_ok) else (0.4 if (l_ok or r_ok) else 0.1)
        return score

    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = time.time()

    def _render_hud(self, hud):
        h, w = hud.shape[:2]
        cv2.putText(hud, f"FPS: {self._fps_display:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add a premium state indicator
        state_colors = {
            self.STATE_IDLE: (150, 150, 150),
            self.STATE_DRIVING: (0, 255, 0),
            self.STATE_OBSTACLE_STOP: (0, 0, 255),
            self.STATE_AVOIDING: (0, 165, 255),
            self.STATE_SEARCHING: (0, 255, 255),
        }
        color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(hud, f"[{self.state}]", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # --- RealSense IR Grayscale Overlay ---
        if self.last_depth is not None and self.last_depth.ir_frame is not None:
            # Apply grayscale processing for reflection mitigation to ensure better understanding
            ir_gray = self.last_depth.ir_frame
            # Enhance contrast without affecting standard CSI logic
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            ir_gray_enhanced = clahe.apply(ir_gray)
            
            ir_bgr = cv2.cvtColor(ir_gray_enhanced, cv2.COLOR_GRAY2BGR)
            # Resize for overlay (e.g. 1/4 size)
            ow, oh = w // 4, h // 4
            overlay = cv2.resize(ir_bgr, (ow, oh))
            
            # Place in top-right
            x0, y0 = w - ow - 10, 10
            # Draw border
            cv2.rectangle(hud, (x0-2, y0-2), (x0+ow+2, y0+oh+2), (0, 255, 255), 2)
            hud[y0:y0+oh, x0:x0+ow] = overlay
            
            # Label
            cv2.putText(hud, "RealSense IR (Gray)", (x0, y0 + oh + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Distance on IR
            if self.last_depth.min_depth_m > 0:
                dist_str = f"{self.last_depth.min_depth_m:.2f}m"
                cv2.putText(hud, dist_str, (x0 + 5, y0 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("QCar2 Perception & Control", hud)

    def _shutdown(self):
        print("\n[Supervisor] Shutting down...")
        try: cv2.destroyAllWindows()
        except: pass
        self.car.terminate()
        self.safety.terminate()
        self.depth.terminate()
        self.camera.terminate()

if __name__ == "__main__":
    app = Supervisor(preview_mode="--preview" in sys.argv)
    app.run()
