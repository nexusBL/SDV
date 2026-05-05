#!/usr/bin/env python3
"""
main.py - QCar2 Autonomous Lane Following & Obstacle Avoidance Orchestrator
============================================================================
This is the main entry point for the QCar2 lane following system.
It coordinates all subsystems: camera, LiDAR, perception, control, and actuation.

State Machine:
  IDLE ──[press 'a']──> DRIVING ──[obstacle]──> OBSTACLE_STOP
    ^                       |                         |
    |                       |                   [clear path]
    +──[press 's']──────────+                         |
                                                      v
                                                  AVOIDING (3-phase maneuver)
                                                  Phase 1: Back up
                                                  Phase 2: Steer right + drive
                                                  Phase 3: Straighten
                                                      |
                                                      v
                                                  RESUMING ──[N frames]──> DRIVING

Usage:
  python3 main.py              # Full autonomous mode (motors enabled)
  python3 main.py --preview    # Preview mode (motors disabled, vision only)
  python3 main.py --lidar-debug  # Print raw LiDAR scan data for angle calibration

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
import cv2
import traceback

from config import AppConfig
from hardware.camera_manager import CameraManager
from hardware.safety_monitor import SafetyMonitor
from hardware.car_controller import CarController
from perception.lane_cv import LaneDetector
from control.pid_controller import PIDController
from control.reactive_avoidance_controller import ReactiveController, ReactiveAvoidanceParams


class Supervisor:
    """
    Main application supervisor that orchestrates all QCar2 subsystems.

    Implements a clean state machine with explicit states:
    - IDLE:          Car is stationary, waiting for 'a' keypress
    - DRIVING:       Car is following the lane autonomously
    - OBSTACLE_STOP: Car has stopped due to LiDAR obstacle detection
    - AVOIDING:      Executing 3-phase avoidance maneuver around obstacle
    - RESUMING:      Avoidance done, waiting N frames before driving again

    The supervisor guarantees safe shutdown via try/finally blocks,
    ensuring motors are zeroed even if the script crashes.
    """

    # State constants
    STATE_IDLE          = "IDLE"
    STATE_DRIVING       = "DRIVING"
    STATE_OBSTACLE      = "OBSTACLE_STOP"
    STATE_AVOIDING      = "AVOIDING"
    STATE_RESUMING      = "RESUMING"

    # Avoidance sub-phases
    AVOID_PHASE_BACKUP      = "BACKUP"
    AVOID_PHASE_DECIDE      = "DECIDE"
    AVOID_PHASE_BYPASS      = "BYPASS"
    AVOID_PHASE_RESUME      = "RESUME_LANE"

    def __init__(self, preview_mode=False, lidar_debug=False):
        """
        Args:
            preview_mode: bool - if True, motors are disabled (vision-only mode)
            lidar_debug:  bool - if True, prints raw LiDAR scan data each frame
        """
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.lidar_debug = lidar_debug
        self.state = self.STATE_IDLE
        self.running = True

        # Resume countdown (frames to wait after avoidance completes)
        self.resume_counter = 0

        # Avoidance maneuver phase tracking
        self.avoid_phase = self.AVOID_PHASE_BACKUP
        self.avoid_phase_counter = 0
        self.avoid_direction = 1.0  # +1 for right, -1 for left

        # Reactive controller for dynamic bypass
        reactive_cfg = ReactiveAvoidanceParams(
            d_safe=self.cfg.safety.detection_range_m,
            d_stop=self.cfg.safety.stop_distance_m,
            car_width_m=self.cfg.safety.car_width_m,
            front_arc_deg=self.cfg.safety.roi_angle_deg
        )
        self.reactive = ReactiveController(reactive_cfg)

        # FPS tracking
        self._fps_counter = 0
        self._fps_start = time.time()
        self._fps_display = 0.0

        # Initialize subsystems
        print("\n" + "=" * 55)
        print("  🚗  QCar2 Autonomous Lane Following System  🚗")
        print("=" * 55)
        mode_str = "PREVIEW (motors disabled)" if preview_mode else "LIVE (motors enabled)"
        print(f"  Mode: {mode_str}")
        if lidar_debug:
            print("  LiDAR Debug: ON (raw scan data will be printed)")
        print("=" * 55 + "\n")

        print("[Supervisor] Initializing subsystem drivers...")
        self.camera     = CameraManager(self.cfg)
        self.safety     = SafetyMonitor(self.cfg, debug_mode=lidar_debug)
        self.car        = CarController(self.cfg)
        self.perception = LaneDetector(self.cfg)
        self.controller = PIDController(self.cfg)

    def run(self):
        """
        Main execution loop. Initializes hardware, runs the perception-control
        loop, and guarantees clean shutdown on exit.
        """
        # ── Phase 1: Hardware Initialization ──
        try:
            self.camera.initialize()
            self.safety.initialize()
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
                loop_start = time.time()

                # ──── 1. CAPTURE FRAME ────
                frame = self.camera.get_frame()
                if frame is None:
                    print("[Supervisor] Warning: Dropped frame.")
                    time.sleep(0.01)
                    continue

                # ──── 2. PERCEPTION (Lane Detection) ────
                error, hud = self.perception.process_frame(frame)

                # ──── 3. SAFETY CHECK (LiDAR) ────
                is_clear, obstacle_dist = self.safety.is_path_clear()

                # ──── 4. STATE MACHINE + CONTROL ────
                dt = max(time.time() - loop_start, 0.01)
                self._update_state(is_clear, obstacle_dist, error, dt, hud)

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

    def _update_state(self, is_clear, obstacle_dist, error, dt, hud):
        """
        State machine logic for driving, obstacle stop, avoidance, and resuming.
        """
        h, w = hud.shape[:2]
        
        # ── Visual Awareness: Show far obstacles on HUD ──
        if 0 < obstacle_dist < self.cfg.safety.detection_range_m:
            color = (0, 255, 255) if obstacle_dist > self.cfg.safety.stop_distance_m else (0, 0, 255)
            cv2.putText(hud, f"Obstacle: {obstacle_dist:.2f}m", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── OBSTACLE DETECTED: Emergency Stop only when near (5cm) ──
        # is_clear is False when obstacle_dist < stop_distance_m (0.05)
        if not is_clear and self.state not in (self.STATE_AVOIDING, self.STATE_RESUMING):
            if self.state != self.STATE_OBSTACLE:
                print(f"[Supervisor] ⚠️  STOPPING: Obstacle at critical {obstacle_dist:.2f}m!")
            self.state = self.STATE_OBSTACLE
            self.car.hazard_stop()
            self.controller.reset_state()
            cv2.putText(hud, "!!! STOP !!!", (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

        # ── OBSTACLE STOPPED -> Start Avoidance ──
        elif self.state == self.STATE_OBSTACLE:
            # We wait for the car to fully stop and for the user/system to initiate avoidance
            # In this version, we start automatically after a brief stop if path is still blocked
            # or if it's cleared. If it's cleared, we just resume. 
            # But the user wants "decide which side to go", so we start avoidance maneuver.
            print("[Supervisor] Initiating dynamic avoidance sequence...")
            self.state = self.STATE_AVOIDING
            self.avoid_phase = self.AVOID_PHASE_DECIDE
            self.controller.reset_state()

        # ── AVOIDING: Execute Dynamic Maneuver ──
        elif self.state == self.STATE_AVOIDING:
            self._execute_avoidance(hud, w, h)

        # ── RESUMING: Countdown Before Re-Engaging Lane Follow ──
        elif self.state == self.STATE_RESUMING:
            self.resume_counter -= 1
            self.car.stop()
            cv2.putText(hud, f"RESUMING IN {self.resume_counter}...",
                        (w // 2 - 150, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if self.resume_counter <= 0:
                print("[Supervisor] ✓ Lane re-acquired. Resuming autonomous driving.")
                self.state = self.STATE_DRIVING
                self.resume_counter = 0

        # ── DRIVING: Autonomous Lane Following ──
        elif self.state == self.STATE_DRIVING:
            if not self.preview_mode:
                steering = self.controller.compute(error, dt)
                self.car.drive(self.cfg.control.base_speed, steering)
                cv2.putText(hud, f"DRIVING [Error:{error:+.1f}]", (w // 2 - 100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.car.stop()
                cv2.putText(hud, "PREVIEW [MOTORS OFF]", (w // 2 - 100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── IDLE STATE ──
        else:
            self.car.stop()
            cv2.putText(hud, "IDLE [Press 'a' to drive]", (w // 2 - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def _execute_avoidance(self, hud, w, h):
        """
        Dynamic avoidance maneuver:
        1. DECIDE: Use LiDAR to find best gap (Left/Right).
        2. BACKUP: Reverse slightly to create space.
        3. BYPASS: Drive around maintaining 4cm gap.
        4. RESUME: Use side cameras to find yellow line.
        """
        avc = self.cfg.avoidance
        saf = self.cfg.safety

        if self.avoid_phase == self.AVOID_PHASE_DECIDE:
            # Get LiDAR scan
            angles, distances = self.safety.get_last_scan()
            adj_angles, adj_dists = self.reactive.process_lidar(angles, distances)
            
            # Find best gap
            gap_angle, passable = self.reactive.find_best_gap(adj_angles, adj_dists)
            
            if passable:
                self.avoid_direction = 1.0 if gap_angle < 0 else -1.0
                print(f"[Supervisor] Decision: Gap found at {math.degrees(gap_angle):.1f}°, steering {'RIGHT' if self.avoid_direction > 0 else 'LEFT'}")
            else:
                self.avoid_direction = 1.0 # Default right
                print("[Supervisor] Decision: No clear gap, defaulting RIGHT")
                
            self.avoid_phase = self.AVOID_PHASE_BACKUP
            self.avoid_phase_counter = avc.backup_duration_frames

        elif self.avoid_phase == self.AVOID_PHASE_BACKUP:
            label = f"AVOID: Backing up... ({self.avoid_phase_counter})"
            self.car.reverse(avc.backup_throttle)
            cv2.putText(hud, label, (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
            self.avoid_phase_counter -= 1
            if self.avoid_phase_counter <= 0:
                print("[Supervisor] Backup complete. Starting bypass maneuver.")
                self.avoid_phase = self.AVOID_PHASE_BYPASS

        elif self.avoid_phase == self.AVOID_PHASE_BYPASS:
            # Dynamic bypass using LiDAR feedback for 4cm gap
            angles, distances = self.safety.get_last_scan()
            
            # Look at a side-arc (e.g. 40° to 120° on the side we chose)
            side_arc_center = 90.0 * self.avoid_direction
            # On QCar2, 180 is front, so side is 180 +/- 90 -> 90 or 270
            # Let's use the centered angles from reactive.process_lidar
            adj_angles, adj_dists = self.reactive.process_lidar(angles, distances)
            
            # Filter for side obstacles
            side_mask = (np.abs(adj_angles - math.radians(side_arc_center)) < math.radians(45))
            side_dists = adj_dists[side_mask]
            
            if side_dists.size > 0:
                current_gap = np.min(side_dists)
                # P-controller for gap
                gap_error = current_gap - saf.side_clearance_target_m # Target is 0.04m
                # If gap too small (error < 0), steer away. 
                # If choosing RIGHT, obstacle is on LEFT? No, if steer RIGHT, obstacle is on LEFT.
                # If avoid_direction is +1 (Steer Right), obstacle is on the Left (side_arc_center = -90).
                # Wait, if avoid_direction was meant for steering:
                # gap_angle < 0 (Right) -> avoid_direction = 1.0 (Right) -> Obstacle on Left (-90).
                
                steer_corr = -gap_error * saf.side_clearance_gain * self.avoid_direction
                steering = np.clip(0.3 * self.avoid_direction + steer_corr, -0.5, 0.5)
            else:
                # No side obstacle seen, curve inwards to find lane
                steering = -0.2 * self.avoid_direction 
            
            self.car.avoid(avc.avoidance_throttle, steering)
            cv2.putText(hud, f"AVOID: Bypass [Gap:{current_gap if 'current_gap' in locals() else 0:.3f}m]", 
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)

            # Check side cameras for yellow line
            left_img, right_img = self.camera.get_side_frames()
            target_img = right_img if self.avoid_direction > 0 else left_img
            found, offset = self.perception.detect_yellow_lane(target_img)
            
            if found:
                print(f"[Supervisor] Side camera DETECTED yellow lane! Rerouting...")
                self.avoid_phase = self.AVOID_PHASE_RESUME
                self.avoid_phase_counter = 20 # frames to straighten

        elif self.avoid_phase == self.AVOID_PHASE_RESUME:
            # Final alignment towards the detected lane
            self.car.avoid(avc.avoidance_throttle, -0.15 * self.avoid_direction)
            cv2.putText(hud, "AVOID: Re-aligning with lane...", (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 3)
            self.avoid_phase_counter -= 1
            if self.avoid_phase_counter <= 0:
                self.state = self.STATE_RESUMING
                self.resume_counter = saf.resume_delay_frames
                self.state = self.STATE_RESUMING
                self.resume_counter = self.cfg.safety.resume_delay_frames

    def _handle_input(self, key):
        """Processes keyboard input for state transitions."""
        if key == ord('q'):
            print("\n[Supervisor] 'q' pressed. Initiating shutdown...")
            self.running = False

        elif key == ord('a'):
            if self.state == self.STATE_IDLE:
                print("[Supervisor] ✓ Autonomous driving ENGAGED!")
                self.state = self.STATE_DRIVING
                self.controller.reset_state()

        elif key == ord('s'):
            if self.state in (self.STATE_DRIVING, self.STATE_AVOIDING, self.STATE_RESUMING):
                print("[Supervisor] ■ Manual stop. Returning to IDLE.")
                self.state = self.STATE_IDLE
                self.car.stop()

    def _update_fps(self):
        """Tracks frames per second over a rolling 1-second window."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = time.time()

    def _render_hud(self, hud):
        """Displays the HUD frame with FPS overlay."""
        cv2.putText(hud, f"FPS: {self._fps_display:.1f}",
                    (20, hud.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # State indicator at bottom-right
        state_colors = {
            self.STATE_IDLE:      (200, 200, 200),
            self.STATE_DRIVING:   (0, 255, 0),
            self.STATE_OBSTACLE:  (0, 0, 255),
            self.STATE_AVOIDING:  (0, 165, 255),
            self.STATE_RESUMING:  (0, 255, 255),
        }
        cv2.putText(hud, f"[{self.state}]",
                    (hud.shape[1] - 250, hud.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    state_colors.get(self.state, (255, 255, 255)), 2)

        cv2.imshow("QCar2 Perception & Control", hud)

    def _shutdown(self):
        """
        Guarantees clean shutdown of all hardware subsystems.
        Called in the finally block - always executes even on crash.
        """
        print("\n[Supervisor] Shutting down all subsystems...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Order matters: stop car first, then sensors
        self.car.terminate()
        self.safety.terminate()
        self.camera.terminate()
        print("[Supervisor] ✓ Shutdown complete. Goodbye!\n")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    is_preview    = "--preview" in sys.argv
    is_lidar_dbg  = "--lidar-debug" in sys.argv
    app = Supervisor(preview_mode=is_preview, lidar_debug=is_lidar_dbg)
    app.run()
