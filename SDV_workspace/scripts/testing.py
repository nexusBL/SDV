#!/usr/bin/env python3
"""
main.py - QCar2 Autonomous Lane Following & Obstacle Avoidance Orchestrator
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
  python3 main.py              # Full autonomous mode (motors enabled)
  python3 main.py --preview    # Preview mode (motors disabled, vision only)

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
import numpy as np

# --- Re-import necessary modules here if they belong to separate files ---
# (Assuming config, hardware, perception, control are external modules as per original prompt)
# You need to ensure the following modules are correctly defined in your file structure.
# For this code to run, you will need to ensure:
# from config import AppConfig
# from hardware.camera_manager import CameraManager
# from hardware.safety_monitor import SafetyMonitor
# from hardware.car_controller import CarController
# from perception.lane_cv import LaneDetector
# from control.pid_controller import PIDController

# --- MOCK CLASSES (Replace with real imports) ---
# --- For demonstration purposes, assume mock classes for external hardware/logic.
class AppConfig:
    class safety:
        resume_delay_frames = 10 # Countdown frames after obstacle clears
        obstacle_dist_m = 1.0  # Distance threshold for LiDAR obstacle detection
        re_entry_vision_threshold = 0.6 # Minimum vision score to re-enter DRIVING state from SEARCHING

    class control:
        base_speed = 0.1 # Base speed for DRIVING state
        search_speed = 0.05 # Slower speed during SEARCHING state
        PID_KP = 0.5 # Example value, tune this in config.py
        PID_KI = 0.01
        PID_KD = 0.1
        PID_INTEGRAL_MAX = 1.0

    class lane_detection:
        canny_thresholds = (70, 200) # Canny tuning for reflections
        hsv_yellow_range_lower = np.array([15, 100, 100])
        hsv_yellow_range_upper = np.array([40, 255, 255])

class CameraManager:
    def __init__(self, cfg): pass
    def initialize(self): print("[Camera] Initializing...")
    def get_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8) # MOCK data
    def terminate(self): pass

# --- SafetyMonitor Mock (Assuming LiDAR logic) ---
class SafetyMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
    def initialize(self): print("[Safety] Initializing...")
    def is_path_clear(self):
        # MOCK clear path behavior. For real implementation, replace with LiDAR logic.
        return True, float('inf')
    def terminate(self): pass

# --- CarController Mock ---
class CarController:
    def __init__(self, cfg): pass
    def initialize(self): print("[Car] Initializing...")
    def drive(self, speed, steering): pass # MOCK drive command
    def stop(self): pass # MOCK stop command
    def hazard_stop(self): pass # MOCK emergency stop
    def terminate(self): pass

# --- PIDController Mock ---
class PIDController:
    def __init__(self, cfg):
        self.kp = cfg.control.PID_KP
        self.ki = cfg.control.PID_KI
        self.kd = cfg.control.PID_KD
        self.integral_max = cfg.control.PID_INTEGRAL_MAX
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_time = None
    def compute(self, error, dt):
        self._prev_time = time.time() if self._prev_time is None else self._prev_time
        self._integral = np.clip(self._integral + error * dt, -self.integral_max, self.integral_max)
        derivative = (error - self._prev_err) / dt if dt > 0 else 0
        self._prev_err = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative
    def reset_state(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_time = None

# --- LaneDetector Mock (For simulation purposes) ---
class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg
    def process_frame(self, frame):
        # MOCK LOGIC for lane detection (replace with your real code from previous iterations)
        # Assuming the external LaneDetector module returns these three values: error, vision_score, and hud.
        # For this example, let's return a simulated error and vision score.
        mock_error = 0.1 * np.random.randn() # Simulate small, random lane deviation
        mock_vision_score = 0.8 # Assume high initial confidence

        hud = frame.copy() # Return the image frame itself for HUD display

        return mock_error, mock_vision_score, hud

# ══════════════════════════════════════════════════════════════════
# Supervisor Code (Re-implemented with Re-entry Logic)
# ══════════════════════════════════════════════════════════════════

class Supervisor:
    """
    Main application supervisor that orchestrates all QCar2 subsystems.

    Implements a clean state machine with explicit states:
    - IDLE:          Car is stationary, waiting for 'a' keypress
    - DRIVING:       Car is following the lane autonomously.
    - OBSTACLE_STOP: Car has stopped due to LiDAR obstacle detection.
    - SEARCHING:     Car is moving slowly, looking for the lane again.
    - LOST_LANE:     Lane lost for too long, requires re-acquisition or manual intervention.
    """

    # --- State constants (Updated) ---
    STATE_IDLE = "IDLE"
    STATE_DRIVING = "DRIVING"
    STATE_OBSTACLE = "OBSTACLE_STOP"
    STATE_SEARCHING = "SEARCHING" # New state for lane re-acquisition

    def __init__(self, preview_mode=False):
        """
        Args:
            preview_mode: bool - if True, motors are disabled (vision-only mode)
        """
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.state = self.STATE_IDLE
        self.running = True

        # FPS tracking
        self._fps_counter = 0
        self._fps_start = time.time()
        self._fps_display = 0.0

        print("\n" + "=" * 50)
        print("  🚗  QCar2 Autonomous Lane Following System  🚗")
        print("=" * 50)
        mode_str = "PREVIEW (motors disabled)" if preview_mode else "LIVE (motors enabled)"
        print(f"  Mode: {mode_str}")
        print("=" * 50 + "\n")

        # --- Subsystem initialization ---
        # Note: The actual initialization calls are in run() for a more robust startup sequence.
        # This allows for warm-up and retries before entering main loop.
        self.car = CarController(self.cfg)
        self.perception = LaneDetector(self.cfg)
        self.controller = PIDController(self.cfg)
        self.camera = CameraManager(self.cfg)
        self.safety = SafetyMonitor(self.cfg)

        self.last_steer = 0.0 # Store steering for re-entry

    def run(self):
        """
        Main execution loop. Initializes hardware, runs the perception-control
        loop, and guarantees clean shutdown on exit.
        """
        # ── Phase 1: Hardware Initialization and Warm-up ──
        print("[Supervisor] Initializing subsystem drivers...")
        try:
            self.camera.initialize()
            self.safety.initialize()
            self.car.initialize()
        except Exception as e:
            print(f"\n[FATAL] Hardware initialization failed: {e}")
            traceback.print_exc()
            self._shutdown()
            return

        # --- [UPDATED] Camera Warm-up Logic ---
        # Wait for camera to produce valid frames (addresses initial black screen issue)
        print("[Supervisor] Warming up camera...")
        # (Assuming CameraManager.get_frame returns None on failure or all zeros)
        for i in range(100): # Attempt up to 100 times to get a valid frame
            frame = self.camera.get_frame()
            if frame is not None and np.max(frame) > 10: # Check if frame contains data (not all black)
                print(f"[Supervisor] ✓ Camera warmed up successfully after {i+1} attempts.")
                break
            else:
                time.sleep(0.05) # Delay before next retry
        else:
            print("[Supervisor] FATAL: Camera still blank after several retries. Check connections.")
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
                # --- FIX: Ensure correct return values and assignment ---
                # The `process_frame` function should return error (offset), vision_score, and hud.
                # (Assuming the external LaneDetector module returns these three values)
                error, vision_score, hud = self.perception.process_frame(frame)

                # ──── 3. SAFETY CHECK (LiDAR) ────
                is_clear, obstacle_dist = self.safety.is_path_clear()

                # ──── 4. STATE MACHINE + CONTROL ────
                dt = max(time.time() - loop_start, 0.01)
                self._update_state(is_clear, obstacle_dist, error, vision_score, dt, hud)

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

    def _update_state(self, is_clear, obstacle_dist, error, vision_score, dt, hud):
        """
        State machine logic for driving, obstacle stop, and searching/re-entry.
        """
        h, w = hud.shape[:2]
        steering = 0.0
        speed = 0.0

        # --- State Machine Transitions ---
        if self.state == self.STATE_OBSTACLE:
            # ── From OBSTACLE_STOP to SEARCHING (If path clears) ──
            if is_clear and self.re_entry_timeout_counter <= 0:
                print(f"[Supervisor] ✓ Path clear at {obstacle_dist:.2f}m. Transitioning to SEARCHING.")
                self.state = self.STATE_SEARCHING
                self.car.stop() # Ensure car stops briefly before starting search maneuver

            cv2.putText(hud, f"!!! OBSTACLE {obstacle_dist:.2f}m !!!", (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        elif self.state == self.STATE_DRIVING or self.state == self.STATE_SEARCHING:
            # ── Obstacle Detection Check ──
            if not is_clear and self.state == self.STATE_DRIVING:
                # Obstacle detected only when in DRIVING state. Avoid re-triggering from SEARCHING.
                print(f"[Supervisor] ⚠️  OBSTACLE detected at {obstacle_dist:.2f}m! Stopping.")
                self.state = self.STATE_OBSTACLE
                self.car.hazard_stop()
                self.controller.reset_state()

        # --- Implement state behaviors ---
        if self.state == self.STATE_SEARCHING:
            # ── SEARCHING State: Lane Re-acquisition Logic (CRITICAL FIX) ──
            # Problem: Car avoids obstacle and keeps driving off course.
            # Fix: Car moves slowly and looks for lane detection before re-entering DRIVING.
            if vision_score >= self.cfg.safety.re_entry_vision_threshold:
                # Lane re-acquired successfully. Continue driving.
                self.state = self.STATE_DRIVING
                print(f"[Supervisor] ✓ Lane re-acquired ({vision_score:.2f}). Resuming driving.")
            else:
                # Lane not yet re-acquired. Re-center slowly or perform search maneuver.
                cv2.putText(hud, f"SEARCHING ({vision_score:.2f})",
                            (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                steering = self.last_steer * 0.5 # Maintain light steering bias during search
                speed = self.cfg.control.search_speed # Slow speed during re-entry search
                if not self.preview_mode:
                    self.car.drive(speed, steering)

        elif self.state == self.STATE_DRIVING:
            # ── AUTONOMOUS DRIVING State ──
            # Lane is present, drive according to PID controller.
            steering = self.controller.compute(error, dt)

            # --- [UPDATED] Check vision score to handle temporary lane loss (Issue 4) ---
            if vision_score < self.cfg.safety.re_entry_vision_threshold:
                # Lane temporarily lost/low confidence. Reduce speed and try to re-acquire.
                speed = self.cfg.control.base_speed * 0.5
                self.state = "LOST_LANE_TEMP" # Intermediate state for low confidence

            elif not self.preview_mode:
                speed = self.cfg.control.base_speed # Maintain full speed if confident
                self.car.drive(speed, steering)
                self.last_steer = steering # Store steering for re-entry

            # Draw HUD based on driving state/sub-state
            drive_text = f"AUTO [Spd:{speed:.2f} Str:{steering:+.3f}]"
            cv2.putText(hud, drive_text, (w // 2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # ── IDLE State ──
            self.car.stop()
            cv2.putText(hud, "IDLE [Press 'a' to drive]",
                        (w // 2 - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def _calculate_vision_score(self, error):
        """
        Calculates a confidence score based on lane detection quality.
        Needs tuning based on actual perception output.
        (MOCK version: assuming smaller errors mean higher confidence)
        """
        # A simple model: new confidence = max(0.0, 1.0 - abs(error) / 0.5) # Example: error of 0.5m = 0 confidence.
        # This function should be part of the LaneDetector class, but for state machine logic testing, a mock works here.
        # Let's return a consistent value for now, assuming detection is working well based on your previous input.
        return 0.8 # Assume high initial confidence for testing re-entry logic.

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
            if self.state == self.STATE_DRIVING or self.state == self.STATE_SEARCHING:
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
        state_colors = {
            self.STATE_IDLE: (200, 200, 200),
            self.STATE_DRIVING: (0, 255, 0),
            self.STATE_OBSTACLE: (0, 0, 255),
            self.STATE_SEARCHING: (255, 165, 0),
            "LOST_LANE_TEMP": (0, 200, 200),
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
    is_preview = "--preview" in sys.argv
    app = Supervisor(preview_mode=is_preview)
    app.run()
