#!/usr/bin/env python3
"""
main.py - QCar2 Autonomous Lane Following & Obstacle Avoidance Orchestrator
============================================================================
This is the main entry point for the QCar2 lane following system.
It coordinates all subsystems: camera, LiDAR, perception, control, and actuation.

State Machine:
  IDLE ──[press 'a']──> DRIVING ──[obstacle]──> OBSTACLE_STOP
    ^                      |                          |
    |                      |                    [obstacle clears]
    +──[press 's']─────────+                          |
                                                      v
                                                  RESUMING ──[N frames]──> DRIVING

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

from config import AppConfig
from hardware.camera_manager import CameraManager
from hardware.safety_monitor import SafetyMonitor
from hardware.car_controller import CarController
from perception.lane_cv import LaneDetector
from control.pid_controller import PIDController


class Supervisor:
    """
    Main application supervisor that orchestrates all QCar2 subsystems.

    Implements a clean state machine with explicit states:
    - IDLE:          Car is stationary, waiting for 'a' keypress
    - DRIVING:       Car is following the lane autonomously
    - OBSTACLE_STOP: Car has stopped due to LiDAR obstacle detection
    - RESUMING:      Obstacle cleared, waiting N frames before driving again

    The supervisor guarantees safe shutdown via try/finally blocks,
    ensuring motors are zeroed even if the script crashes.
    """

    # State constants
    STATE_IDLE = "IDLE"
    STATE_DRIVING = "DRIVING"
    STATE_OBSTACLE = "OBSTACLE_STOP"
    STATE_RESUMING = "RESUMING"

    def __init__(self, preview_mode=False):
        """
        Args:
            preview_mode: bool - if True, motors are disabled (vision-only mode)
        """
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.state = self.STATE_IDLE
        self.running = True

        # Resume countdown (frames to wait after obstacle clears)
        self.resume_counter = 0

        # FPS tracking
        self._fps_counter = 0
        self._fps_start = time.time()
        self._fps_display = 0.0

        # Initialize subsystems
        print("\n" + "=" * 50)
        print("  🚗  QCar2 Autonomous Lane Following System  🚗")
        print("=" * 50)
        mode_str = "PREVIEW (motors disabled)" if preview_mode else "LIVE (motors enabled)"
        print(f"  Mode: {mode_str}")
        print("=" * 50 + "\n")

        print("[Supervisor] Initializing subsystem drivers...")
        self.camera = CameraManager(self.cfg)
        self.safety = SafetyMonitor(self.cfg)
        self.car = CarController(self.cfg)
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
        State machine logic for driving, obstacle stop, and resuming.
        """
        h, w = hud.shape[:2]

        if not is_clear:
            # ── OBSTACLE DETECTED: Emergency Stop ──
            if self.state != self.STATE_OBSTACLE:
                print(f"[Supervisor] ⚠️  OBSTACLE at {obstacle_dist:.2f}m! Stopping.")
            self.state = self.STATE_OBSTACLE
            self.resume_counter = self.cfg.safety.resume_delay_frames
            self.car.hazard_stop()
            self.controller.reset_state()

            # Draw obstacle warning on HUD
            cv2.putText(hud, f"!!! OBSTACLE {obstacle_dist:.2f}m !!!",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        elif self.state == self.STATE_OBSTACLE or self.state == self.STATE_RESUMING:
            # ── OBSTACLE CLEARED: Countdown to Resume ──
            self.state = self.STATE_RESUMING
            self.resume_counter -= 1
            self.car.stop()

            cv2.putText(hud,
                        f"RESUMING IN {self.resume_counter}...",
                        (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            if self.resume_counter <= 0:
                print("[Supervisor] ✓ Path clear. Resuming driving.")
                self.state = self.STATE_DRIVING
                self.resume_counter = 0

        elif self.state == self.STATE_DRIVING:
            # ── AUTONOMOUS DRIVING ──
            if not self.preview_mode:
                steering = self.controller.compute(error, dt)
                self.car.drive(self.cfg.control.base_speed, steering)

                cv2.putText(hud,
                            f"AUTO [Spd:{self.cfg.control.base_speed:.2f} "
                            f"Str:{steering:+.3f}]",
                            (w // 2 - 50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.car.stop()
                cv2.putText(hud, "PREVIEW [MOTORS OFF]",
                            (w // 2 - 100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            # ── IDLE STATE ──
            self.car.stop()
            cv2.putText(hud, "IDLE [Press 'a' to drive]",
                        (w // 2 - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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
            if self.state == self.STATE_DRIVING:
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
            self.STATE_IDLE: (200, 200, 200),
            self.STATE_DRIVING: (0, 255, 0),
            self.STATE_OBSTACLE: (0, 0, 255),
            self.STATE_RESUMING: (0, 255, 255),
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
