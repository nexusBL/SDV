import sys
import os
import time
import csv
import cv2

# Ensure Quanser libraries are on path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from config import AppConfig
from hardware.camera_manager import CameraManager
from hardware.safety_monitor import SafetyMonitor
from hardware.car_controller import CarController
from perception.lane_cv import LaneDetector
from control.pid_controller import PIDController


class Supervisor:
    """
    Main control loop for QCar2 autonomous lane following.

    Modes:
        --preview : Motors disabled, perception only (safe for testing)
        (default) : Full autonomous driving with obstacle avoidance

    Controls (in OpenCV window):
        [a] Start autonomous driving
        [s] Stop / manual override
        [q] Quit and shutdown hardware
    """

    def __init__(self, preview_mode=False):
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.running = True
        self.autonomous = False

        print("\n" + "="*50)
        print("  🚗 QCar2 Lane Following System")
        print("="*50)
        print(f"  Mode:       {'PREVIEW (motors off)' if preview_mode else 'LIVE'}")
        print(f"  Resolution: {self.cfg.camera.width}×{self.cfg.camera.height}")
        print(f"  Camera FPS: {self.cfg.camera.fps}")
        print("="*50 + "\n")

        # Initialize subsystems
        print("[Supervisor] Initializing subsystems...")
        self.camera     = CameraManager(self.cfg)
        self.safety     = SafetyMonitor(self.cfg)
        self.car        = CarController(self.cfg)
        self.perception = LaneDetector(self.cfg)
        self.controller = PIDController(self.cfg)

        # CSV logging
        self.log_file = None
        self.csv_writer = None

    def _init_logging(self):
        """Initialize CSV log file for post-run analysis."""
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(log_dir, "run_log.csv")
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp', 'offset_m', 'curvature_m', 'confidence',
            'steering_cmd', 'throttle_cmd', 'left_pixels', 'right_pixels'
        ])
        print(f"[Supervisor] Logging to: {log_path}")

    def run(self):
        # 1. Hardware initialization
        try:
            self.camera.initialize()
            self.safety.initialize()
            self.car.initialize()
            self._init_logging()
        except Exception as e:
            print(f"[FATAL] Failed to initialize hardware: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown()
            return

        print("\n[Supervisor] All systems nominal. Starting main loop.")
        self._print_controls()

        fps_counter = 0
        fps_start = time.time()
        fps_display = 0.0

        obstacle_cleared_frames = 0
        was_blocked = False

        try:
            while self.running:
                loop_start = time.time()

                # ── 1. Perception ──────────────────────────────────────
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                error_m, hud = self.perception.process_frame(frame)

                # ── 2. Safety / LiDAR ─────────────────────────────────
                is_clear, dist = self.safety.is_path_clear()

                # ── 3. Control & Arbitration ──────────────────────────
                steering = 0.0
                throttle = 0.0

                if not is_clear:
                    # Obstacle → immediate stop
                    self.car.stop()
                    self.controller.reset_state()
                    if not was_blocked:
                        print(f"[Supervisor] !! OBSTACLE DETECTED at {dist:.2f}m !! Halting.")
                    was_blocked = True
                    cv2.putText(hud, f"!!! OBSTACLE {dist:.2f}m !!!",
                                (150, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3)
                else:
                    if was_blocked:
                        obstacle_cleared_frames += 1
                        if obstacle_cleared_frames < self.cfg.safety.resume_time_frames:
                            self.car.stop()
                            cv2.putText(hud,
                                f"CLEARING ({obstacle_cleared_frames}/"
                                f"{self.cfg.safety.resume_time_frames})",
                                (250, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 255), 3)
                        else:
                            print("[Supervisor] Path clear. Resuming.")
                            was_blocked = False
                            obstacle_cleared_frames = 0

                    if not was_blocked:
                        if self.autonomous and not self.preview_mode:
                            dt = max(time.time() - loop_start, 0.005)
                            steering = self.controller.compute(error_m, dt)

                            # Adaptive speed: slow on curves
                            if self.perception.curvature_radius_m < \
                                    self.cfg.control.curvature_speed_threshold:
                                throttle = self.cfg.control.curve_speed
                            else:
                                throttle = self.cfg.control.base_speed

                            # Periodic debug print for autonomous driving
                            if int(time.time() * 2) % 2 == 0:
                                sys.stdout.write(f"\r[AUTO] Thr:{throttle:.2f} Str:{steering:+.3f} Err:{error_m if error_m else 0:+.3f} L:{self.perception.left_pixel_count} R:{self.perception.right_pixel_count}   ")
                                sys.stdout.flush()

                            self.car.drive(throttle, steering)

                            cv2.putText(hud,
                                f"AUTO [Spd:{throttle:.2f} Str:{steering:+.3f}]",
                                (380, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                        else:
                            self.car.stop()
                            label = "PREVIEW [MOTORS OFF]" if self.preview_mode \
                                else "MANUAL [Press 'a']"
                            cv2.putText(hud, label, (380, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 255), 2)

                # ── 4. FPS Counter ─────────────────────────────────────
                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    fps_start = time.time()

                cv2.putText(hud, f"FPS: {fps_display:.1f}",
                            (20, 175), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)

                # ── 5. Display ─────────────────────────────────────────
                cv2.imshow("QCar2 Lane Following", hud)

                # ── 6. CSV Log ─────────────────────────────────────────
                if self.csv_writer:
                    self.csv_writer.writerow([
                        f"{time.time():.3f}",
                        f"{self.perception.lateral_offset_m:.4f}",
                        f"{self.perception.curvature_radius_m:.2f}",
                        f"{self.perception.confidence:.2f}",
                        f"{steering:.4f}",
                        f"{throttle:.3f}",
                        self.perception.left_pixel_count,
                        self.perception.right_pixel_count,
                    ])

                # ── 7. Keyboard Input ──────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Supervisor] 'q' pressed. Shutting down...")
                    self.running = False
                elif key == ord('a'):
                    if not self.autonomous:
                        print("[Supervisor] → Autonomous Mode ENGAGED ←")
                    self.autonomous = True
                    self.controller.reset_state()
                elif key == ord('s'):
                    if self.autonomous:
                        print("[Supervisor] → Autonomous Mode DISABLED ←")
                    self.autonomous = False
                    self.car.stop()

        except KeyboardInterrupt:
            print("\n[Supervisor] Caught Ctrl+C. Shutting down...")
        except Exception as e:
            print(f"\n[Supervisor] FATAL: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def _print_controls(self):
        print("\n  Controls (focus OpenCV window):")
        print("    [a] Start Autonomous Driving")
        print("    [s] Stop Car / Manual Override")
        print("    [q] Quit & Shutdown Hardware\n")

    def shutdown(self):
        print("\n[Supervisor] Shutting down all hardware...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.car.terminate()
        self.safety.terminate()
        self.camera.terminate()
        if self.log_file:
            self.log_file.close()
            print("[Supervisor] CSV log saved.")
        print("[Supervisor] Shutdown complete. Goodbye! 👋")


if __name__ == "__main__":
    is_preview = "--preview" in sys.argv
    app = Supervisor(preview_mode=is_preview)
    app.run()
