#!/usr/bin/env python3
"""
navigation_main.py - QCar2 Pure Pursuit Waypoint Navigator
==========================================================
Replaces the camera-based lane follower with a GPS-style Path Follower.
Uses the GlobalPlanner, PurePursuitController, and a KinematicTracker
to drive the physical track.

Controls (focus the OpenCV window):
  [g] - Generate new path & Start autonomous driving
  [s] - Stop (manual override)
  [q] - Quit and shutdown all hardware safely
"""

import os
# ── CRITICAL: Fix nvarguscamerasrc EGL authorization ──
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

import sys
import time
import math
import cv2
import numpy as np
import traceback

# ── Imports ──
import sys
import time
import math
import cv2
import numpy as np
import traceback

from config import AppConfig
from hardware.safety_monitor import SafetyMonitor
from hardware.car_controller import CarController
from hardware.kinematic_tracker import KinematicTracker
from control.pure_pursuit import PurePursuitController
from global_planner import GlobalPlanner


class NavigationSupervisor:
    """
    Main application supervisor for Pure Pursuit waypoint navigation.
    """

    STATE_IDLE = "IDLE"
    STATE_DRIVING = "DRIVING"
    STATE_OBSTACLE = "OBSTACLE_STOP"

    def __init__(self, target_nodes, preview_mode=False, autostart=False, headless=False):
        self.cfg = AppConfig()
        self.preview_mode = preview_mode
        self.autostart = autostart
        self.headless = headless
        self.target_nodes = target_nodes
        
        self.state = self.STATE_IDLE
        self.running = True
        
        self.path = None
        self.trajectory_coords = None
        
        # Sensor feedback state
        self._last_velocity = 0.0
        self._gyro_yaw_rate = 0.0
        self._motor_tach = 0.0

        print("\n" + "=" * 50)
        print("  🧭  QCar2 Pure Pursuit Navigator  🧭")
        print("=" * 50)
        
        self.safety = SafetyMonitor(self.cfg)
        self.car = CarController(self.cfg)
        self.planner = GlobalPlanner(use_small_map=True, map_image_dir=".")
        
        # Planners & Trackers
        self.pursuit = PurePursuitController(self.cfg)
        self.tracker = KinematicTracker(self.cfg)

    def initialize_route(self):
        """Generates the global path and sets up trackers."""
        print(f"[Navigator] Requesting route through nodes: {self.target_nodes}")
        self.trajectory_coords = self.planner.plan_path(self.target_nodes)
        
        if self.trajectory_coords is None:
            print("[Navigator] FATAL: Invalid route requested.")
            return False
            
        self.pursuit.set_path(self.trajectory_coords)
        
        # Initialize the tracker exactly at the start of the path, facing the second point
        start_x = self.trajectory_coords[0, 0]
        start_y = self.trajectory_coords[1, 0]
        next_x = self.trajectory_coords[0, 1]
        next_y = self.trajectory_coords[1, 1]
        
        start_theta = math.atan2(next_y - start_y, next_x - start_x)
        self.tracker.reset((start_x, start_y, start_theta))
        print(f"[Navigator] ✓ Tracker initialized at spawn pose: X:{start_x:.2f}, Y:{start_y:.2f}, Th:{math.degrees(start_theta):.0f}°")
        return True

    def run(self):
        """Main execution loop."""
        try:
            self.safety.initialize()
            self.car.initialize()
            if not self.initialize_route():
                return
        except Exception as e:
            print(f"\n[FATAL] Initialization failed: {e}")
            self._shutdown()
            return

        print("\n[Navigator] All systems nominal. Entering control loop.")
        if self.autostart:
            print("[Navigator] 🚀 AUTOSTART ENABLED. Engaging autonomous mode...")
            self.state = self.STATE_DRIVING
        else:
            print("[Navigator] Controls: [g]=Go  [s]=Stop  [q]=Quit\n")

        try:
            while self.running:
                loop_start = time.time()

                # ──── 1. SAFETY CHECK (LiDAR) ────
                is_clear, obstacle_dist = self.safety.is_path_clear()

                # ──── 2. STATE ESTIMATION ────
                # (Normally this comes from OptiTrack. We query our Kinematic Tracker)
                current_pose = self.tracker.get_pose()

                # ──── 3. CONTROL LOOP ────
                target_velocity = 0.0
                steering_cmd = 0.0
                lookahead_pt = None

                if not is_clear:
                    # EMERGENCY STOP OVERRIDE
                    if self.state != self.STATE_OBSTACLE:
                         print(f"[Navigator] ⚠️  OBSTACLE at {obstacle_dist:.2f}m! Stopping.")
                    self.state = self.STATE_OBSTACLE
                    self.car.hazard_stop()
                    
                elif self.state == self.STATE_OBSTACLE and is_clear:
                    # OBSTACLE MOVED
                    print("[Navigator] ✓ Path clear. Resuming.")
                    self.state = self.STATE_DRIVING
                    
                if self.state == self.STATE_DRIVING:
                    # Check distance to final waypoint
                    end_x = self.trajectory_coords[0, -1]
                    end_y = self.trajectory_coords[1, -1]
                    dist_to_goal = math.hypot(end_x - current_pose[0], end_y - current_pose[1])
                    if dist_to_goal < 0.2:
                        print(f"\\n[Navigator] 🏁 Destination reached! Stopping.")
                        self.state = self.STATE_IDLE
                        self.car.stop()
                        continue

                    # PURE PURSUIT ALGORITHM
                    steering_cmd, target_velocity, lookahead_pt = self.pursuit.compute(current_pose)
                    
                    if not self.preview_mode:
                        self.car.drive(target_velocity, steering_cmd)
                    else:
                        self.car.stop()
                        
                elif self.state == self.STATE_IDLE:
                    self.car.stop()

                # ──── 4. READ SENSORS & UPDATE ODOMETRY ────
                # Use REAL IMU gyroscope + encoder tachometer for pose estimation
                # instead of integrating commanded values (which drift rapidly)
                self._gyro_yaw_rate, self._motor_tach = self.car.read_sensors()
                
                if self.state == self.STATE_DRIVING:
                    self.tracker.update_from_sensors(
                        self._gyro_yaw_rate, 
                        self._motor_tach, 
                        target_velocity
                    )
                    self._last_velocity = target_velocity

                # ──── 5. RENDER TELEMETRY HUD ────
                self._render_telemetry_hud(current_pose, lookahead_pt, target_velocity, steering_cmd)

                # ──── 6. INPUT HANDLING ────
                if not self.headless:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        self.running = False
                    elif key == ord('g') and self.state == self.STATE_IDLE:
                        print("[Navigator] ✓ Autonomous waypoint navigation ENGAGED!")
                        self.state = self.STATE_DRIVING
                    elif key == ord('s') and self.state == self.STATE_DRIVING:
                        print("[Navigator] ■ Manual stop.")
                        self.state = self.STATE_IDLE
                        self.car.stop()
                else:
                    # In headless mode, waitKey won't work. We just loop at roughly 30Hz.
                    time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n[Navigator] Ctrl+C caught. Shutting down...")
        finally:
            self._shutdown()

    def _render_telemetry_hud(self, pose, lookahead_pt, vel, steering):
        """Draws a live OpenCV map showing the global path, car, and lookahead point."""
        # Create a black canvas (800x800)
        hud = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Scaling parameters (1 real-world meter = 150 pixels)
        # Shift origin to center of screen
        scale = 150
        offset_x, offset_y = 400, 400
        
        # Helper function to convert real (x,y) to screen (px, py)
        def to_screen(rx, ry):
             # Negate Y because OpenCV Y goes down, math Y goes up
             return int(rx * scale + offset_x), int(-ry * scale + offset_y)
             
        # Draw the target route (green)
        if self.trajectory_coords is not None:
            pts = []
            for i in range(self.trajectory_coords.shape[1]):
                rx = self.trajectory_coords[0, i]
                ry = self.trajectory_coords[1, i]
                pts.append(to_screen(rx, ry))
            
            # Draw as lines
            for i in range(1, len(pts)):
                cv2.line(hud, pts[i-1], pts[i], (0, 100, 0), 2)
                
        # Draw Lookahead Point (Red)
        if lookahead_pt is not None:
             sx, sy = to_screen(lookahead_pt[0], lookahead_pt[1])
             cv2.circle(hud, (sx, sy), 5, (0, 0, 255), -1)
             
        # Draw Car (Blue)
        car_x, car_y = to_screen(pose[0], pose[1])
        cv2.circle(hud, (car_x, car_y), 8, (255, 0, 0), -1)
        
        # Draw Car Heading Line
        heading_end_x = pose[0] + 0.5 * math.cos(pose[2])
        heading_end_y = pose[1] + 0.5 * math.sin(pose[2])
        hx, hy = to_screen(heading_end_x, heading_end_y)
        cv2.line(hud, (car_x, car_y), (hx, hy), (255, 255, 0), 2)
        
        # Telemetry Text
        cv2.putText(hud, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(hud, f"Pose: X:{pose[0]:.2f} Y:{pose[1]:.2f} Th:{math.degrees(pose[2]):.0f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(hud, f"Cmd : V:{vel:.2f}m/s Str:{math.degrees(steering):.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if not self.headless:
            cv2.imshow("QCar2 Pure Pursuit Navigator", hud)
        else:
            # In headless mode, save the HUD image periodically
            if int(time.time() * 2) % 2 == 0: # Save once every 0.5s approx
                 cv2.imwrite("telemetry_hud.jpg", hud)


    def _shutdown(self):
        """Guarantees clean shutdown of all hardware."""
        print("\n[Navigator] Shutting down...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.car.terminate()
        self.safety.terminate()
        print("[Navigator] ✓ Shutdown complete.\n")


if __name__ == "__main__":
    is_preview = "--preview" in sys.argv
    is_autostart = "--autostart" in sys.argv
    is_headless = "--headless" in sys.argv or os.environ.get("DISPLAY") is None
    
    # Define the nodes we want to travel through.
    waypoints = [0, 4, 12]
    
    app = NavigationSupervisor(
        target_nodes=waypoints, 
        preview_mode=is_preview,
        autostart=is_autostart,
        headless=is_headless
    )
    app.run()
