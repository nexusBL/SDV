"""
safety_monitor.py - QCar2 LiDAR-Based Obstacle Detection
=========================================================
Encapsulates the RPLidar A2M12 sensor on QCar2 for real-time
frontal obstacle detection within a configurable Region of Interest.

Key features:
  - Pre-init cleanup via pkill to prevent the dreaded error -10
  - Configurable frontal cone ROI (±30° default)
  - Minimum distance filter to reject LiDAR self-reflections
  - Minimum point count filter to reject single-point noise
  - Returns both boolean status and closest obstacle distance
  - --lidar-debug mode: prints raw scan data for angle calibration

CALIBRATION: Run `python3 main.py --lidar-debug` to see which angle
corresponds to "forward" on your car, then set lidar_front_angle_deg
in config.py accordingly. Typical QCar2 = 180.0°.
"""

import subprocess
import time
import math
import collections

import numpy as np

try:
    from pal.products.qcar import QCarLidar
except ImportError:
    print("[WARNING] QCarLidar PAL library not found. SafetyMonitor runs in MOCK mode.")
    QCarLidar = None


class SafetyMonitor:
    """
    LiDAR-based obstacle detection for the QCar2.

    The RPLidar A2M12 provides 360° scanning at up to 8000 samples/sec.
    We filter readings to a narrow frontal cone and check for obstacles
    within the configured stop distance.

    Methods:
        initialize()        - Cleans up stale processes and starts LiDAR
        is_path_clear()     - Returns (bool, float) = (is_clear, closest_dist_m)
        print_debug_scan()  - Prints raw angle/distance data for calibration
        terminate()         - Safely stops LiDAR hardware
    """

    def __init__(self, config, debug_mode=False):
        """
        Args:
            config:     AppConfig instance containing safety parameters.
            debug_mode: If True, prints live LiDAR scan data each frame.
        """
        self.cfg = config.safety
        self.lidar = None
        self._mock_mode = (QCarLidar is None)
        self._debug_mode = debug_mode
        self._frame_count = 0
        # Latest scan (filled in is_path_clear); used by reactive avoidance
        self._last_angles = np.array([], dtype=float)
        self._last_distances = np.array([], dtype=float)

    def _cleanup_stale_lidar(self):
        """
        Kills any lingering lidar processes from previous crashed runs.
        This prevents the dreaded 'error -10' on QCarLidar initialization
        which occurs when the USB serial port is still held by a zombie process.
        """
        print("[SafetyMonitor] Cleaning up stale lidar processes...")
        try:
            subprocess.run(
                "pkill -f lidar",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                timeout=5
            )
        except subprocess.TimeoutExpired:
            print("[SafetyMonitor] Warning: pkill timed out.")
        # Allow OS to fully release the serial port
        time.sleep(1.0)

    def initialize(self):
        """
        Initializes the RPLidar A2M12 sensor.
        Performs pre-cleanup to avoid error -10 from held USB ports.
        """
        if self._mock_mode:
            print("[SafetyMonitor] MOCK mode - no physical LiDAR.")
            return

        self._cleanup_stale_lidar()

        print("[SafetyMonitor] Starting QCarLidar (RPLidar A2M12)...")
        try:
            self.lidar = QCarLidar()
            print("[SafetyMonitor] ✓ LiDAR initialized successfully.")
            if self._debug_mode:
                print("[SafetyMonitor] *** DEBUG MODE ON - printing scan data each frame ***")
                print(f"[SafetyMonitor] Watching frontal cone: "
                      f"{self.cfg.lidar_front_angle_deg}° ± {self.cfg.roi_angle_deg}°")
        except Exception as e:
            print(f"[SafetyMonitor] ✗ FATAL: LiDAR init failed: {e}")
            print("[SafetyMonitor] Try: sudo pkill -9 -f lidar && sleep 2")
            raise

    def is_path_clear(self):
        """
        Scans the frontal ROI cone for obstacles.

        Fixes vs original:
          1. Enforces min_obstacle_points (was in config but never used)
          2. Correctly handles angle wrapping for any front angle (0-360°)
          3. Debug mode prints raw scan data for angle calibration

        Returns:
            tuple: (is_clear: bool, closest_distance: float)
                   - is_clear = True if no obstacle in ROI
                   - closest_distance = meters to nearest obstacle, or -1.0
        """
        if self._mock_mode:
            self._last_angles = np.array([], dtype=float)
            self._last_distances = np.array([], dtype=float)
            return True, -1.0

        # Read latest LiDAR scan
        self.lidar.read()
        angles = self.lidar.angles        # Array of angles in radians
        distances = self.lidar.distances  # Array of distances in meters
        try:
            self._last_angles = np.asarray(angles, dtype=float).ravel()
            self._last_distances = np.asarray(distances, dtype=float).ravel()
        except Exception:
            self._last_angles = np.array([], dtype=float)
            self._last_distances = np.array([], dtype=float)

        min_dist = float('inf')
        obstacle_points = 0
        frontal_hits = []  # for debug output

        for angle_rad, dist in zip(angles, distances):
            # Convert to degrees and compute angular difference from configured front
            angle_deg = math.degrees(angle_rad) % 360
            angular_diff = (angle_deg - self.cfg.lidar_front_angle_deg + 180) % 360 - 180

            # Check if this reading falls within our frontal ROI cone
            if abs(angular_diff) <= self.cfg.roi_angle_deg:
                # Filter out self-reflections/noise and track min dist for slow-down
                if self.cfg.min_valid_distance_m < dist < self.cfg.max_lidar_range_m:
                    min_dist = min(min_dist, dist)
                    if dist <= (self.cfg.stop_distance_cm / 100.0):
                        obstacle_points += 1

                if self._debug_mode:
                    frontal_hits.append((round(angle_deg, 1), round(dist, 3)))

        # Debug output (print every 10 frames to avoid spam)
        self._frame_count += 1
        if self._debug_mode and (self._frame_count % 10 == 0):
            self._print_debug(angles, distances, frontal_hits, obstacle_points, min_dist)

        # Require minimum point count to avoid false positives from noise
        if obstacle_points >= self.cfg.min_obstacle_points:
            return False, min_dist

        return True, -1.0

    def get_last_scan(self):
        """
        Polar scan from the most recent is_path_clear() call (same frame).

        Returns:
            (angles_rad, distances_m): numpy 1-D arrays; empty in MOCK mode.
        """
        return self._last_angles, self._last_distances

    def get_front_arc_scan(self, front_arc_deg: float):
        """
        Returns LiDAR points in a front arc centered on cfg.lidar_front_angle_deg.

        Output angles are centered such that 0 rad = front, positive = left, negative = right.

        Args:
            front_arc_deg: half-angle of arc (e.g. 45 means [-45,+45])

        Returns:
            (angles_rad_centered, distances_m, min_dist_m)
        """
        if self._mock_mode or self._last_angles.size == 0 or self._last_distances.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float), float("inf")

        angles = self._last_angles
        distances = self._last_distances

        # Convert to degrees and center on configured "front"
        angle_deg = (np.degrees(angles) % 360.0).astype(float)
        angular_diff = (angle_deg - float(self.cfg.lidar_front_angle_deg) + 180.0) % 360.0 - 180.0

        mask = (
            (np.abs(angular_diff) <= float(front_arc_deg))
            & (distances > float(self.cfg.min_valid_distance_m))
            & (distances < float(self.cfg.max_lidar_range_m))
        )

        arc_diff_deg = angular_diff[mask]
        arc_dist = distances[mask]
        if arc_dist.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float), float("inf")

        arc_angles_rad = np.radians(arc_diff_deg)  # centered: 0=front
        return arc_angles_rad.astype(float), arc_dist.astype(float), float(np.min(arc_dist))

    def _print_debug(self, angles, distances, frontal_hits, obstacle_points, min_dist):
        """Prints calibration-friendly scan data."""
        print("\n" + "─" * 55)
        print(f"[LiDAR DEBUG] Frame {self._frame_count} | "
              f"Total points: {len(angles)} | "
              f"Front angle cfg: {self.cfg.lidar_front_angle_deg}°")

        # Show 5 closest overall readings to help find the real front
        valid = [(round(math.degrees(a) % 360, 1), round(d, 3))
                 for a, d in zip(angles, distances)
                 if self.cfg.min_valid_distance_m < d < self.cfg.max_lidar_range_m]
        top5 = sorted(valid, key=lambda x: x[1])[:5]
        print(f"  Top-5 closest readings (angle°, dist_m): {top5}")
        print(f"  Frontal cone hits ({self.cfg.lidar_front_angle_deg}° ±{self.cfg.roi_angle_deg}°): "
              f"{frontal_hits[:10]}")
        print(f"  Obstacle points in ROI: {obstacle_points} "
              f"(need >= {self.cfg.min_obstacle_points})")
        if obstacle_points >= self.cfg.min_obstacle_points:
            print(f"  ⚠️  OBSTACLE DETECTED at {min_dist:.3f}m")
        else:
            print(f"  ✓  Path clear")
        print("─" * 55)

    def terminate(self):
        """Safely stops the LiDAR motor and releases the USB port."""
        if self.lidar is not None:
            try:
                self.lidar.terminate()
                print("[SafetyMonitor] ✓ LiDAR terminated safely.")
            except Exception as e:
                print(f"[SafetyMonitor] Warning during termination: {e}")
