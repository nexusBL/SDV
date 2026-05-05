"""
safety_monitor.py - QCar2 LiDAR-Based Obstacle Detection
=========================================================
Encapsulates the RPLidar A2M12 sensor on QCar2 for real-time
frontal obstacle detection within a configurable Region of Interest.

Key features:
  - Pre-init cleanup via pkill to prevent the dreaded error -10
  - Configurable frontal cone ROI (±15° default)
  - Minimum distance filter to reject LiDAR self-reflections
  - Returns both boolean status and closest obstacle distance
"""

import subprocess
import time
import math

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
        initialize()     - Cleans up stale processes and starts LiDAR
        is_path_clear()  - Returns (bool, float) = (is_clear, closest_dist_m)
        terminate()      - Safely stops LiDAR hardware
    """

    def __init__(self, config):
        """
        Args:
            config: AppConfig instance containing safety parameters.
        """
        self.cfg = config.safety
        self.lidar = None
        self._mock_mode = (QCarLidar is None)

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
        except Exception as e:
            print(f"[SafetyMonitor] ✗ FATAL: LiDAR init failed: {e}")
            print("[SafetyMonitor] Try: sudo pkill -9 -f lidar && sleep 2")
            raise

    def is_path_clear(self):
        """
        Scans the frontal ROI cone for obstacles.

        The ROI is defined as a cone of ±roi_angle_deg centered on
        lidar_front_angle_deg. Any valid LiDAR hit within this cone
        that is closer than stop_distance_m triggers an obstacle alert.

        Returns:
            tuple: (is_clear: bool, closest_distance: float)
                   - is_clear = True if no obstacle in ROI
                   - closest_distance = meters to nearest obstacle, or -1.0
        """
        if self._mock_mode:
            return True, -1.0

        # Read latest LiDAR scan
        self.lidar.read()
        angles = self.lidar.angles        # Array of angles in radians
        distances = self.lidar.distances   # Array of distances in meters

        min_dist = float('inf')
        obstacle_found = False

        for angle_rad, dist in zip(angles, distances):
            # Convert to degrees and compute angular difference from front
            angle_deg = math.degrees(angle_rad)
            angular_diff = (angle_deg - self.cfg.lidar_front_angle_deg + 180) % 360 - 180

            # Check if this reading falls within our frontal ROI cone
            if abs(angular_diff) <= self.cfg.roi_angle_deg:
                # Filter out self-reflections and noise (< 5cm)
                # and check against stop distance
                if self.cfg.min_valid_distance_m < dist < self.cfg.stop_distance_m:
                    obstacle_found = True
                    min_dist = min(min_dist, dist)

        if obstacle_found:
            return False, min_dist

        return True, -1.0

    def terminate(self):
        """Safely stops the LiDAR motor and releases the USB port."""
        if self.lidar is not None:
            try:
                self.lidar.terminate()
                print("[SafetyMonitor] ✓ LiDAR terminated safely.")
            except Exception as e:
                print(f"[SafetyMonitor] Warning during termination: {e}")
