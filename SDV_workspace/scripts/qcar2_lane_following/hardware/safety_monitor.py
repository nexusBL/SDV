import subprocess
import time
import math

try:
    from pal.products.qcar import QCarLidar
except ImportError:
    print("[WARNING] QCar PAL libraries not found. Running SafetyMonitor in MOCK Mode.")
    QCarLidar = None

class SafetyMonitor:
    def __init__(self, config):
        self.cfg = config
        self.lidar = None
        self._mock_mode = (QCarLidar is None)

    def _cleanup_lidar(self):
        print("[SafetyMonitor] Cleaning up existing lidar processes via pkill...")
        subprocess.run("pkill -f lidar", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1.0) 

    def initialize(self):
        if self._mock_mode:
            print("[SafetyMonitor] Simulated LiDAR initialized.")
            return

        self._cleanup_lidar()
        print("[SafetyMonitor] Starting QCarLidar...")
        self.lidar = QCarLidar()
        print("[SafetyMonitor] LiDAR started successfully.")

    def is_path_clear(self):
        if self._mock_mode:
            return True, -1.0

        self.lidar.read()
        angles = self.lidar.angles
        distances = self.lidar.distances
        
        min_dist = float('inf')
        obstacle_detected = False

        for angle_rad, dist in zip(angles, distances):
            angle_deg = math.degrees(angle_rad)
            angular_diff = (angle_deg - self.cfg.safety.lidar_front_angle_deg + 180) % 360 - 180
            
            if abs(angular_diff) < self.cfg.safety.roi_angle_deg:
                if 0.05 < dist < self.cfg.safety.stop_distance_m:
                    obstacle_detected = True
                    min_dist = min(min_dist, dist)

        if obstacle_detected:
            return False, min_dist

        return True, -1.0

    def terminate(self):
        if self.lidar is not None:
            self.lidar.terminate()
            print("[SafetyMonitor] LiDAR terminated safely.")

