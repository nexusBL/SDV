import time
import numpy as np
import cv2
import math
from pal.utilities.lidar import Lidar

class LidarProcessor:
    """
    Manages the RPLidar A2M12 on the QCar 2 for forward obstacle detection.
    Maps LIDAR points to an occupancy grid and filters for a strict 
    forward Region of Interest (ROI) corridor.
    """
    def __init__(self):
        # 1 meter = 50 pixels in occupancy map
        self.pixelsPerMeter = 50
        
        # 8 meters x 8 meters total map area (400x400 grid)
        self.sideLengthScale = 8 * self.pixelsPerMeter
        self.decay = 0.9  # Fade factor for temporal smoothing
        
        # Max distance to care about (2 meters is safe for 1/10 scale)
        self.maxDistance = 2.0 
        self.map = np.zeros((self.sideLengthScale, self.sideLengthScale), dtype=np.float32)

        # Lidar settings (A2M12 has higher density, but 360 covers the full circle)
        self.numMeasurements = 360
        self.lidarMeasurementMode = 2
        self.lidarInterpolationMode = 0

        print("Initializing RPLidar A2M12...")
        try:
            self.myLidar = Lidar(
                type='RPLidar',
                numMeasurements=self.numMeasurements,
                rangingDistanceMode=self.lidarMeasurementMode,
                interpolationMode=self.lidarInterpolationMode
            )
            self.lidar_active = True
        except Exception as e:
            print(f"WARNING: LIDAR initialization failed. Is the port locked by ROS? Error: {e}")
            print("Obstacle Avoidance is now DISABLED. Lane filtering will continue.")
            self.lidar_active = False
            self.myLidar = None

        # Bounding box for obstacles (relative to map center at 200, 200)
        # Car width is roughly ~20 cm. Let's make the corridor 40 cm wide (-20 to +20).
        # y is right/left. 20 cm = 0.2m = 10 px.  Center is 200. => 190 to 210.
        self.roi_left_px = 190
        self.roi_right_px = 210
        # x is forward. we only want points in front of the car (> 0.2m forward).
        # Forward is -x in px space relative to 200 center, so px <= 190
        self.roi_forward_max_px = 190
        
        # Danger zone depth limit (how close an obstacle is to trigger stop)
        self.danger_zone_px = 160  # ~80cm away (200 - 160 = 40px = 80cm)

    def lidar_measure(self):
        if not getattr(self, 'lidar_active', False):
            return self.map, [], []
            
        self.map = self.decay * self.map
        self.myLidar.read()
        
        # Transform LIDAR polar coordinates to vehicle Cartesian frame
        anglesInBodyFrame = self.myLidar.angles * -1 + np.pi/2
        
        # Filter valid points
        idx = [i for i, v in enumerate(self.myLidar.distances) if 0.05 < v < self.maxDistance]
        
        x = self.myLidar.distances[idx] * np.cos(anglesInBodyFrame[idx])
        y = self.myLidar.distances[idx] * np.sin(anglesInBodyFrame[idx])

        # Convert to occupancy map coordinates
        pX = (self.sideLengthScale/2 - x*self.pixelsPerMeter).astype(np.uint16)
        pY = (self.sideLengthScale/2 - y*self.pixelsPerMeter).astype(np.uint16)
        
        # Filter for the strict forward corridor
        coordenadas_filtradas = [
            (px, py) for px, py in zip(pX, pY) 
            if self.roi_left_px <= py <= self.roi_right_px and px <= self.roi_forward_max_px
        ]
        
        if coordenadas_filtradas:
            pX_filtrado, pY_filtrado = zip(*coordenadas_filtradas)
            self.map[pX_filtrado, pY_filtrado] = 1
            return self.map, pX_filtrado, pY_filtrado
            
        return self.map, [], []

    def detect_object(self):
        """
        Returns True if a significant object is detected immediately in the vehicle's path.
        """
        map_, pX_, pY_ = self.lidar_measure()
        
        # Look for points that are *extremely* close (in the danger zone)
        # Remember, smaller pX means FURTHER away from center (200). 
        # So pX > 160 means within 0.8 meters. 
        # Actually in original code it was Y_ = [y for y in pY_ if y > 165], 
        # but pY is left/right. pX is forward! 
        # Original: px <= 190 (forward), py in [190, 210]. 
        # Danger check should be: pX > 140 (less than 1.2m away). 
        # Let's use pX to check distance instead of pY.
        
        close_points = [x for x in pX_ if x > self.danger_zone_px]
        
        # Require a cluster (e.g. at least 5 points) to avoid false positives from noise
        return len(close_points) > 5

    def end_lidar(self):
        if getattr(self, 'lidar_active', False) and self.myLidar:
            self.myLidar.terminate()
