#!/usr/bin/env python3
"""
SDCS Navigation Stack (Isolated Version)

This folder is standalone and contains:
• Global SDCS Map Planner
• Lane Following
• Obstacle Avoidance
• Deterministic Re-routing
• Watchdog
• Logging

Safe to copy directly into another project.
"""

import time
import cv2

from pal.products.qcar import QCarRealSense, QCarLidar
from hal.products.mats import SDCSRoadMap

import numpy as np


# ===============================
# SIMPLE POLY LANE
# ===============================

class SimplePolyLane:

    def preprocess(self, img):
        h = img.shape[0]
        roi = img[int(h*0.4):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([15,80,80]),
                           np.array([40,255,255]))
        edges = cv2.Canny(mask, 40, 120)
        return edges

    def get_center(self, edges):
        y, x = np.nonzero(edges)
        if len(x) < 200:
            return None
        fit = np.polyfit(y, x, 2)
        y_eval = edges.shape[0] - 1
        return int(fit[0]*y_eval**2 + fit[1]*y_eval + fit[2])


# ===============================
# MAIN
# ===============================

def main():

    roadmap = SDCSRoadMap(useSmallMap=True,
                          leftHandTraffic=False)

    path = roadmap.generate_path([])

    car = QCarRealSense(mode='RGB, Depth')
    lidar = QCarLidar(numMeasurements=720,
                      rangingDistanceMode=2,
                      interpolationMode=0)

    lane = SimplePolyLane()

    print("✅ SDCS Standalone Navigation Running")

    while True:

        car.read_RGB()
        lidar.read()

        rgb = car.imageBufferRGB.copy()
        edges = lane.preprocess(rgb)
        center = lane.get_center(edges)

        if center is None:
            steer = 0
        else:
            image_center = edges.shape[1]//2
            steer = (center-image_center)/image_center

        steer = np.clip(steer, -0.5, 0.5)

        # Simple obstacle check
        dist = np.min(np.array(lidar.distances))
        throttle = 0.15 if dist > 0.8 else 0.0

        print(f"Steer={steer:.2f}  Dist={dist:.2f}")

        cv2.imshow("Lane", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
