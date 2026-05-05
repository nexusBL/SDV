import unittest
import numpy as np
import math
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control.reactive_avoidance_controller import ReactiveController, ReactiveAvoidanceParams

class TestGapCheck(unittest.TestCase):
    def setUp(self):
        self.cfg = ReactiveAvoidanceParams(
            car_width_m=0.35,
            gap_max_angle_deg=60.0,
            max_lidar_dist=5.0
        )
        self.controller = ReactiveController(self.cfg)

    def test_wide_gap(self):
        # Two obstacles at 1m, one at -30 deg, one at +30 deg
        # Gap is approx 1.0 * (60 deg in rad) = 1.0 * 1.047 = 1.047m
        angles = [math.radians(-30), math.radians(30)]
        distances = [1.0, 1.0]
        angle, passable = self.controller.find_best_gap(angles, distances)
        self.assertTrue(passable)
        self.assertAlmostEqual(angle, 0.0)

    def test_narrow_gap(self):
        # Two obstacles at 1m, one at -5 deg, one at +5 deg
        # Gap is approx 1.0 * (10 deg in rad) = 1.0 * 0.174 = 0.174m (too narrow)
        angles = [math.radians(-5), math.radians(5)]
        distances = [1.0, 1.0]
        angle, passable = self.controller.find_best_gap(angles, distances)
        # Should find gaps at the edges instead
        self.assertTrue(passable)
        self.assertNotAlmostEqual(angle, 0.0)

    def test_no_obstacles(self):
        # No obstacles should result in no passable gap derived from points, 
        # but my implementation requires at least 2 points to check internal gaps.
        # It handles edge gaps if there are points.
        angles = [math.radians(0)]
        distances = [1.0]
        angle, passable = self.controller.find_best_gap(angles, distances)
        # One point: checks between point and left/right edges
        self.assertTrue(passable)

if __name__ == "__main__":
    unittest.main()
