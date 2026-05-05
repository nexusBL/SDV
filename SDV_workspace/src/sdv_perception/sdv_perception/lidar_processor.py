#!/usr/bin/env python3
"""
SDV LiDAR Processor — Thread-safe LiDAR scan analysis with threat zones.
"""

import numpy as np
import threading
from typing import Optional, Tuple, Dict

from .config import SDVConfig


class LidarProcessor:
    """Analyses LiDAR scans to detect obstacles in front/left/right arcs
    and returns a threat level with distance data."""

    THREAT_CRITICAL = 'CRITICAL'
    THREAT_WARNING = 'WARNING'
    THREAT_CLEAR = 'CLEAR'
    THREAT_UNKNOWN = 'UNKNOWN'

    def __init__(self, config: Optional[SDVConfig] = None):
        cfg = (config or SDVConfig.get()).lidar
        self.critical_dist = cfg['critical_distance']
        self.warning_dist = cfg['warning_distance']
        self.front_arc_deg = cfg['front_arc_degrees']
        self.max_range = cfg['max_range']
        self.min_range = cfg['min_range']

        self._latest_scan = None
        self._lock = threading.Lock()

    def update(self, scan_msg):
        """Store the latest LiDAR scan message (ROS2 LaserScan).
        Thread-safe — can be called from a ROS2 callback thread."""
        with self._lock:
            self._latest_scan = scan_msg

    def update_raw(self, ranges: np.ndarray, angle_min: float,
                   angle_increment: float):
        """Store raw scan data (for non-ROS2 usage)."""
        with self._lock:
            self._latest_scan = _RawScan(ranges, angle_min, angle_increment)

    def analyze(self) -> Tuple[str, float, Dict[str, float]]:
        """Analyze the latest scan and return threat assessment.

        Returns:
            (threat_level, min_front_distance, zone_distances_dict)
            threat_level: CRITICAL | WARNING | CLEAR | UNKNOWN
            min_front_distance: closest obstacle in front arc (meters)
            zone_distances: {'front': x, 'left': x, 'right': x}
        """
        with self._lock:
            scan = self._latest_scan

        if scan is None:
            return self.THREAT_UNKNOWN, 99.0, {}

        ranges = np.array(scan.ranges)
        angle_inc = scan.angle_increment
        angle_min = scan.angle_min
        n = len(ranges)

        if n == 0:
            return self.THREAT_UNKNOWN, 99.0, {}

        # Replace invalid readings
        ranges = np.where(
            (ranges < self.min_range) | (ranges > self.max_range),
            self.max_range, ranges
        )

        # Convert front arc to indices
        arc_rad = np.radians(self.front_arc_deg)
        idx_size = max(1, int(arc_rad / angle_inc))

        # Front = indices near 0° and near 360°
        front_l = max(0, n - idx_size)
        front_r = min(n, idx_size)
        front_ranges = np.concatenate([ranges[front_l:], ranges[:front_r]])

        # Left = ~90° (quarter turn)
        left_center = n // 4
        half = idx_size // 2
        left_ranges = ranges[
            max(0, left_center - half):min(n, left_center + half)
        ]

        # Right = ~270° (three-quarter turn)
        right_center = 3 * n // 4
        right_ranges = ranges[
            max(0, right_center - half):min(n, right_center + half)
        ]

        # Instead of taking the absolute minimum (which is highly sensitive to noise/dust),
        # use the 3rd percentile to reject extreme outliers (like a single noisy ray)
        def robust_min(zone_ranges, default=99.0):
            if len(zone_ranges) == 0:
                return default
            return float(np.percentile(zone_ranges, 3))

        min_front = robust_min(front_ranges)
        min_left  = robust_min(left_ranges)
        min_right = robust_min(right_ranges)

        zones = {
            'front': round(min_front, 2),
            'left': round(min_left, 2),
            'right': round(min_right, 2),
        }

        # ── Temporal Hysteresis for Threat Level ──
        # Prevent flickering between WARNING/CRITICAL due to noise
        raw_threat = self.THREAT_CLEAR
        if min_front < self.critical_dist:
            raw_threat = self.THREAT_CRITICAL
        elif min_front < self.warning_dist:
            raw_threat = self.THREAT_WARNING

        # Initialize history if missing
        if not hasattr(self, '_threat_history'):
            self._threat_history = []
        
        self._threat_history.append(raw_threat)
        if len(self._threat_history) > 3:  # Look at last 3 frames
            self._threat_history.pop(0)
            
        # Upgrade threat immediately (safety first), but downgrade slowly (needs consensus)
        if raw_threat == self.THREAT_CRITICAL:
            threat = self.THREAT_CRITICAL
        elif raw_threat == self.THREAT_WARNING:
            threat = self.THREAT_CRITICAL if self.THREAT_CRITICAL in self._threat_history else self.THREAT_WARNING
        else:
            if self.THREAT_CRITICAL in self._threat_history:
                threat = self.THREAT_CRITICAL
            elif self.THREAT_WARNING in self._threat_history:
                threat = self.THREAT_WARNING
            else:
                threat = self.THREAT_CLEAR

        return threat, min_front, zones


class _RawScan:
    """Simple container mimicking ROS2 LaserScan for non-ROS2 usage."""
    __slots__ = ('ranges', 'angle_min', 'angle_increment')

    def __init__(self, ranges, angle_min, angle_increment):
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_increment = angle_increment
