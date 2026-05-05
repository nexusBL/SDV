#!/usr/bin/env python3
"""
SDV Depth Fusion — Enrich YOLO detections with real 3D distance from RealSense depth.
"""

import numpy as np
from typing import List, Optional

from .config import SDVConfig
from .object_detector import Detection


class DepthFusion:
    """Fuses RealSense depth data with YOLO detections to compute real-world
    distances in meters for each detected object."""

    def __init__(self, config: Optional[SDVConfig] = None):
        cfg = (config or SDVConfig.get()).depth
        self.min_dist = cfg['min_distance']
        self.max_dist = cfg['max_distance']
        self.sample_ratio = cfg['sample_ratio']
        self.scale = cfg['scale_factor']

    def fuse(
        self,
        detections: List[Detection],
        depth_image: np.ndarray,
    ) -> List[Detection]:
        """Add distance_m to each detection using the depth image.

        Args:
            detections: List of Detection objects from ObjectDetector.
            depth_image: Raw RealSense depth image (16UC1, values in mm
                         by default, or already scaled depending on config).

        Returns:
            Same list of detections with distance_m populated.
        """
        if depth_image is None or depth_image.size == 0:
            return detections

        dh, dw = depth_image.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Clamp bbox to depth image dimensions
            x1c = max(0, min(x1, dw - 1))
            y1c = max(0, min(y1, dh - 1))
            x2c = max(0, min(x2, dw - 1))
            y2c = max(0, min(y2, dh - 1))

            if x2c <= x1c or y2c <= y1c:
                continue

            # Sample the center region (avoid edge noise)
            bw = x2c - x1c
            bh = y2c - y1c
            pad_x = int(bw * (1 - self.sample_ratio) / 2)
            pad_y = int(bh * (1 - self.sample_ratio) / 2)

            roi = depth_image[
                y1c + pad_y : y2c - pad_y,
                x1c + pad_x : x2c - pad_x,
            ]

            if roi.size == 0:
                continue

            # Convert to float meters
            roi_m = roi.astype(np.float64) * self.scale

            # Filter valid depths
            valid = roi_m[(roi_m > self.min_dist) & (roi_m < self.max_dist)]

            if valid.size > 0:
                det.distance_m = float(np.median(valid))

        return detections
