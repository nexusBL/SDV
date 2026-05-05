"""
depth_monitor.py - RealSense Depth-Based Obstacle Signal
=======================================================
Provides a simple, robust depth-derived obstacle check intended for fusion
with LiDAR. Designed to be conservative under poor depth validity (e.g.
shiny/reflective surfaces).

Public API:
  - initialize()
  - get_obstacle() -> (is_clear: bool, min_depth_m: float, confidence: float)
  - terminate()

confidence is the fraction of valid depth pixels in the ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DepthReading:
    is_clear: bool
    min_depth_m: float
    confidence: float
    ir_frame: np.ndarray | None = None


class DepthMonitor:
    def __init__(self, config):
        self.cfg = config.safety
        self._mock_mode = False
        self._rs = None
        self._pipeline = None
        self._profile = None
        self._depth_scale = 0.001  # meters per unit (fallback)

    def initialize(self):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception:
            self._mock_mode = True
            print("[DepthMonitor] MOCK mode - pyrealsense2 not available (obstacle fusion = LiDAR-only).")
            return

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

        try:
            self._profile = self._pipeline.start(cfg)
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            print(f"[DepthMonitor] ✓ RealSense depth active — fused with LiDAR in tessting.py (scale={self._depth_scale} m/unit).")
        except Exception as e:
            self._mock_mode = True
            print(f"[DepthMonitor] Warning: failed to start RealSense ({e}). Using MOCK mode.")

    def get_obstacle(self) -> DepthReading:
        if self._mock_mode or self._pipeline is None:
            return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=0.0)

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=50)
            depth = frames.get_depth_frame()
            ir = frames.get_infrared_frame(1)
            
            ir_np = np.asanyarray(ir.get_data()) if ir is not None else None

            if depth is None:
                return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=0.0, ir_frame=ir_np)

            w = depth.get_width()
            h = depth.get_height()

            roi_w = max(1, int(w * float(self.cfg.depth_roi_width_frac)))
            roi_h = max(1, int(h * float(self.cfg.depth_roi_height_frac)))
            x0 = (w - roi_w) // 2
            y0 = (h - roi_h) // 2

            # Vectorized numpy extraction
            depth_data = np.asanyarray(depth.get_data())
            roi = depth_data[y0:y0+roi_h, x0:x0+roi_w] * self._depth_scale
            
            # Filter zero (invalid) and out-of-range points
            valid_mask = (roi > 0.01) & (roi < 3.0)
            valid = int(np.sum(valid_mask))
            total = roi_w * roi_h
            
            confidence = (valid / total) if total > 0 else 0.0
            
            if valid == 0:
                return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=confidence, ir_frame=ir_np)
                
            min_m = float(np.min(roi[valid_mask]))
            is_clear = not (min_m < float(self.cfg.depth_stop_distance_m))
            return DepthReading(is_clear=is_clear, min_depth_m=min_m, confidence=confidence, ir_frame=ir_np)

        except Exception:
            # If depth glitches, treat as low-confidence clear (LiDAR remains primary)
            return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=0.0, ir_frame=None)

    def terminate(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("[DepthMonitor] ✓ RealSense stopped.")
            except Exception:
                pass
