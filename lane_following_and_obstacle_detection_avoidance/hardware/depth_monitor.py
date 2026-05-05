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


@dataclass
class DepthReading:
    is_clear: bool
    min_depth_m: float
    confidence: float


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
            if depth is None:
                return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=0.0)

            w = depth.get_width()
            h = depth.get_height()

            roi_w = max(1, int(w * float(self.cfg.depth_roi_width_frac)))
            roi_h = max(1, int(h * float(self.cfg.depth_roi_height_frac)))
            x0 = (w - roi_w) // 2
            y0 = (h - roi_h) // 2

            valid = 0
            total = roi_w * roi_h
            min_m = None

            # Iterate ROI (keep pure python; ROI is small by design)
            for yy in range(y0, y0 + roi_h):
                for xx in range(x0, x0 + roi_w):
                    d = depth.get_distance(xx, yy)  # meters
                    if d and d > 0.0:
                        valid += 1
                        if (min_m is None) or (d < min_m):
                            min_m = d

            confidence = (valid / total) if total > 0 else 0.0
            if min_m is None:
                return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=confidence)

            is_clear = not (min_m < float(self.cfg.depth_stop_distance_m))
            return DepthReading(is_clear=is_clear, min_depth_m=float(min_m), confidence=float(confidence))

        except Exception:
            # If depth glitches, treat as low-confidence clear (LiDAR remains primary)
            return DepthReading(is_clear=True, min_depth_m=-1.0, confidence=0.0)

    def terminate(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("[DepthMonitor] ✓ RealSense stopped.")
            except Exception:
                pass
