#!/usr/bin/env python3
"""
SDV Visualization — Market-level HUD overlay for the perception pipeline.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

from .config import SDVConfig
from .object_detector import Detection


class HUDRenderer:
    """Draws a market-level heads-up display on the perception output frame."""

    _THREAT_COLORS = {
        'CLEAR':    (0, 220, 0),
        'WARNING':  (0, 165, 255),
        'CRITICAL': (0, 0, 255),
        'UNKNOWN':  (128, 128, 128),
    }

    def __init__(self, config: Optional[SDVConfig] = None):
        cfg = (config or SDVConfig.get()).hud
        self.bar_h = cfg['top_bar_height']
        self.font_title = cfg['font_scale_title']
        self.font_info = cfg['font_scale_info']
        self.font_panel = cfg['font_scale_panel']
        self.threat_w = cfg['threat_panel_width']
        self.threat_h = cfg['threat_panel_height']
        self.lane_w = cfg['lane_panel_width']
        self.lane_h = cfg['lane_panel_height']
        self.steer_half = cfg['steering_bar_half_width']
        self.bg = tuple(cfg['background_color'])

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        lane_offset: float,
        lane_status: str,
        threat: str,
        min_dist: float,
        zones: Dict[str, float],
        latency_ms: float,
        fps: float,
    ) -> np.ndarray:
        """Draw complete HUD overlay on the frame. Modifies in-place and returns it."""
        h, w = frame.shape[:2]

        self._draw_top_bar(frame, w, detections, latency_ms, fps)
        self._draw_threat_panel(frame, h, threat, min_dist, zones)
        self._draw_lane_panel(frame, w, h, lane_offset, lane_status)
        self._draw_steering_bar(frame, w, h, lane_offset)
        self._draw_threat_border(frame, w, h, threat)

        return frame

    def _draw_top_bar(self, frame, w, detections, latency_ms, fps):
        cv2.rectangle(frame, (0, 0), (w, self.bar_h), self.bg, -1)
        cv2.putText(
            frame, 'SDV PERCEPTION v2.0', (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_title, (0, 200, 255), 2,
        )
        info = (f'FPS: {fps:.1f}  |  Latency: {latency_ms:.1f}ms'
                f'  |  Objects: {len(detections)}')
        cv2.putText(
            frame, info, (w - 450, 26),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_info, (200, 200, 200), 1,
        )

    def _draw_threat_panel(self, frame, h, threat, min_dist, zones):
        tc = self._THREAT_COLORS.get(threat, (128, 128, 128))
        cv2.rectangle(
            frame, (0, h - self.threat_h), (self.threat_w, h), self.bg, -1
        )
        cv2.putText(
            frame, 'LIDAR THREAT', (10, h - self.threat_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )
        cv2.putText(
            frame, threat, (10, h - self.threat_h + 52),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, tc, 3,
        )
        cv2.putText(
            frame, f'Front: {min_dist:.2f}m', (10, h - self.threat_h + 80),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_panel, (200, 200, 200), 1,
        )
        left_d = zones.get('left', 0)
        right_d = zones.get('right', 0)
        cv2.putText(
            frame, f'L:{left_d:.1f}m  R:{right_d:.1f}m',
            (10, h - self.threat_h + 102),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_panel, (200, 200, 200), 1,
        )

    def _draw_lane_panel(self, frame, w, h, lane_offset, lane_status):
        lane_color = (0, 220, 0) if lane_status == 'BOTH' else (0, 165, 255)
        cv2.rectangle(
            frame, (w - self.lane_w, h - self.lane_h), (w, h), self.bg, -1
        )
        cv2.putText(
            frame, 'LANE', (w - self.lane_w + 10, h - self.lane_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )
        cv2.putText(
            frame, lane_status, (w - self.lane_w + 10, h - self.lane_h + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, lane_color, 2,
        )
        cv2.putText(
            frame, f'Offset: {lane_offset:+.3f}',
            (w - self.lane_w + 10, h - self.lane_h + 72),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_panel, (200, 200, 200), 1,
        )

    def _draw_steering_bar(self, frame, w, h, lane_offset):
        cx = w // 2
        bar_y = h - 20
        cv2.rectangle(
            frame,
            (cx - self.steer_half, bar_y - 8),
            (cx + self.steer_half, bar_y + 8),
            (40, 40, 40), -1,
        )
        indicator_x = int(cx + lane_offset * self.steer_half)
        indicator_x = max(cx - self.steer_half + 2,
                          min(cx + self.steer_half - 2, indicator_x))
        cv2.rectangle(
            frame,
            (indicator_x - 6, bar_y - 10),
            (indicator_x + 6, bar_y + 10),
            (0, 255, 255), -1,
        )
        cv2.line(frame, (cx, bar_y - 12), (cx, bar_y + 12),
                 (100, 100, 100), 1)

    def _draw_threat_border(self, frame, w, h, threat):
        if threat == 'CRITICAL':
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        elif threat == 'WARNING':
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
