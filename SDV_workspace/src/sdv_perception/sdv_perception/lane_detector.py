#!/usr/bin/env python3
"""
SDV Lane Detector — Bird's-eye perspective transform + polynomial fit lane detection.
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple

from .config import SDVConfig


class LaneDetector:
    """Advanced lane detection: HLS color filtering, perspective transform,
    sliding-window polynomial fit, temporal smoothing, overlay drawing."""

    def __init__(self, width: int = None, height: int = None,
                 config: Optional[SDVConfig] = None):
        cfg_obj = config or SDVConfig.get()
        cam = cfg_obj.camera
        lcfg = cfg_obj.lane

        self.width = width or cam['width']
        self.height = height or cam['height']
        self.roi_top = int(self.height * lcfg['roi_top_ratio'])

        # Color thresholds (HLS)
        self.white_low = np.array(lcfg['white_low'])
        self.white_high = np.array(lcfg['white_high'])
        self.yellow_low = np.array(lcfg['yellow_low'])
        self.yellow_high = np.array(lcfg['yellow_high'])

        # Edge detection
        self.blur_kernel = tuple(lcfg['blur_kernel'])
        self.canny_low = lcfg['canny_low']
        self.canny_high = lcfg['canny_high']

        # Sliding window
        sw = lcfg['sliding_window']
        self.n_windows = sw['n_windows']
        self.margin = sw['margin']
        self.min_pixels = sw['min_pixels']

        # Perspective transform
        p = lcfg['perspective']
        src = np.float32([
            [self.width * p['src_bottom_left'][0],
             self.height * p['src_bottom_left'][1]],
            [self.width * p['src_top_left'][0],
             self.height * p['src_top_left'][1]],
            [self.width * p['src_top_right'][0],
             self.height * p['src_top_right'][1]],
            [self.width * p['src_bottom_right'][0],
             self.height * p['src_bottom_right'][1]],
        ])
        dst = np.float32([
            [self.width * p['dst_left'], self.height],
            [self.width * p['dst_left'], 0],
            [self.width * p['dst_right'], 0],
            [self.width * p['dst_right'], self.height],
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        # Temporal smoothing
        self.left_fit_hist = deque(maxlen=lcfg['fit_history_length'])
        self.right_fit_hist = deque(maxlen=lcfg['fit_history_length'])
        self.lane_center_hist = deque(maxlen=lcfg['center_history_length'])

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Extract white + yellow lane markings with adaptive illumination compensation."""
        # 1. Adaptive Brightness Equalization (CLAHE on L channel)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        hls_eq = cv2.merge((h, l_eq, s))

        # 2. Dynamic thresholds based on overall image brightness
        mean_brightness = np.mean(l_eq)
        brightness_compensation = int(mean_brightness - 128) // 4
        
        white_low = self.white_low.copy()
        white_low[1] = max(150, self.white_low[1] + brightness_compensation)

        white_mask = cv2.inRange(hls_eq, white_low, self.white_high)
        yellow_mask = cv2.inRange(hls_eq, self.yellow_low, self.yellow_high)

        # 3. Morphological operations to clean up noise
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        blurred = cv2.GaussianBlur(cleaned, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges

    def get_birdseye(self, edges: np.ndarray) -> np.ndarray:
        """Warp edge image to bird's-eye view."""
        return cv2.warpPerspective(
            edges, self.M, (self.width, self.height)
        )

    def find_lane_pixels(self, birdseye: np.ndarray):
        """Sliding window search for left and right lane pixels."""
        histogram = np.sum(birdseye[birdseye.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        win_height = birdseye.shape[0] // self.n_windows
        nonzero = birdseye.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_x_curr = left_base
        right_x_curr = right_base
        left_inds = []
        right_inds = []

        for window in range(self.n_windows):
            win_y_low = birdseye.shape[0] - (window + 1) * win_height
            win_y_high = birdseye.shape[0] - window * win_height

            good_left = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                & (nonzero_x >= left_x_curr - self.margin)
                & (nonzero_x < left_x_curr + self.margin)
            ).nonzero()[0]

            good_right = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                & (nonzero_x >= right_x_curr - self.margin)
                & (nonzero_x < right_x_curr + self.margin)
            ).nonzero()[0]

            left_inds.append(good_left)
            right_inds.append(good_right)

            if len(good_left) > self.min_pixels:
                left_x_curr = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > self.min_pixels:
                right_x_curr = int(np.mean(nonzero_x[good_right]))

        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)

        return (
            nonzero_x[left_inds], nonzero_y[left_inds],
            nonzero_x[right_inds], nonzero_y[right_inds],
        )

    def fit_polynomial(self, left_x, left_y, right_x, right_y):
        """Fit 2nd-degree polynomial with temporal smoothing."""
        left_fit = right_fit = None

        if len(left_x) > 50:
            fit = np.polyfit(left_y, left_x, 2)
            self.left_fit_hist.append(fit)
        if len(self.left_fit_hist):
            left_fit = np.mean(self.left_fit_hist, axis=0)

        if len(right_x) > 50:
            fit = np.polyfit(right_y, right_x, 2)
            self.right_fit_hist.append(fit)
        if len(self.right_fit_hist):
            right_fit = np.mean(self.right_fit_hist, axis=0)

        return left_fit, right_fit

    def compute_steering(self, left_fit, right_fit) -> float:
        """Compute normalized lane center offset (- = left, + = right) with Low-Pass Filtering."""
        y_eval = self.height
        lane_center = self.width / 2

        if left_fit is not None and right_fit is not None:
            left_x = (left_fit[0] * y_eval**2
                      + left_fit[1] * y_eval + left_fit[2])
            right_x = (right_fit[0] * y_eval**2
                       + right_fit[1] * y_eval + right_fit[2])
            lane_center = (left_x + right_x) / 2

        raw_offset = (self.width / 2 - lane_center) / (self.width / 2)
        
        # Apply Low-Pass Filter (Exponential Moving Average) to prevent jerky steering
        alpha = 0.3  # Smoothing factor (lower = smoother but more delay)
        if len(self.lane_center_hist) > 0:
            last_offset = self.lane_center_hist[-1]
            smoothed_offset = alpha * raw_offset + (1 - alpha) * last_offset
        else:
            smoothed_offset = raw_offset
            
        self.lane_center_hist.append(smoothed_offset)
        return float(smoothed_offset)

    def draw_lane_overlay(
        self, frame: np.ndarray, left_fit, right_fit
    ) -> Tuple[np.ndarray, float]:
        """Draw filled lane polygon back on the original frame."""
        if left_fit is None or right_fit is None:
            return frame, 0.0

        overlay = np.zeros_like(frame)
        plot_y = np.linspace(0, self.height - 1, self.height)

        left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_x = (right_fit[0] * plot_y**2
                    + right_fit[1] * plot_y + right_fit[2])

        pts_left = np.array(
            [np.transpose(np.vstack([left_x, plot_y]))]
        )
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_x, plot_y])))]
        )
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(overlay, np.int_([pts]), (0, 200, 0))
        unwarped = cv2.warpPerspective(
            overlay, self.M_inv, (self.width, self.height)
        )

        result = cv2.addWeighted(frame, 1, unwarped, 0.35, 0)
        offset = self.compute_steering(left_fit, right_fit)

        # Steering arrow
        cx = int(self.width / 2)
        cy = int(self.height * 0.85)
        tx = int(cx - offset * 150)
        cv2.arrowedLine(result, (cx, cy), (tx, cy),
                        (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(result, f'Offset: {offset:+.3f}', (cx - 80, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return result, offset

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool, bool]:
        """Full lane detection pipeline.

        Returns:
            annotated_frame, steering_offset, left_detected, right_detected
        """
        edges = self.preprocess(frame)
        birdseye = self.get_birdseye(edges)
        lx, ly, rx, ry = self.find_lane_pixels(birdseye)
        left_fit, right_fit = self.fit_polynomial(lx, ly, rx, ry)
        result, offset = self.draw_lane_overlay(frame, left_fit, right_fit)
        return result, offset, left_fit is not None, right_fit is not None
