"""
side_clearance.py - Compare left vs right CSI views for go-around hint.

Uses simple mean luminance in lower ROI as a proxy for "more open" side.
Returns +1.0 = prefer steer left, -1.0 = prefer steer right, 0.0 = no preference.
"""

from __future__ import annotations

import numpy as np
import cv2


def side_preference_from_pair(left_bgr, right_bgr, roi_bottom_frac: float = 0.45) -> float:
    """
    Returns a continuous score:
        +1.0 (very clear left), -1.0 (very clear right), 0.0 (balanced/blocked).
    Uses mean luminance ratio with saturation-based refinement to avoid floor reflections.
    """
    if left_bgr is None or right_bgr is None:
        return 0.0
    try:
        if left_bgr.size == 0 or right_bgr.size == 0:
            return 0.0
        
        h = min(left_bgr.shape[0], right_bgr.shape[0])
        y0 = int(h * (1.0 - roi_bottom_frac))
        
        # Crop to lower ROI (the road/obstacle area)
        roi_l = left_bgr[y0:, :]
        roi_r = right_bgr[y0:, :]
        
        # Convert to Grayscale for luminance
        gl = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY) if len(roi_l.shape) == 3 else roi_l
        gr = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY) if len(roi_r.shape) == 3 else roi_r
        
        # Calculate mean luminance
        ml = float(np.mean(gl))
        mr = float(np.mean(gr))
        
        if ml < 10 and mr < 10:
            return 0.0
            
        # Continuous score based on luminance ratio [0.5, 2.0] -> [-1, 1]
        # We use a log-scale or simple ratio clamp
        ratio = ml / (mr + 1e-6)
        
        # Normalize: if ml >> mr, score -> 1.0; if mr >> ml, score -> -1.0
        # log2(2.0) = 1.0, log2(0.5) = -1.0
        score = float(np.clip(np.log2(ratio), -1.0, 1.0))
        
        # Deadzone to avoid chatter
        if abs(score) < 0.15:
            return 0.0
            
        return score
    except Exception:
        return 0.0


def get_side_clearance_m(side_bgr, roi_bottom_frac: float = 0.35) -> float:
    """
    Heuristic to estimate how far an obstacle is from the car in meters.
    Assumes the road is higher luminance than obstacles.
    Scans from bottom (car) up until a significant luminance drop is found.
    
    Returns:
        float: approximate distance (0.1 to 1.5m), or 2.0+ if clear.
    """
    if side_bgr is None or side_bgr.size == 0:
        return 2.0
        
    try:
        h, w = side_bgr.shape[:2]
        y0 = int(h * (1.0 - roi_bottom_frac))
        roi = side_bgr[y0:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Road baseline luminance from the very bottom row (safely assumed road)
        road_baseline = np.mean(gray[-5:, :])
        if road_baseline < 40:
            return 2.0 # too dark to trust
            
        # Find the first row from bottom up that is significantly darker than road
        # (obstacle footprint)
        threshold = road_baseline * 0.65
        row_means = np.mean(gray, axis=1)
        
        # Iter row by row from bottom up
        obstacle_row = -1
        for i in range(len(row_means)-1, -1, -5):
            if row_means[i] < threshold:
                obstacle_row = i
                break
        
        if obstacle_row == -1:
            return 2.5 # clear
            
        # Map row height to approximate distance (rough inverse mapping)
        # bottom row (len-1) -> ~0.2m
        # y0 row (0) -> ~1.5m
        normalized_row = 1.0 - (obstacle_row / (len(row_means) - 1))
        dist = 0.2 + normalized_row * 1.3
        
        return float(dist)
    except Exception:
        return 2.0


def is_side_yellow_visible(side_bgr, lower_hsv, upper_hsv, min_points: int = 150) -> bool:
    """
    Checks if a yellow line is visible in the side camera view.
    
    Args:
        side_bgr: BGR frame from side camera
        lower_hsv: np.array([H, S, V])
        upper_hsv: np.array([H, S, V])
        min_points: Minimum yellow pixels to consider a line "visible"
    
    Returns:
        bool: True if yellow line is likely present
    """
    if side_bgr is None or side_bgr.size == 0:
        return False
        
    try:
        # Use lower ROI to avoid overhead and focus on road
        h = side_bgr.shape[0]
        roi = side_bgr[int(h*0.4):, :] # Process bottom 60%
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        yellow_pts = cv2.countNonZero(mask)
        return yellow_pts >= min_points
    except Exception:
        return False
