"""
qcar2_lane_detection.py
=======================
Clean, standalone lane detection + path following for the Quanser QCar2 (Physical Hardware).

Source: Best-of extraction from Repos 11 (BIUST/sbuda47) and 12 (StengerJ/SelfdrivingCar2026),
        confirmed working on real QCar2 hardware (skyview logs from Feb 2026).

Requirements:
    - Quanser HAL/PAL libraries installed (pal, hal packages)
    - pip install numpy opencv-python

Usage:
    python3 qcar2_lane_detection.py
    python3 qcar2_lane_detection.py --nodes 10,2,4,14,10   # follow a node route
    python3 qcar2_lane_detection.py --cruise-speed 0.4      # faster
    python3 qcar2_lane_detection.py --left-hand             # left-hand traffic map

Tuning guide (printed at bottom of this file).
"""

# ── MUST be before ANY other import ──────────────────────────────────
# Fixes "No EGL Display / nvbufsurftransform: Could not get EGL display"
# error on physical QCar2 (Jetson) when running headless (no monitor).
import os, sys
os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)

# Add Quanser library path (matches your QCar2 setup)
_QUANSER_LIB = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.isdir(_QUANSER_LIB) and _QUANSER_LIB not in sys.path:
    sys.path.insert(0, _QUANSER_LIB)
# ─────────────────────────────────────────────────────────────────────

import argparse
import signal
import time

import cv2
import numpy as np

# --- Quanser imports ---
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarCameras, QCarGPS
from pal.utilities.math import Filter, wrap_to_pi

# ============================================================
# SECTION 1 — LANE DETECTION
# ============================================================

def _make_lowpass_filter(cutoff_hz: float, dt: float):
    """
    Creates a generator-based first-order low-pass filter.
    Send (new_value, dt) to get filtered output.
    This is critical on physical hardware where camera vibration
    and lighting changes cause noisy steering commands.
    """
    rc = 1.0 / (2.0 * np.pi * max(cutoff_hz, 0.1))
    y = 0.0

    def _filt():
        nonlocal y, rc
        while True:
            new_val, dt_in = yield y
            alpha = float(dt_in) / (rc + float(dt_in))
            y = alpha * float(new_val) + (1.0 - alpha) * y

    gen = _filt()
    next(gen)  # prime the generator
    return gen


def compute_ego_path(
    front_bgr,
    binary,
    roi_offsets: tuple,
    left_hand_traffic: bool = False,
    lane_offset_ratio: float = 0.55,
    draw: bool = True,
):
    """
    Computes and optionally draws the ego path on the camera frame.

    The ego path is the predicted trajectory of the car center — offset
    from the detected yellow line by one lane width to the right
    (for left-hand traffic) or left (right-hand traffic).

    Args:
        front_bgr:         Original BGR camera frame (will be annotated in-place).
        binary:            Binary yellow mask (output of HSV threshold on ROI).
        roi_offsets:       (row_start, row_end, col_start, col_end) of the ROI.
        left_hand_traffic: True for left-hand traffic.
        lane_offset_ratio: Fraction of ROI width to offset ego path from yellow line.
                           Default 0.55 (car drives ~55% of ROI width from yellow line).
        draw:              If True, draws cyan lane line + green ego path on front_bgr.

    Returns:
        dict with keys:
            'steering'   : float steering estimate from ego path [rad]
            'valid'      : bool — True if lane was detected
            'ego_pts'    : list of (x,y) pixel points along ego path
            'lane_pts'   : list of (x,y) pixel points along yellow line fit
            'poly'       : numpy polynomial coefficients [a, b, c] or None
    """
    row_start, row_end, col_start, col_end = roi_offsets
    h, w = front_bgr.shape[:2]
    roi_h = row_end - row_start

    rows_idx, cols_idx = np.where(binary > 0)
    roi_w = col_end - col_start

    result = {'steering': 0.0, 'valid': False,
              'ego_pts': [], 'lane_pts': [], 'poly': None}

    if len(rows_idx) < 30:
        if draw:
            cv2.putText(front_bgr, "NO LANE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    # ── FIX 1: Linear fit (degree=1) — stable, no wild extrapolation ─
    poly = np.polyfit(rows_idx, cols_idx, 1)
    result['poly'] = poly

    # ── FIX 2: Only evaluate where yellow pixels actually exist ───────
    y_min  = int(rows_idx.min())
    y_max  = int(rows_idx.max())
    y_vals = np.linspace(y_min, y_max, 50).astype(int)

    # FIX 3: Clip x to ROI bounds — no out-of-frame coordinates
    x_lane = np.clip(np.polyval(poly, y_vals).astype(int), 0, roi_w - 1)

    # FIX 4: Smaller offset (28% of ROI width, not 55%)
    offset_px = int(roi_w * min(lane_offset_ratio, 0.30))
    if left_hand_traffic:
        x_ego = np.clip(x_lane + offset_px, 0, w - col_start - 1)
    else:
        x_ego = np.clip(x_lane - offset_px, 0, w - col_start - 1)

    # ── Convert ROI-relative coords to full-image coords ──────────────
    lane_pts = [(int(x) + col_start, int(y) + row_start) for x, y in zip(x_lane, y_vals)]
    ego_pts  = [(int(x) + col_start, int(y) + row_start) for x, y in zip(x_ego,  y_vals)]

    # Filter to valid frame bounds
    lane_pts = [(x, y) for x, y in lane_pts if 0 <= x < w and 0 <= y < h]
    ego_pts  = [(x, y) for x, y in ego_pts  if 0 <= x < w and 0 <= y < h]

    result['valid']     = True
    result['lane_pts']  = lane_pts
    result['ego_pts']   = ego_pts

    # ── Steering estimate from ego path bottom point vs image center ──
    if ego_pts:
        ego_bottom_x    = ego_pts[-1][0]
        img_cx          = w // 2
        error_norm      = (ego_bottom_x - img_cx) / float(w)
        result['steering'] = float(np.clip(-0.45 * error_norm, -0.52, 0.52))

    # ── Draw on frame ─────────────────────────────────────────────────
    if draw and lane_pts and ego_pts:
        h_f, w_f = front_bgr.shape[:2]

        # 1. Detected yellow pixels → bright yellow highlight
        ry, cx2 = np.where(binary > 0)
        if len(ry):
            px_overlay = np.zeros((row_end - row_start,
                                   col_end - col_start, 3), dtype=np.uint8)
            px_overlay[ry, cx2] = [0, 220, 255]
            front_bgr[row_start:row_end, col_start:col_end] = cv2.addWeighted(
                front_bgr[row_start:row_end, col_start:col_end], 1.0,
                px_overlay, 0.9, 0)

        # 2. Corridor filled polygon (45% opacity)
        n = min(len(lane_pts), len(ego_pts))
        corridor = lane_pts[:n] + list(reversed(ego_pts[:n]))
        if len(corridor) > 3:
            ov2 = front_bgr.copy()
            cv2.fillPoly(ov2, [np.array(corridor, dtype=np.int32)], (0, 200, 60))
            front_bgr[:] = cv2.addWeighted(front_bgr, 0.55, ov2, 0.45, 0)

        # 3. Lane fit — thick cyan 6px + dot markers
        for i in range(len(lane_pts) - 1):
            cv2.line(front_bgr, lane_pts[i], lane_pts[i+1], (0, 255, 255), 6)
        for i in range(0, len(lane_pts), 6):
            cv2.circle(front_bgr, lane_pts[i], 5, (0, 255, 255), -1)

        # 4. Ego path — thick green 6px + dot markers
        for i in range(len(ego_pts) - 1):
            cv2.line(front_bgr, ego_pts[i], ego_pts[i+1], (0, 255, 0), 6)
        for i in range(0, len(ego_pts), 6):
            cv2.circle(front_bgr, ego_pts[i], 5, (0, 255, 0), -1)

        # 5. Endpoint markers
        cv2.circle(front_bgr, lane_pts[-1], 10, (0, 255, 255), -1)
        cv2.circle(front_bgr, ego_pts[-1],  16, (0, 255,   0), -1)
        cv2.circle(front_bgr, ego_pts[-1],   8, (255,255,255), -1)
        cv2.putText(front_bgr, "EGO",
                    (ego_pts[-1][0]+20, ego_pts[-1][1]+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. Steering arrow at bottom center
        steer     = result['steering']
        arrow_cx  = w_f // 2
        arrow_cy  = h_f - 30
        arrow_len = int(abs(steer) * 300)
        arrow_dx  = int(np.sign(-steer) * arrow_len)
        s_col = (0,255,0) if abs(steer)<0.05 else \
                (0,200,255) if abs(steer)<0.15 else (0,80,255)
        if arrow_len > 5:
            cv2.arrowedLine(front_bgr,
                (arrow_cx, arrow_cy), (arrow_cx+arrow_dx, arrow_cy),
                s_col, 4, tipLength=0.3)
        cv2.circle(front_bgr, (arrow_cx, arrow_cy), 8, (255,255,255), -1)

        # 7. HUD panel
        direction = "STRAIGHT" if abs(steer)<0.03 \
            else (">> RIGHT" if steer>0 else "LEFT <<")
        cv2.rectangle(front_bgr, (0,0), (w_f, 90), (20,20,20), -1)
        cv2.putText(front_bgr, f"Steering: {steer:+.4f} rad   {direction}",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, s_col, 2)
        cv2.putText(front_bgr,
                    f"Yellow px: {len(ry) if len(ry) else 0}  "
                    f"ROI: full-width  rows:{row_start}-{row_end}",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # 8. Legend
        lh = front_bgr.shape[0]
        cv2.rectangle(front_bgr, (10, lh-72), (345, lh-4), (20,20,20), -1)
        for i, (txt, clr) in enumerate([
            ("Bright yellow = detected pixels",  (0,220,255)),
            ("Cyan line     = yellow lane fit",   (0,255,255)),
            ("Green line    = ego path",           (0,255,  0)),
        ]):
            cv2.putText(front_bgr, txt, (14, lh-52+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1)

    return result


def compute_lane_steering(
    front_bgr,
    dt: float,
    steering_filter,
    lane_steer_limit: float = 0.5,
    left_hand_traffic: bool = False,
    yellow_line_target_ratio: float = 0.30,
    yellow_line_distance_gain: float = 0.45,
    yellow_line_bottom_roi: float = 0.55,
    draw_ego_path: bool = True,
    lane_offset_ratio: float = 0.55,
):
    """
    Detects the yellow lane line and returns a steering correction.

    How it works:
        1. Crops a horizontal ROI from the lower-middle of the camera frame
           (rows ~64-82% of height, right half of width for RH traffic).
        2. Thresholds in HSV for yellow (hue 10-45, sat 50-255, val 100-255).
        3. Computes slope+intercept of the yellow pixels via Quanser's HAL function.
           → Converts to a heading correction: 1.5*(slope - 0.3419) + (1/150)*(intercept + 5)
        4. Also measures the lateral position of the yellow line near the bottom
           of the ROI and corrects cross-track error:
           → steering -= distance_gain * (line_x - target_x) / width
        5. Clips and low-pass filters the result.

    Args:
        front_bgr:               Camera frame in BGR (from QCar2 CSI camera).
        dt:                      Time since last call in seconds.
        steering_filter:         Low-pass filter generator (from _make_lowpass_filter).
        lane_steer_limit:        Max steering output magnitude [rad]. Default 0.5.
        left_hand_traffic:       Set True if driving on the left side of road.
        yellow_line_target_ratio: Where yellow line should sit horizontally (0=left, 1=right).
                                  Default 0.30 (yellow line 30% from left = right-hand traffic).
        yellow_line_distance_gain: How aggressively to correct lateral drift. Default 0.45.
                                   Increase if car drifts away from yellow line.
                                   Decrease if steering oscillates.
        yellow_line_bottom_roi:  Fraction of ROI height to use for lateral measurement.
                                  Default 0.55 (bottom 45% of ROI).

    Returns:
        float steering correction [rad], or None if no yellow line detected.
        Also draws ego path on front_bgr in-place if draw_ego_path=True.
    """
    h, w = front_bgr.shape[:2]

    # --- ROI: full width, bottom 68-95% — road only, no car body ---
    row_start = int(np.clip(round(0.68 * h), 0, h - 1))
    row_end   = int(np.clip(round(0.95 * h), row_start + 1, h))
    col_start = 0
    col_end   = w   # FULL WIDTH — yellow line at ~30% from left

    cropped = front_bgr[row_start:row_end, col_start:col_end]
    if cropped.size == 0:
        return None

    # --- Yellow HSV threshold (calibrated to exclude orange QCar2 body) ---
    # Hue 18-38:  starts ABOVE orange (car body hue ~10), catches true yellow
    # Sat 30-160: caps below car body saturation (~190), keeps dim lane markings
    # Val 60-210: captures road-level lighting conditions
    hsv_buf = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(
        hsv_buf,
        np.array([18, 30, 60],  dtype=np.uint8),   # ← TUNE hue lower if yellow missed
        np.array([38, 160, 210], dtype=np.uint8),   # ← TUNE sat upper if car still detected
    )
    # Morphological cleanup — removes small noise blobs
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel)

    # --- Extract pixel coordinates ONCE (used by both corrections below) ---
    rows_idx, cols_idx = np.where(binary > 0)

    # --- Early exit if no yellow pixels found ---
    if rows_idx.size < 30:
        return None   # no lane → caller falls back to map steering

    # --- Heading correction (linear fit of yellow pixels) ---
    poly1 = np.polyfit(rows_idx, cols_idx, 1)
    slope, intercept = float(poly1[0]), float(poly1[1])
    raw_steering = 1.5 * (slope - 0.3419) + (1.0 / 150.0) * (intercept + 5.0)

    # --- Lateral correction (cross-track error from yellow line position) ---
    rows, cols = rows_idx, cols_idx   # alias for clarity
    if rows.size > 20:
        cutoff_row = int(np.clip(
            round(yellow_line_bottom_roi * binary.shape[0]),
            0, binary.shape[0] - 1
        ))
        near_mask = rows >= cutoff_row
        if np.count_nonzero(near_mask) >= 10:
            line_x = float(np.median(cols[near_mask]))
        else:
            line_x = float(np.median(cols))

        target_x    = yellow_line_target_ratio * float(max(binary.shape[1] - 1, 1))
        x_error_norm = (line_x - target_x) / float(max(binary.shape[1], 1))
        raw_steering += -float(yellow_line_distance_gain) * x_error_norm

    # --- Clip and filter ---
    clipped = float(np.clip(raw_steering, -lane_steer_limit, lane_steer_limit))
    try:
        steering = float(steering_filter.send((clipped, dt)))
    except Exception:
        steering = clipped

    # --- Draw ego path on frame (in-place) ---
    if draw_ego_path:
        compute_ego_path(
            front_bgr=front_bgr,
            binary=binary,
            roi_offsets=(row_start, row_end, col_start, col_end),
            left_hand_traffic=left_hand_traffic,
            lane_offset_ratio=lane_offset_ratio,
            draw=True,
        )

    return None if np.isnan(steering) else steering


# ============================================================
# SECTION 2 — STANLEY STEERING CONTROLLER (for waypoint routes)
# ============================================================

class StanleySteeringController:
    """
    Stanley controller for path following using a waypoint sequence.
    Use this when you have a node route (e.g. from SDCSRoadMap).

    The output can be blended with lane steering — see main loop below.
    """

    def __init__(
        self,
        waypoints,
        k: float = 0.6,
        cyclic: bool = False,
        max_steer: float = 0.52,
        switch_distance: float = 0.5,
        search_window: int = 60,
    ):
        """
        Args:
            waypoints:       Shape (2, N) array of [x; y] waypoints.
            k:               Stanley gain. Higher = more aggressive cross-track correction.
                             Physical QCar2: start at 0.4-0.6.
            cyclic:          True to loop the route forever.
            max_steer:       Maximum steering angle [rad]. Default π/6 ≈ 0.52.
            switch_distance: How close (m) to advance to next waypoint. Default 0.5.
            search_window:   How many waypoints ahead/behind to search. Default 60.
        """
        self.maxSteeringAngle = float(max_steer)
        self.wp               = waypoints
        self.N                = int(waypoints.shape[1])
        self.wpi              = 0
        self.k                = float(k)
        self.cyclic           = bool(cyclic)
        self.switchDistance   = float(switch_distance)
        self.searchWindow     = max(10, int(search_window))
        self.pathComplete     = False

    def reanchor(self, p):
        """Snap waypoint index to nearest point. Call once after GPS fix."""
        if self.N < 2:
            self.pathComplete = True
            return
        p2  = np.asarray(p[:2], dtype=np.float64)
        pts = self.wp[:2, :].T
        self.wpi = int(np.argmin(np.linalg.norm(pts - p2, axis=1)))
        if not self.cyclic and self.wpi >= self.N - 1:
            self.wpi = self.N - 2
            self.pathComplete = True

    def _advance_to_local_nearest(self, p):
        if self.N < 2:
            return
        p2    = np.asarray(p[:2], dtype=np.float64)
        start = max(0, self.wpi - 2)
        end   = min(self.N, self.wpi + self.searchWindow)
        if end - start < 2:
            return
        local_pts   = self.wp[:2, start:end].T
        nearest_idx = start + int(np.argmin(np.linalg.norm(local_pts - p2, axis=1)))
        if nearest_idx > self.wpi:
            self.wpi = nearest_idx

    def _advance(self):
        if self.cyclic:
            self.wpi = int((self.wpi + 1) % max(self.N - 1, 1))
        elif self.wpi < self.N - 2:
            self.wpi += 1
        else:
            self.pathComplete = True

    def update(self, p, th: float, speed: float) -> float:
        """
        Args:
            p:     Current position [x, y] (front axle recommended: p + 0.2*[cos(th), sin(th)]).
            th:    Current heading [rad].
            speed: Current speed [m/s].

        Returns:
            Steering angle [rad], clipped to ±max_steer.
        """
        if self.N < 2 or (not self.cyclic and self.wpi >= self.N - 2):
            self.pathComplete = True
            return 0.0

        self._advance_to_local_nearest(p)

        i1 = int(self.wpi)
        i2 = int((i1 + 1) % max(self.N - 1, 1)) if self.cyclic else min(i1 + 1, self.N - 1)

        wp_1 = self.wp[:2, i1]
        wp_2 = self.wp[:2, i2]
        p2   = np.asarray(p[:2], dtype=np.float64)

        v_seg = wp_2 - wp_1
        v_mag = float(np.linalg.norm(v_seg))
        if v_mag < 1e-6:
            self._advance()
            return 0.0

        v_uv    = v_seg / v_mag
        tangent = float(np.arctan2(v_uv[1], v_uv[0]))
        s       = float(np.dot(p2 - wp_1, v_uv))

        if s >= v_mag or np.linalg.norm(p2 - wp_2) < min(v_mag, max(self.switchDistance, 0.08)):
            self._advance()
        elif np.dot(np.array([np.cos(th), np.sin(th)], dtype=np.float64), wp_2 - p2) < -0.1:
            self._advance()

        ep       = wp_1 + v_uv * np.clip(s, 0.0, v_mag)
        ct       = ep - p2
        side_dir = float(wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent))
        ect      = float(np.linalg.norm(ct) * np.sign(side_dir))
        psi      = float(wrap_to_pi(tangent - th))

        steering = psi + np.arctan2(self.k * ect, max(float(abs(speed)), 0.2))
        return float(np.clip(wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle))


# ============================================================
# SECTION 3 — SPEED CONTROLLER
# ============================================================

class SpeedController:
    """
    PI speed controller with anti-windup and anti-surge logic.
    Designed for QCar2 physical hardware — avoids jerky throttle changes.
    """

    def __init__(self, kp: float = 0.04, ki: float = 0.15, max_throttle: float = 0.14):
        """
        Args:
            kp:           Proportional gain. Default 0.04.
            ki:           Integral gain. Default 0.15.
            max_throttle: Maximum throttle command. Default 0.14 (conservative for physical).
                          Increase to 0.20 for faster driving, but test carefully.
        """
        self.kp          = float(kp)
        self.ki          = float(ki)
        self.maxThrottle = float(max_throttle)
        self.ei          = 0.0

    def reset(self):
        self.ei = 0.0

    def update(self, v: float, v_ref: float, dt: float) -> float:
        """
        Args:
            v:     Measured speed [m/s] (from qcar.motorTach).
            v_ref: Target speed [m/s].
            dt:    Time step [s].

        Returns:
            Throttle command [0, max_throttle].
        """
        e         = float(v_ref - v)
        self.ei  += max(float(dt), 1e-3) * e
        self.ei   = float(np.clip(self.ei, -0.2, 0.2))  # anti-windup
        if v_ref <= 0.01:
            self.ei *= 0.9   # bleed off integral when stopped
        u = self.kp * e + self.ki * self.ei
        return float(np.clip(u, 0.0, self.maxThrottle))


# ============================================================
# SECTION 4 — RATE LIMITER (smooths abrupt steering changes)
# ============================================================

def rate_limit(current: float, target: float, dt: float,
               up_rate: float, down_rate: float) -> float:
    """
    Limits how fast a value can change per second.
    Prevents sudden jerks in steering on physical hardware.

    Args:
        current:   Current value.
        target:    Desired value.
        dt:        Time step [s].
        up_rate:   Max increase per second [units/s].
        down_rate: Max decrease per second [units/s].
    """
    max_up   = max(float(up_rate),   1e-3) * float(dt)
    max_down = max(float(down_rate), 1e-3) * float(dt)
    if target >= current:
        return min(target, current + max_up)
    return max(target, current - max_down)


# ============================================================
# SECTION 5 — ARGUMENT PARSING
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="QCar2 physical lane detection + path following")
    p.add_argument("--nodes",          default="",    help="Comma-separated node route, e.g. '10,2,4,10'")
    p.add_argument("--cruise-speed",   type=float, default=0.35, help="Target speed [m/s]. Default 0.35")
    p.add_argument("--runtime",        type=float, default=300.0, help="Max runtime [s]. Default 300")
    p.add_argument("--start-delay",    type=float, default=1.0,   help="Seconds before moving. Default 1")
    p.add_argument("--max-throttle",   type=float, default=0.14,  help="Max throttle. Default 0.14")
    p.add_argument("--min-throttle",   type=float, default=0.065, help="Min effective throttle. Default 0.065")
    p.add_argument("--stanley-k",      type=float, default=0.5,   help="Stanley gain. Default 0.5")
    p.add_argument("--lane-weight",    type=float, default=0.45,  help="Lane steering weight [0-1]. Default 0.45")
    p.add_argument("--map-weight",     type=float, default=0.90,  help="Map/Stanley weight [0-1]. Default 0.90")
    p.add_argument("--lane-gain",      type=float, default=0.45,  help="Yellow line distance gain. Default 0.45")
    p.add_argument("--lane-limit",     type=float, default=0.5,   help="Max lane steering [rad]. Default 0.5")
    p.add_argument("--lane-cutoff-hz", type=float, default=25.0,  help="Lane filter cutoff [Hz]. Default 25")
    p.add_argument("--steer-rate",     type=float, default=0.95,  help="Max steering rate [rad/s]. Default 0.95")
    p.add_argument("--max-steer",      type=float, default=0.52,  help="Max steering angle [rad]. Default 0.52")
    p.add_argument("--left-hand",      action="store_true",        help="Left-hand traffic map")
    p.add_argument("--no-ego-path",    action="store_true",        help="Disable ego path drawing")
    p.add_argument("--lane-offset",    type=float, default=0.55,   help="Ego path offset ratio. Default 0.55")
    p.add_argument("--no-gps",         action="store_true",        help="Run without GPS (lane-only mode)")
    p.add_argument("--calibrate-gps",  action="store_true",        help="Recalibrate GPS on startup")
    p.add_argument("--sample-rate",    type=float, default=100.0,  help="Control loop rate [Hz]. Default 100")
    return p.parse_args()


# ============================================================
# SECTION 6 — MAIN CONTROL LOOP
# ============================================================

STOP_REQUESTED = False

def _sig_handler(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGINT,  _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def main():
    args = parse_args()

    dt        = 1.0 / max(args.sample_rate, 10.0)
    use_route = bool(args.nodes.strip())

    # --- Route setup ---
    waypoints      = None
    steer_ctrl     = None
    initial_pose   = [0.0, 0.0, 0.0]

    if use_route:
        node_list  = [int(n.strip()) for n in args.nodes.split(",") if n.strip()]
        roadmap    = SDCSRoadMap(leftHandTraffic=args.left_hand)
        waypoints  = roadmap.generate_path(node_list)
        init_pose  = roadmap.get_node_pose(node_list[0]).squeeze()
        initial_pose = [float(init_pose[0]), float(init_pose[1]), float(init_pose[2])]
        steer_ctrl = StanleySteeringController(
            waypoints=waypoints,
            k=args.stanley_k,
            cyclic=False,
            max_steer=args.max_steer,
        )
        print(f"[INFO] Route: {node_list} → {waypoints.shape[1]} waypoints")
    else:
        print("[INFO] No route specified → lane-follow only mode")

    # --- Controllers ---
    speed_ctrl = SpeedController(max_throttle=args.max_throttle)
    lane_filter = _make_lowpass_filter(args.lane_cutoff_hz, dt)

    # --- Hardware init ---
    qcar    = QCar(readMode=1, frequency=int(args.sample_rate))

    # Front CSI camera = index 2 on physical QCar2 (Right=0, Rear=1, Front=2, Left=3)
    # Resolution 820x616 matches your working CSI test script
    cameras = QCarCameras(
        frameWidth=820, frameHeight=616, frameRate=30,
        enableFront=True,    # Front camera (ID=2) for lane detection
        enableBack=False, enableLeft=False, enableRight=False,
    )
    print("[INFO] Warming up front CSI camera (2s)...")
    time.sleep(2.0)   # let nvarguscamerasrc settle before first read

    # ── Verify camera is working before starting control loop ─────────
    print("[INFO] Checking camera feed...")
    for _attempt in range(20):
        cameras.readAll()
        time.sleep(0.1)
        try:
            _img = cameras.csi[2].imageData
            if _img is not None and _img.max() > 10:
                cv2.imwrite("/tmp/qcar2_startup_frame.jpg", _img)
                print(f"[INFO] ✅ Camera OK — saved /tmp/qcar2_startup_frame.jpg")
                print(f"[INFO]    Frame: {_img.shape[1]}x{_img.shape[0]}  max_pixel={_img.max()}")
                break
        except Exception:
            pass
    else:
        print("[WARN] ❌ Camera blank after 20 attempts — lane detection will output None")
        print("[WARN]    Check: is front CSI camera (ID=2) connected and enabled?")

    gps = None
    ekf = None
    if use_route and not args.no_gps:
        gps = QCarGPS(initialPose=initial_pose, calibrate=args.calibrate_gps)
        ekf = QCarEKF(x_0=initial_pose)
        print("[INFO] GPS + EKF enabled")

    # --- State ---
    delta          = 0.0   # last steering command (for EKF)
    x, y, th       = initial_pose
    steer_anchored = False
    t0             = None

    print("[INFO] Starting in", args.start_delay, "seconds... Press Ctrl+C to stop.")

    with qcar, cameras:
        if gps:
            gps.__enter__()

        t0 = time.time()
        t_prev = t0
        last_print_t = -999.0   # ensures first print fires immediately

        while not STOP_REQUESTED:
            t_now = time.time()
            t  = t_now - t0
            dt_actual = float(np.clip(t_now - t_prev, 1e-4, 0.1))  # real elapsed, clamped
            t_prev = t_now

            # ── Read sensors ──────────────────────────────────────────
            qcar.read()
            v = float(qcar.motorTach)

            if ekf and gps:
                if gps.readGPS():
                    y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                    ekf.update([v, delta], dt_actual, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([v, delta], dt_actual, None, qcar.gyroscope[2])
                x  = float(ekf.x_hat[0, 0])
                y  = float(ekf.x_hat[1, 0])
                th = float(ekf.x_hat[2, 0])

            # ── Read camera ───────────────────────────────────────────
            # Front camera = csi[2] on physical QCar2
            # csiFront is an alias but csi[2] is always reliable
            cameras.readAll()
            frame = None
            try:
                img = cameras.csi[2].imageData
                if img is not None and img.max() > 10:   # not blank
                    frame = img
            except Exception:
                frame = None

            # ── Start delay ───────────────────────────────────────────
            if t < args.start_delay:
                qcar.write(0.0, 0.0)
                continue

            if t >= args.runtime:
                print("[INFO] Runtime reached.")
                break

            # ── Anchor Stanley to current position (once) ─────────────
            if steer_ctrl and not steer_anchored and ekf:
                steer_ctrl.reanchor([x, y])
                steer_anchored = True

            # ── Lane steering ─────────────────────────────────────────
            lane_steer = None
            if frame is not None:
                # Work on a copy so original frame stays clean
                display_frame = frame.copy()
                lane_steer = compute_lane_steering(
                    front_bgr=display_frame,
                    dt=dt_actual,
                    steering_filter=lane_filter,
                    lane_steer_limit=args.lane_limit,
                    left_hand_traffic=args.left_hand,
                    yellow_line_distance_gain=args.lane_gain,
                    draw_ego_path=not args.no_ego_path,
                    lane_offset_ratio=args.lane_offset,
                )

                # ── Save annotated frame to disk every N seconds ──────
                # View output by running on your PC:
                #   scp nvidia@<IP>:/tmp/qcar2_lane_*.jpg .
                if t - last_print_t < 0.05:   # save same cadence as print (2s)
                    save_path = f"/tmp/qcar2_lane_{int(t):04d}s.jpg"
                    cv2.imwrite(save_path, display_frame)
                    # Also keep a "latest" file for easy live checking
                    cv2.imwrite("/tmp/qcar2_lane_latest.jpg", display_frame)

            # ── Map/Stanley steering ──────────────────────────────────
            map_steer = 0.0
            if steer_ctrl and ekf:
                # Use front axle position for Stanley (more accurate)
                p_front = np.array([
                    x + 0.2 * np.cos(th),
                    y + 0.2 * np.sin(th),
                ])
                map_steer = steer_ctrl.update(p_front, th, v)

                if steer_ctrl.pathComplete:
                    print("[INFO] Path complete — stopping.")
                    break

            # ── Blend lane + map steering ─────────────────────────────
            # Both weights are applied independently and summed.
            # If lane not detected, falls back to map-only.
            # If no route, uses lane-only.
            if lane_steer is not None and steer_ctrl:
                # Blend: map guides route, lane keeps within lane
                blended = args.map_weight * map_steer + args.lane_weight * lane_steer
            elif lane_steer is not None:
                blended = lane_steer
            else:
                blended = map_steer

            # ── Rate-limit steering (prevents jerks on physical) ──────
            delta = rate_limit(
                current=delta,
                target=float(np.clip(blended, -args.max_steer, args.max_steer)),
                dt=dt_actual,
                up_rate=args.steer_rate,
                down_rate=args.steer_rate,
            )

            # ── Speed control ─────────────────────────────────────────
            throttle = speed_ctrl.update(v, args.cruise_speed, dt_actual)
            # Apply minimum effective throttle deadband so QCar2 actually moves
            # (below ~0.065 the motors don't overcome static friction)
            if throttle > 0.0:
                throttle = max(throttle, args.min_throttle)

            # ── Send commands ─────────────────────────────────────────
            qcar.write(throttle, delta)

            # ── Status print — exactly once every 2 seconds ───────────
            if t - last_print_t >= 2.0:
                last_print_t = t
                lane_str = f"{lane_steer:+.3f}" if lane_steer is not None else "  None"
                det_str  = "LANE OK" if lane_steer is not None else "NO LANE"
                cam_str  = "frame OK" if frame is not None else "NO FRAME"
                print(
                    f"[t={t:6.1f}s] {det_str}  {cam_str}  v={v:.2f}m/s  "
                    f"thr={throttle:.3f}  lane={lane_str}  δ={delta:+.3f}  "
                    f"→ /tmp/qcar2_lane_latest.jpg"
                )

            time.sleep(dt)

        # ── Safe stop ─────────────────────────────────────────────────
        qcar.write(0.0, 0.0)
        print("[INFO] Stopped.")

        if gps:
            gps.__exit__(None, None, None)


# ============================================================
# SECTION 7 — TUNING GUIDE
# ============================================================

TUNING_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║              QCar2 LANE DETECTION TUNING GUIDE                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SYMPTOM                    → WHAT TO CHANGE                         ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Yellow line not detected   → Lower lowerBounds[1] (saturation)      ║
║                               e.g. [10, 30, 80] instead of          ║
║                               [10, 50, 100]                          ║
║                                                                      ║
║  Detecting wrong objects    → Raise lowerBounds (more selective)     ║
║  as yellow lane             → Narrow hue range (try [15, 40])        ║
║                                                                      ║
║  Car drifts away from       → Increase --lane-gain (default 0.45)    ║
║  yellow line                → Try 0.55 or 0.65                       ║
║                                                                      ║
║  Car oscillates / wiggles   → Decrease --lane-gain                   ║
║  around yellow line         → Decrease --lane-cutoff-hz (try 15)     ║
║                               Decrease --steer-rate (try 0.6)        ║
║                                                                      ║
║  Car cuts corners           → Increase --stanley-k (try 0.7-0.9)     ║
║                                                                      ║
║  Car overshoots turns       → Decrease --stanley-k (try 0.3-0.4)     ║
║                               Decrease --cruise-speed                ║
║                                                                      ║
║  Car too slow               → Increase --max-throttle (try 0.20)     ║
║                               Increase --cruise-speed (try 0.5)      ║
║                                                                      ║
║  Physical vs Virtual:                                                ║
║    Physical QCar2 recommended starting values:                       ║
║      --cruise-speed 0.35  --max-throttle 0.14                        ║
║      --stanley-k 0.5  --lane-gain 0.45  --steer-rate 0.95            ║
║                                                                      ║
║    Virtual (QLabs) recommended starting values:                      ║
║      --cruise-speed 0.5   --max-throttle 0.20                        ║
║      --stanley-k 0.6  --lane-gain 0.50  --steer-rate 1.5             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(TUNING_GUIDE)
    main()
