import numpy as np
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════
#  QCar2 Lane Following — Master Configuration
#  All tunable parameters in one place. Adjust per-track as needed.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    width:  int = 820
    height: int = 410       # QCar2 PAL default (820×410)
    fps:    int = 30
    camera_id: int = 2      # Front CSI on QCar2
    warmup_frames: int = 15 # Skip first N frames (dark/unstable)

@dataclass
class CVConfig:
    # ── ROI (Region of Interest) ──────────────────────────────────────────
    # Fraction of frame height to keep (bottom portion = road)
    roi_top_fraction: float = 0.45   # Keep bottom 55% of 410px = rows 184..410

    # Trapezoidal ROI vertices as fractions of (width, height)
    # [top-left, top-right, bottom-right, bottom-left]
    roi_poly_frac: np.ndarray = field(default_factory=lambda: np.float32([
        [0.15, 0.45],   # top-left
        [0.85, 0.45],   # top-right
        [1.00, 1.00],   # bottom-right
        [0.00, 1.00],   # bottom-left
    ]))

    # ── Preprocessing ─────────────────────────────────────────────────────
    blur_ksize: int = 5                    # Gaussian blur kernel size
    adaptive_block_size: int = 25          # Adaptive threshold block size (odd)
    adaptive_C: int = 4                    # Constant subtracted from mean
    canny_low: int = 50                    # Canny lower threshold
    canny_high: int = 150                  # Canny upper threshold

    # ── Perspective Transform (Bird's Eye View) ───────────────────────────
    # Source points in the ORIGINAL camera frame coordinates.
    # MUST BE CALIBRATED using calibrate.py on your physical track.
    # These defaults are for 820×410 with the standard QCar2 camera mount.
    src_points: np.ndarray = field(default_factory=lambda: np.float32([
        [260, 200],   # Top-left
        [560, 200],   # Top-right
        [780, 400],   # Bottom-right
        [ 40, 400],   # Bottom-left
    ]))

    # Destination points (bird's-eye output). Full frame rectangle.
    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [100, 0],     # Top-left (slight inset for better line visibility)
        [720, 0],     # Top-right
        [720, 410],   # Bottom-right
        [100, 410],   # Bottom-left
    ]))

    # ── Sliding Window ────────────────────────────────────────────────────
    n_windows:   int = 10    # Number of vertical search windows
    margin:      int = 80    # Half-width of each search window (pixels)
    min_pixels:  int = 50    # Min pixels to re-center window

    # ── Polynomial Fitting ────────────────────────────────────────────────
    poly_degree: int = 2

    # ── Temporal Smoothing (EMA) ──────────────────────────────────────────
    ema_alpha: float = 0.7   # Exponential moving average factor (0=ignore new, 1=no smoothing)

    # ── Confidence Gating ─────────────────────────────────────────────────
    min_lane_pixels: int = 300   # Below this → low confidence, hold last fit

    # ── Pixel-to-Meter Conversion ─────────────────────────────────────────
    # Measured on your physical track:
    #   lane_width_m  = physical distance between left and right lane edges
    #   lane_width_px = pixel distance between lanes in the bird's-eye view
    lane_width_m:  float = 0.37   # ~0.37m for 1/10th scale QCar2 track
    lane_width_px: float = 620.0  # Approximate pixel width in BEV (calibrate!)

    # Vertical: how many meters correspond to full BEV height
    bev_height_m:  float = 0.80   # ~0.8m of road visible in BEV
    bev_height_px: float = 410.0  # BEV height in pixels


@dataclass
class ControlConfig:
    # ── PID Gains (error is in METERS, output is steering radians) ────────
    kp: float = 2.5         # Proportional (meters → radians)
    ki: float = 0.3         # Integral
    kd: float = 0.8         # Derivative

    base_speed:    float = 0.12    # Straight-line throttle (increased from 0.08)
    curve_speed:   float = 0.08    # Reduced speed on curves (increased from 0.05)
    max_steering:  float = 0.5     # Max steering angle (radians)
    curvature_speed_threshold: float = 3.0  # Curvature radius below this → use curve_speed

    # PID anti-windup (integral clamp, in meter·seconds)
    anti_windup: float = 0.5

    # Derivative low-pass filter factor (0=no filter, 1=full filter)
    d_filter_alpha: float = 0.3


@dataclass
class SafetyConfig:
    stop_distance_m:     float = 0.5
    lidar_front_angle_deg: float = 0.0
    roi_angle_deg:       int   = 15
    resume_time_frames:  int   = 5


@dataclass
class AppConfig:
    camera:  CameraConfig  = field(default_factory=CameraConfig)
    cv:      CVConfig      = field(default_factory=CVConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety:  SafetyConfig  = field(default_factory=SafetyConfig)
