"""
config.py - Centralized Configuration for QCar2 Lane Following & Obstacle Avoidance
====================================================================================
Values are taken DIRECTLY from the original QCar v1 repository:
  - lines.py:        perspective points, HSV ranges, RANSAC, sliding window
  - control.py:      PID gains, setpoint, anti-windup
  - camera.py:       resolution, framerate
  - lidar.py:        detection distance, thresholds
  - LineFollower.py:  throttle, steering limits, resume time
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CAMERA CONFIGURATION  (from original camera.py)
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    """
    Original QCar v1 used Camera2D(cameraId='3', 1640x820, 60fps).
    QCar2 uses QCarCameras. We capture at 820x616 (supported by QCar2)
    then resize to 820x410 in the pipeline to match original perspective points.
    """
    camera_id: int = 2              # Front CSI on QCar2 is index 2 (Right=0, Rear=1, Front=2, Left=3)
    capture_width: int = 820        # Native Orin AGX IMX219 resolution
    capture_height: int = 616       # Native Orin AGX IMX219 resolution
    fps: int = 30                   # Stable framerate

    # Processing resolution after 0.5x resize (matches original pipeline)
    process_width: int = 820
    process_height: int = 410


# ---------------------------------------------------------------------------
# COMPUTER VISION PIPELINE  (from original lines.py)
# ---------------------------------------------------------------------------
@dataclass
class CVConfig:
    """
    All values directly from the original lines.py LaneDetect class.
    """
    # Perspective transform points (from lines.py __init__)
    # These are calibrated for 820x410 (post-resize) images
    src_points: np.ndarray = field(default_factory=lambda: np.float32([
        [270, 270], [550, 270],     # Top-left, Top-right
        [0, 380],   [820, 380]      # Bottom-left, Bottom-right
    ]))
    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [270, 0],   [550, 0],
        [270, 410], [550, 410]
    ]))

    # HSV bounds for yellow lane detection (from lines.py __init__)
    hsv_lower: np.ndarray = field(default_factory=lambda: np.array([10, 50, 100]))
    hsv_upper: np.ndarray = field(default_factory=lambda: np.array([45, 255, 255]))

    # Adaptive threshold brightness boundaries (from lines.py adaptive_threshold)
    dark_brightness_threshold: int = 80
    bright_brightness_threshold: int = 180

    # Sliding window parameters (from lines.py locate_lanes)
    n_windows: int = 10             # nwindows = 10
    margin: int = 40                # margin = 40
    min_pixels: int = 30            # minpix = 30

    # Histogram minimum peak threshold (from lines.py histogram)
    histogram_min_peak: int = 50

    # Polynomial degree
    poly_degree: int = 2

    # RANSAC parameters (from lines.py __init__)
    ransac_min_samples: int = 10     # self.ransac_min_samples = 10
    ransac_residual_threshold: float = 2.0  # self.ransac_residual_threshold = 2.0
    ransac_max_trials: int = 100     # self.ransac_max_trials = 100

    # Temporal smoothing (from lines.py __init__)
    smoothing_frames: int = 5        # self.max_history = 5

    # Lane width offset when only one line visible (from lines.py draw_lines)
    single_lane_offset_px: int = 125  # center_lines = columnL_aux + 125

    # Camera center offset (from lines.py draw_lines)
    center_cam_offset: int = 22       # center_cam = (width // 2) + 22


# ---------------------------------------------------------------------------
# PID CONTROL  (from original control.py)
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    """
    All values directly from the original control.py ControlSystem class.
    """
    kp: float = 0.00225              # self.kp = 0.00225
    ki: float = 0.00015              # self.ki = 0.00015
    kd: float = 0.00075              # self.kd = 0.00075

    setpoint: float = -22.452118490490875   # self.setpoint (original exact value)

    base_speed: float = 0.08         # throttle_axis = 0.08 (from LineFollower.py)
    max_steering: float = 0.6        # saturate limit = 0.6 (from LineFollower.py)

    anti_windup: float = 100.0       # self.max_integral = 100 (from control.py)

    # Derivative low-pass filter (NEW improvement for QCar2)
    derivative_alpha: float = 0.3


# ---------------------------------------------------------------------------
# SAFETY / LIDAR  (from original lidar.py)
# ---------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    """
    Values from the original lidar.py LidarProcessor class.
    The original used pixel-grid-based detection; we convert to
    angular ROI which is cleaner but produces equivalent behavior.
    """
    max_lidar_range_m: float = 1.5    # self.maxDistance = 1.5
    stop_distance_m: float = 0.5      # Effective stop dist from pixel math
    lidar_front_angle_deg: float = 0.0
    roi_angle_deg: int = 15           # ±15° frontal cone
    resume_delay_frames: int = 5      # RESUME_TIME = 5 (from LineFollower.py)
    min_valid_distance_m: float = 0.05
    min_obstacle_points: int = 10     # len(Y_) > 10 (from lidar.py detect_object)


# ---------------------------------------------------------------------------
# LED CONFIGURATION  (from original LineFollower.py)
# ---------------------------------------------------------------------------
@dataclass
class LEDConfig:
    """
    LED arrays from the original LineFollower.py.
    """
    # All off
    off: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    # Normal driving: LEDs = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    headlights: list = field(default_factory=lambda: [0, 0, 0, 0, 1, 1, 0, 0])
    # Obstacle: LEDs = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    hazard: list = field(default_factory=lambda: [1, 0, 1, 0, 1, 0, 1, 0])
    # Braking (same as hazard in original)
    braking: list = field(default_factory=lambda: [1, 0, 1, 0, 1, 0, 1, 0])


# ---------------------------------------------------------------------------
# MASTER APPLICATION CONFIG
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    """Top-level config that bundles all sub-configs."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    leds: LEDConfig = field(default_factory=LEDConfig)
