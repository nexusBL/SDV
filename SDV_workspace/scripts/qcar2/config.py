"""
config.py - Centralized Configuration for QCar2 Lane Following & Obstacle Avoidance
====================================================================================
Merged version: Stable base + User-requested 35cm stop, 3cm gap, and 60s delay.
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CAMERA CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    camera_id: int = 2              # Front CSI on QCar2 is index 2
    enable_side_csi: bool = True
    capture_width: int = 820
    capture_height: int = 616
    fps: int = 30
    process_width: int = 820
    process_height: int = 410


# ---------------------------------------------------------------------------
# COMPUTER VISION PIPELINE
# ---------------------------------------------------------------------------
@dataclass
class CVConfig:
    src_points: np.ndarray = field(default_factory=lambda: np.float32([
        [270, 270], [550, 270],
        [0, 380],   [820, 380]
    ]))
    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [270, 0],   [550, 0],
        [270, 410], [550, 410]
    ]))
    hsv_lower: np.ndarray = field(default_factory=lambda: np.array([10, 50, 100]))
    hsv_upper: np.ndarray = field(default_factory=lambda: np.array([45, 255, 255]))
    dark_brightness_threshold: int = 80
    bright_brightness_threshold: int = 180
    n_windows: int = 10
    margin: int = 40
    min_pixels: int = 30
    histogram_min_peak: int = 50
    poly_degree: int = 2
    ransac_min_samples: int = 10
    ransac_residual_threshold: float = 2.0
    ransac_max_trials: int = 100
    smoothing_frames: int = 5
    single_lane_offset_px: int = 125
    center_cam_offset: int = 22


# ---------------------------------------------------------------------------
# PID CONTROL
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    kp: float = 0.00225
    ki: float = 0.00015
    kd: float = 0.00075
    setpoint: float = -22.452118490490875
    base_speed: float = 0.08
    crawl_speed: float = 0.04
    search_speed: float = 0.07
    max_steering: float = 0.6
    anti_windup: float = 100.0
    derivative_alpha: float = 0.3


# ---------------------------------------------------------------------------
# SAFETY / LIDAR
# ---------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    max_lidar_range_m: float = 1.5
    stop_distance_cm: float = 35.0    # User Requirement: 35cm
    detection_range_m: float = 0.75
    lidar_front_angle_deg: float = 180.0
    roi_angle_deg: int = 45
    resume_delay_frames: int = 5
    min_valid_distance_m: float = 0.03
    min_obstacle_points: int = 2
    
    # --- Tessting.py additions ---
    avoid_trigger_distance_cm: float = 35.0 # Stop at 35cm
    slow_down_distance_cm: float = 55.0
    avoid_trigger_arc_deg: float = 45.0
    lane_block_arc_deg: float = 20.0
    lane_block_points_min: int = 2
    re_entry_vision_threshold: float = 0.45
    car_width_m: float = 0.35
    gap_max_angle_deg: float = 60.0
    
    depth_stop_distance_m: float = 0.85
    depth_roi_width_frac: float = 0.35
    depth_roi_height_frac: float = 0.25
    depth_min_confidence: float = 0.4
    depth_persist_frames: int = 4

    # --- Lane re-entry ---
    lane_reentry_frames_required: int = 6
    side_camera_blend: float = 0.45
    side_clearance_target_cm: float = 3.0 # User Requirement: 3cm
    side_clearance_gain: float = 0.8


# ---------------------------------------------------------------------------
# REACTIVE AVOIDANCE
# ---------------------------------------------------------------------------
@dataclass
class ReactiveAvoidanceConfig:
    influence_radius_m: float = 1.2
    repulse_gain: float = 0.45
    forward_speed: float = 0.06
    slow_when_steering: float = 0.35
    max_steer: float = 0.55
    front_arc_deg: float = 60.0
    lidar_reverse: bool = True
    car_width_m: float = 0.35
    gap_max_angle_deg: float = 60.0


# ---------------------------------------------------------------------------
# AVOIDANCE MANEUVER CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class AvoidanceConfig:
    # Phase 1: Brief reverse
    backup_throttle: float = -0.06
    backup_duration_frames: int = 15

    # Phase 2: Rotate ~45 degrees (User Requirement)
    rotate_throttle: float = 0.06
    rotate_steering: float = 0.55
    rotate_duration_frames: int = 12 # 0.4s @ 30fps

    # Phase 3: Bypass
    avoidance_throttle: float = 0.07
    avoidance_steer: float = 0.4 
    straighten_throttle: float = 0.07
    straighten_steer: float = -0.15
    straighten_duration_frames: int = 20


# ---------------------------------------------------------------------------
# LED CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class LEDConfig:
    off: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    headlights: list = field(default_factory=lambda: [0, 0, 0, 0, 1, 1, 0, 0])
    hazard: list = field(default_factory=lambda: [1, 0, 1, 0, 1, 0, 1, 0])
    braking: list = field(default_factory=lambda: [1, 0, 1, 0, 1, 0, 1, 0])
    right_blinker: list = field(default_factory=lambda: [0, 1, 0, 1, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# MASTER APPLICATION CONFIG
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    reactive: ReactiveAvoidanceConfig = field(default_factory=ReactiveAvoidanceConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    leds: LEDConfig = field(default_factory=LEDConfig)
