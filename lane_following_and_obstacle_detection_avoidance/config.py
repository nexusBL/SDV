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
    camera_id: int = 2              # Front CSI on QCar2 is index 2
    enable_side_csi: bool = True     # NEW: support side-based avoidance
    capture_width: int = 820
    capture_height: int = 616
    fps: int = 30

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
    # These are calibrated for 820x410 (post-resize) images.
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
# PID CONTROL  (from original control.py)
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    """
    All values directly from the original control.py ControlSystem class.
    """
    kp: float = 0.00225
    ki: float = 0.00015
    kd: float = 0.00075
    setpoint: float = -22.452118490490875
    base_speed: float = 0.08
    search_speed: float = 0.07  # NEW: search speed for tessting.py
    max_steering: float = 0.6
    anti_windup: float = 100.0
    derivative_alpha: float = 0.3


# ---------------------------------------------------------------------------
# SAFETY / LIDAR  (from original lidar.py)
# ---------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    """
    Values from the original lidar.py LidarProcessor class.
    """
    max_lidar_range_m: float = 1.5
    stop_distance_m: float = 1.5
    lidar_front_angle_deg: float = 180.0 # Standard QCar2
    roi_angle_deg: int = 45
    resume_delay_frames: int = 5
    min_valid_distance_m: float = 0.05
    min_obstacle_points: int = 2
    
    # --- Tessting.py additions ---
    avoid_trigger_distance_m: float = 1.3
    avoid_trigger_arc_deg: float = 45.0
    lane_block_arc_deg: float = 20.0
    lane_block_points_min: int = 2
    re_entry_vision_threshold: float = 0.45
    car_width_m: float = 0.35
    gap_max_angle_deg: float = 60.0
    
    depth_stop_distance_m: float = 0.85      # Increased from 0.55 to stop earlier
    depth_roi_width_frac: float = 0.35       # Narrower central ROI
    depth_roi_height_frac: float = 0.25      # MUCH shorter ROI to look further ahead/up
    depth_min_confidence: float = 0.4        # More sensitive depth
    depth_persist_frames: int = 4            # Less persistence (quicker reaction)

    # --- Lane re-entry hysteresis ---
    lane_reentry_frames_required: int = 6    # Consecutive good-vision frames to re-enter DRIVING
    # Blend factor for left/right CSI vs LiDAR side bias during AVOIDING (0..1)
    side_camera_blend: float = 0.45
    # Target lateral clearance to maintain using side cameras (meters)
    # Reduced slightly to 0.55m for "near obstacle" requirement, with higher gain.
    side_clearance_target_m: float = 0.55
    side_clearance_gain: float = 0.5


# ---------------------------------------------------------------------------
# REACTIVE AVOIDANCE (potential field on LiDAR, LiDAR-primary over depth)
# ---------------------------------------------------------------------------
@dataclass
class ReactiveAvoidanceConfig:
    """
    Artificial potential field on polar LiDAR: repulsion from nearby points,
    weak attraction forward. Use while STATE_AVOIDING when path is blocked.
    """
    influence_radius_m: float = 1.2    # points closer contribute repulsion
    repulse_gain: float = 0.45         # scales lateral repulsion → steering
    forward_speed: float = 0.06        # cautious forward while avoiding
    slow_when_steering: float = 0.35   # throttle scale factor at full steer
    max_steer: float = 0.55            # clamp (radians, same as control path)

    # Match production-grade controller semantics
    front_arc_deg: float = 60.0
    lidar_reverse: bool = True
    car_width_m: float = 0.35
    gap_max_angle_deg: float = 60.0



# ---------------------------------------------------------------------------
# AVOIDANCE MANEUVER CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class AvoidanceConfig:
    """
    Parameters for the simple obstacle avoidance maneuver.
    Sequence: back-up → steer+drive → straighten → resume lane.
    """
    # Phase 1: Brief reverse to create clearance
    backup_throttle: float = -0.06       # Slow reverse
    backup_duration_frames: int = 10     # ~0.3s at 30fps

    # Phase 2: Steer around obstacle (default: right)
    avoidance_throttle: float = 0.07     # Slow crawl
    avoidance_steer: float = 0.4         # Hard right (positive = right on QCar2)
    avoidance_duration_frames: int = 40  # ~1.3s at 30fps

    # Phase 3: Straighten out and let lane detection re-acquire
    straighten_throttle: float = 0.07
    straighten_steer: float = -0.15      # Slight left correction
    straighten_duration_frames: int = 20 # ~0.7s at 30fps


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
    # Right blinker: used during avoidance maneuver
    right_blinker: list = field(default_factory=lambda: [0, 1, 0, 1, 0, 0, 0, 0])


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
    reactive: ReactiveAvoidanceConfig = field(default_factory=ReactiveAvoidanceConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    leds: LEDConfig = field(default_factory=LEDConfig)
