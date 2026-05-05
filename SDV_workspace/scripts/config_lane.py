"""
config_lane.py — Lane Following Configuration for QCar2
========================================================
All tunable parameters in one place for easy adjustment.

Adjust these values based on your:
- Camera mounting position/angle
- Road/track lighting conditions
- Lane marking visibility
- Desired driving behavior
"""

# ══════════════════════════════════════════════════════════════════
#  CAMERA SETTINGS
# ══════════════════════════════════════════════════════════════════

# CSI Camera native resolution for QCar2
CAMERA_WIDTH = 820
CAMERA_HEIGHT = 616

# Frame rate for display/processing
PROCESS_FPS = 30


# ══════════════════════════════════════════════════════════════════
#  LANE DETECTION — REGION OF INTEREST (ROI)
# ══════════════════════════════════════════════════════════════════

# ROI defines the trapezoid area where we look for lanes
# Values are fractions of image dimensions (0.0 to 1.0)

# Vertical bounds (top = closer to horizon, bottom = closer to car)
ROI_TOP_Y = 0.55        # Top of ROI (55% down from top of image)
ROI_BOTTOM_Y = 0.95     # Bottom of ROI (95% down — near bottom edge)

# Horizontal bounds at bottom (wider — near car)
ROI_BOTTOM_LEFT_X = 0.05    # 5% from left edge
ROI_BOTTOM_RIGHT_X = 0.95   # 95% from left edge (5% from right edge)

# Horizontal bounds at top (narrower — vanishing point)
ROI_TOP_LEFT_X = 0.40      # 40% from left edge
ROI_TOP_RIGHT_X = 0.60     # 60% from left edge

# This creates a trapezoid:
#
#        top_left -------- top_right        (narrow, far away)
#           /                    \
#          /                      \
#   bottom_left ------------ bottom_right  (wide, close to car)


# ══════════════════════════════════════════════════════════════════
#  LANE DETECTION — IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════

# Gaussian blur kernel size (must be odd number)
# Larger = more smoothing, removes more noise but may blur edges
BLUR_KERNEL_SIZE = 5

# Canny edge detection thresholds
# Low threshold: weak edges
# High threshold: strong edges
# Edges with gradient > high_threshold = strong edges (kept)
# Edges with gradient < low_threshold = discarded
# Edges between = kept only if connected to strong edges
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


# ══════════════════════════════════════════════════════════════════
#  LANE DETECTION — HOUGH LINE DETECTION
# ══════════════════════════════════════════════════════════════════

# Hough Transform parameters
# rho: Distance resolution in pixels (1 = 1 pixel precision)
HOUGH_RHO = 2

# theta: Angle resolution in radians (np.pi/180 = 1 degree precision)
HOUGH_THETA_FACTOR = 180  # Used as: np.pi / HOUGH_THETA_FACTOR

# threshold: Minimum number of intersections to detect a line
# Higher = fewer but stronger lines
HOUGH_THRESHOLD = 50

# minLineLength: Minimum line length in pixels
# Shorter lines are rejected
HOUGH_MIN_LINE_LENGTH = 50

# maxLineGap: Maximum gap between line segments to treat them as one line
HOUGH_MAX_LINE_GAP = 150


# ══════════════════════════════════════════════════════════════════
#  LANE DETECTION — LINE FILTERING
# ══════════════════════════════════════════════════════════════════

# Minimum absolute slope for a line to be considered a lane line
# Filters out near-horizontal lines (road markings, shadows)
# Typical lane lines have slopes between 0.5 and 2.0
MIN_LANE_SLOPE = 0.4

# Maximum absolute slope to avoid near-vertical lines
MAX_LANE_SLOPE = 3.0


# ══════════════════════════════════════════════════════════════════
#  STEERING CONTROL
# ══════════════════════════════════════════════════════════════════

# Steering gain: How aggressively to correct lane deviation
# Higher = more aggressive steering (may oscillate)
# Lower = smoother but slower correction
# 
# deviation is normalized: -1.0 (far left) to +1.0 (far right)
# steer output will be: deviation * STEERING_GAIN
STEERING_GAIN = 0.5

# Maximum steering angle (radians)
# QCar typical range: -0.5 to +0.5 radians (~±30 degrees)
MAX_STEERING_ANGLE = 0.5

# Steering deadzone: Don't steer if deviation is very small
# Reduces oscillation and unnecessary corrections
STEERING_DEADZONE = 0.05  # ±5% deviation = go straight


# ══════════════════════════════════════════════════════════════════
#  SPEED CONTROL
# ══════════════════════════════════════════════════════════════════

# Base forward speed when lane is detected
FORWARD_SPEED = 0.12

# Reduced speed when turning sharply (abs(steer) > TURN_THRESHOLD)
TURN_SPEED = 0.08

# Steering threshold to trigger speed reduction
TURN_THRESHOLD = 0.3  # If abs(steer) > 0.3, slow down


# ══════════════════════════════════════════════════════════════════
#  SAFETY — LIDAR OBSTACLE DETECTION
# ══════════════════════════════════════════════════════════════════

# LiDAR stop distance in meters
# If obstacle detected closer than this, STOP immediately
LIDAR_STOP_DISTANCE = 0.8  # 80 cm

# LiDAR angle range for front detection (degrees)
# ±15 degrees = front cone
LIDAR_FRONT_ANGLE = 15


# ══════════════════════════════════════════════════════════════════
#  LANE CONFIDENCE & FAIL-SAFE
# ══════════════════════════════════════════════════════════════════

# Minimum number of lines to consider lane detection valid
# If fewer lines detected, confidence = 0
MIN_LINES_FOR_CONFIDENCE = 2

# Number of consecutive frames with no lane before stopping
# Prevents stopping due to temporary detection failures
NO_LANE_STOP_FRAMES = 15  # ~0.5 seconds at 30 fps

# When lane is lost, use this default steering
FALLBACK_STEERING = 0.0  # Go straight


# ══════════════════════════════════════════════════════════════════
#  OCR SIGN DETECTION (OPTIONAL)
# ══════════════════════════════════════════════════════════════════

# Enable/disable OCR sign detection
OCR_ENABLED = False  # Set to True to enable sign detection

# OCR languages
OCR_LANGS = ['en']

# OCR confidence threshold (0.0 to 1.0)
OCR_CONF_THRESHOLD = 0.55

# Run OCR every N frames (to reduce CPU load)
OCR_EVERY_N_FRAMES = 10

# Center ROI for OCR (fraction of image width/height)
OCR_CENTER_ROI = 0.60  # 60% center region

# Number of frames a sign must be detected to trigger action
OCR_TRIGGER_FRAMES = 2


# ══════════════════════════════════════════════════════════════════
#  STATE MACHINE — FIXED TURNS (FOR OCR SIGNS)
# ══════════════════════════════════════════════════════════════════

# Fixed turn parameters (used when OCR detects "left" or "right" sign)
TURN_STEER = 0.5      # Steering angle for fixed turn
TURN_DURATION = 1.2   # Duration of fixed turn in seconds


# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION & DEBUG
# ══════════════════════════════════════════════════════════════════

# Colors for visualization (BGR format)
COLOR_ROI = (0, 255, 0)         # Green
COLOR_LANE_LINES = (0, 0, 255)  # Red
COLOR_LANE_CENTER = (0, 255, 255)  # Cyan
COLOR_IMAGE_CENTER = (255, 0, 0)   # Blue

# Line thickness
LINE_THICKNESS = 3
CENTER_LINE_THICKNESS = 2

# Save debug images every N frames (0 = disabled)
SAVE_DEBUG_EVERY_N_FRAMES = 0  # Set to 30 to save every 30 frames
DEBUG_SAVE_PATH = "/home/nvidia/Desktop/SDV_workspace/scripts/debug_frames"

