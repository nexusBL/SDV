"""
SDV Perception Package — Modular autonomous driving perception for QCar2.
"""

from .config import SDVConfig
from .object_detector import ObjectDetector, Detection
from .depth_fusion import DepthFusion
from .lane_detector import LaneDetector
from .lidar_processor import LidarProcessor
from .visualization import HUDRenderer

__all__ = [
    'SDVConfig',
    'ObjectDetector',
    'Detection',
    'DepthFusion',
    'LaneDetector',
    'LidarProcessor',
    'HUDRenderer',
]
