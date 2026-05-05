#!/usr/bin/env python3
"""
SDV Configuration Loader
Loads sdv_config.yaml with sensible defaults.
"""

import os
import yaml
import numpy as np

# Default config path — can be overridden via SDV_CONFIG_PATH env var
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'config', 'sdv_config.yaml'
)


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Hardcoded defaults (used if YAML is missing) ──
_DEFAULTS = {
    'camera': {'width': 1280, 'height': 720, 'fps': 30},
    'yolo': {
        'model_path': '/home/nvidia/Desktop/SDV_workspace/yolov8n.pt',
        'confidence': 0.45,
        'iou_threshold': 0.45,
        'image_size': 640,
        'device': 'cuda',
        'warmup_frames': 3,
        'critical_classes': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic light',
            11: 'stop sign', 12: 'parking meter',
        },
    },
    'lane': {
        'roi_top_ratio': 0.55,
        'blur_kernel': [5, 5],
        'canny_low': 50, 'canny_high': 150,
        'hough_threshold': 50, 'hough_min_line_length': 200,
        'hough_max_line_gap': 15,
        'perspective': {
            'src_bottom_left': [0.20, 0.95],
            'src_top_left': [0.45, 0.60],
            'src_top_right': [0.55, 0.60],
            'src_bottom_right': [0.80, 0.95],
            'dst_left': 0.25, 'dst_right': 0.75,
        },
        'sliding_window': {'n_windows': 9, 'margin': 80, 'min_pixels': 40},
        'white_low': [0, 200, 0], 'white_high': [180, 255, 255],
        'yellow_low': [15, 80, 100], 'yellow_high': [35, 255, 255],
        'fit_history_length': 8, 'center_history_length': 10,
    },
    'depth': {
        'min_distance': 0.1, 'max_distance': 10.0,
        'sample_ratio': 0.3, 'scale_factor': 0.001,
    },
    'lidar': {
        'critical_distance': 0.5, 'warning_distance': 1.5,
        'front_arc_degrees': 30, 'max_range': 10.0, 'min_range': 0.05,
    },
    'hud': {
        'top_bar_height': 38, 'font_scale_title': 0.7,
        'font_scale_info': 0.6, 'font_scale_panel': 0.55,
        'threat_panel_width': 280, 'threat_panel_height': 130,
        'lane_panel_width': 220, 'lane_panel_height': 100,
        'steering_bar_half_width': 100,
        'background_color': [15, 15, 15],
    },
    'performance': {'max_fps': 30, 'fps_update_interval': 30},
    'pid': {
        'kp': 0.8, 'ki': 0.05, 'kd': 0.1,
        'max_integral': 0.3, 'dt': 0.033,
    },
    'speed': {
        'base_throttle': 0.10,
        'warning_throttle_multiplier': 0.5,
        'max_throttle': 0.15,
        'min_throttle': -0.05,
    },
    'acc': {
        'desired_distance': 1.0,
    },
    'hardware': {
        'loop_rate_hz': 100,
        'steering_max': 0.3,
        'steering_threshold': 0.05,
    },
    'topics': {
        'rgb_image': '/camera/color_image',
        'depth_image': '/camera/depth_image',
        'lidar_scan': '/scan',
        'annotated_image': '/sdv/perception/annotated',
        'detections_json': '/sdv/perception/detections',
        'lane_data': '/sdv/perception/lane',
        'threat_level': '/sdv/perception/threat',
    },
}


class SDVConfig:
    """Singleton-style configuration object for the SDV perception pipeline."""

    _instance = None

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.environ.get('SDV_CONFIG_PATH', _DEFAULT_CONFIG_PATH)

        config_path = os.path.abspath(config_path)
        self._data = _DEFAULTS.copy()

        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_cfg = yaml.safe_load(f) or {}
                self._data = _deep_merge(_DEFAULTS, user_cfg)
            except Exception as e:
                print(f'[SDVConfig] Warning: failed to load {config_path}: {e}')
                print('[SDVConfig] Using built-in defaults.')
        else:
            print(f'[SDVConfig] Config file not found: {config_path}')
            print('[SDVConfig] Using built-in defaults.')

    # ── Accessors ──
    @property
    def camera(self):
        return self._data['camera']

    @property
    def yolo(self):
        return self._data['yolo']

    @property
    def lane(self):
        return self._data['lane']

    @property
    def depth(self):
        return self._data['depth']

    @property
    def lidar(self):
        return self._data['lidar']

    @property
    def hud(self):
        return self._data['hud']

    @property
    def performance(self):
        return self._data['performance']

    @property
    def topics(self):
        return self._data['topics']

    @property
    def pid(self):
        return self._data['pid']

    @property
    def speed(self):
        return self._data['speed']

    @property
    def acc(self):
        return self._data['acc']

    @property
    def hardware(self):
        return self._data['hardware']

    @property
    def raw(self):
        """Access the raw config dict."""
        return self._data

    @classmethod
    def get(cls, config_path=None):
        """Get or create the singleton config instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton — mainly for testing."""
        cls._instance = None
