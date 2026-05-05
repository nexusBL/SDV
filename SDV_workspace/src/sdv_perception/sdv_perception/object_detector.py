#!/usr/bin/env python3
"""
SDV Object Detector — YOLOv8 GPU-accelerated detection for autonomous driving.
"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .config import SDVConfig


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list                    # [x1, y1, x2, y2]
    center: list                  # [cx, cy]
    distance_m: float = -1.0     # filled by DepthFusion
    color: tuple = (255, 255, 0)  # BGR drawing color


class ObjectDetector:
    """YOLOv8-based object detector optimized for Jetson AGX Orin."""

    # Color palette by class priority
    _COLORS = {
        'person':        (0, 0, 255),      # Red — highest priority
        'bicycle':       (0, 100, 255),    # Dark orange
        'car':           (0, 165, 255),    # Orange
        'motorcycle':    (0, 165, 255),    # Orange
        'bus':           (0, 165, 255),    # Orange
        'truck':         (0, 165, 255),    # Orange
        'traffic light': (0, 255, 255),    # Yellow
        'stop sign':     (0, 255, 255),    # Yellow
        'parking meter': (255, 255, 0),    # Cyan
    }
    _DEFAULT_COLOR = (255, 255, 0)         # Cyan fallback

    def __init__(self, config: Optional[SDVConfig] = None):
        if YOLO is None:
            raise ImportError(
                'ultralytics is not installed. '
                'Install with: pip3 install --no-deps ultralytics'
            )

        self.cfg = (config or SDVConfig.get()).yolo
        self.model = YOLO(self.cfg['model_path'])
        
        # Performance optimization: use FP16 half-precision and optimize for TensorRT
        self.device = self.cfg['device']
        self.half = self.device != 'cpu'
        
        # Enforce Half precision and prepare model
        self.model.to(self.device)
        if self.half:
            # Optimize inference explicitly
            pass  # Half is handled in predict()
            
        self.critical_classes = {
            int(k): v for k, v in self.cfg['critical_classes'].items()
        }

        # Detection history for temporal smoothing (reduces flickering)
        self._history = {}
        self._frames_since_seen = {}
        self._max_history = 5     # Keep track for 5 frames max
        
        # Tracker configuration
        self._agnostic_nms = True # Better overlapping object rejection

        self._warmup()

    def _warmup(self):
        """Run dummy inference to warm up the GPU and compile any JIT kernels."""
        img_size = self.cfg['image_size']
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        for _ in range(self.cfg['warmup_frames']):
            with torch.no_grad():
                self.model.predict(
                    source=dummy,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 on a BGR frame with temporal smoothing."""
        # ── Optional: Dynamic ROI cropping to save compute if needed ──
        # Not using crop because we need to detect traffic lights at the top
        results = self.model.predict(
            source=frame,
            device=self.device,
            half=self.half,
            conf=self.cfg['confidence'],
            iou=self.cfg['iou_threshold'],
            agnostic_nms=self._agnostic_nms,  # Merge overlapping boxes of different classes
            imgsz=self.cfg['image_size'],
            verbose=False,
            classes=list(self.critical_classes.keys()),
        )

        current_frame_dets = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.critical_classes.get(cls_id, f'cls_{cls_id}')
                color = self._COLORS.get(label, self._DEFAULT_COLOR)

                current_frame_dets.append({
                    'id': cls_id, 'label': label, 'conf': conf,
                    'bbox': [x1, y1, x2, y2], 'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'color': color
                })

        # ── Temporal Smoothing (Anti-Flicker) ──
        smoothed_detections = []
        matched_hist_keys = set()
        
        for det in current_frame_dets:
            # Simple matching: same class and centers within 50 pixels
            cx, cy = det['center']
            match_key = None
            for key, hist_det in self._history.items():
                if hist_det['label'] == det['label'] and key not in matched_hist_keys:
                    hcx, hcy = hist_det['center']
                    if abs(cx - hcx) < 50 and abs(cy - hcy) < 50:
                        match_key = key
                        break
            
            if match_key:
                # Smooth the bounding box (EMA)
                alpha = 0.6
                old_bbox = self._history[match_key]['bbox']
                new_bbox = [int(alpha * n + (1 - alpha) * o) for n, o in zip(det['bbox'], old_bbox)]
                
                det['bbox'] = new_bbox
                det['center'] = [(new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2]
                
                self._history[match_key] = det
                self._frames_since_seen[match_key] = 0
                matched_hist_keys.add(match_key)
            else:
                # New object
                new_key = f"{det['label']}_{cx}_{cy}_{np.random.randint(1000)}"
                self._history[new_key] = det
                self._frames_since_seen[new_key] = 0
                matched_hist_keys.add(new_key)
                
            smoothed_detections.append(Detection(
                class_id=det['id'], class_name=det['label'],
                confidence=det['conf'], bbox=det['bbox'],
                center=det['center'], color=det['color']
            ))

        # ── Retain briefly lost objects ("ghosting" for robustness) ──
        keys_to_remove = []
        for key in self._history.keys():
            if key not in matched_hist_keys:
                self._frames_since_seen[key] += 1
                if self._frames_since_seen[key] > self._max_history:
                    keys_to_remove.append(key)
                else:
                    # Keep showing it briefly if we just lost it for a frame or two
                    hist_det = self._history[key]
                    smoothed_detections.append(Detection(
                        class_id=hist_det['id'], class_name=hist_det['label'],
                        confidence=hist_det['conf'] * 0.9, # fade confidence
                        bbox=hist_det['bbox'], center=hist_det['center'], color=hist_det['color']
                    ))
                    
        for key in keys_to_remove:
            del self._history[key]
            del self._frames_since_seen[key]

        return smoothed_detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame. Returns annotated copy."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text
            dist_str = f' {det.distance_m:.1f}m' if det.distance_m > 0 else ''
            label_text = f'{det.class_name} {det.confidence:.0%}{dist_str}'
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 8), (x1 + tw + 4, y1),
                color, -1,
            )
            # Label text
            cv2.putText(
                annotated, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
            )

        return annotated
