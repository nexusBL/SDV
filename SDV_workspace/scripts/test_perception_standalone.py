#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║       SDV STANDALONE PERCEPTION TEST — No ROS2 Required      ║
║       Runs full pipeline directly with Quanser PAL APIs      ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 test_perception_standalone.py

Requirements:
    - Quanser PAL libraries (QCarRealSense, QCarLidar)
    - RealSense camera connected
    - RPLidar A2 connected
    - YOLOv8 model at configured path
"""

import sys
import os
import time
import signal
import numpy as np
import cv2

# ── Add workspace to path so we can import sdv_perception ──
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_DIR = os.path.join(_WS_ROOT, 'src', 'sdv_perception')
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from sdv_perception.config import SDVConfig
from sdv_perception.object_detector import ObjectDetector
from sdv_perception.depth_fusion import DepthFusion
from sdv_perception.lane_detector import LaneDetector
from sdv_perception.lidar_processor import LidarProcessor
from sdv_perception.visualization import HUDRenderer

# ── Quanser PAL imports ──
try:
    from pal.products.qcar import QCarRealSense, QCarLidar
    HAS_QUANSER = True
except ImportError:
    HAS_QUANSER = False
    print("[WARN] Quanser PAL not found — will use synthetic test data.")


_RUNNING = True

def _signal_handler(sig, frame):
    global _RUNNING
    print("\n[INFO] Caught Ctrl+C — shutting down...")
    _RUNNING = False

signal.signal(signal.SIGINT, _signal_handler)


def run_with_quanser():
    """Run full pipeline with real QCar2 hardware via Quanser PAL."""
    cfg = SDVConfig.get()
    cam = cfg.camera
    width, height = cam['width'], cam['height']

    print("=" * 60)
    print(" SDV Standalone Perception Test — LIVE HARDWARE")
    print("=" * 60)

    # Initialize sensors
    print("[1/6] Initializing RealSense camera...")
    realsense = QCarRealSense(
        mode='RGB&DEPTH',
        frameWidthRGB=width,
        frameHeightRGB=height,
    )

    print("[2/6] Initializing RPLidar A2...")
    lidar = QCarLidar(numMeasurements=400)

    print("[3/6] Loading YOLOv8 object detector (GPU)...")
    detector = ObjectDetector(cfg)

    print("[4/6] Initializing depth fusion...")
    depth_fuser = DepthFusion(cfg)

    print("[5/6] Initializing lane detector...")
    lane_det = LaneDetector(width, height, cfg)

    print("[6/6] Initializing LiDAR processor + HUD...")
    lidar_proc = LidarProcessor(cfg)
    hud = HUDRenderer(cfg)

    print("\n✅ All systems initialized! Starting perception loop...")
    print("   Press 'q' in the OpenCV window or Ctrl+C to stop.\n")

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    fps_interval = cfg.performance['fps_update_interval']

    try:
        while _RUNNING:
            t_start = time.time()

            # Read sensors
            realsense.read_RGB()
            rgb_frame = realsense.imageBufferRGB
            if rgb_frame is None or rgb_frame.size == 0:
                time.sleep(0.01)
                continue

            # Ensure correct shape (H, W, 3) BGR
            if len(rgb_frame.shape) == 2:
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
            elif rgb_frame.shape[2] == 4:
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)

            # Read depth
            realsense.read_depth()
            depth_frame = realsense.imageBufferDepth

            # Read LiDAR
            lidar.read()

            # ── Pipeline ──
            # 1) Object detection
            detections = detector.detect(rgb_frame)

            # 2) Depth fusion
            if depth_frame is not None and depth_frame.size > 0:
                detections = depth_fuser.fuse(detections, depth_frame)

            # 3) Draw detections
            annotated = ObjectDetector.draw_detections(rgb_frame, detections)

            # 4) Lane detection
            annotated, lane_offset, left_ok, right_ok = lane_det.process(annotated)

            lane_status = ('BOTH' if (left_ok and right_ok)
                           else 'LEFT_ONLY' if left_ok
                           else 'RIGHT_ONLY' if right_ok
                           else 'NONE')

            # 5) LiDAR analysis
            if lidar.distances is not None:
                lidar_proc.update_raw(
                    ranges=np.array(lidar.distances).flatten(),
                    angle_min=0.0,
                    angle_increment=2 * np.pi / max(len(lidar.distances), 1),
                )
            threat, min_dist, zones = lidar_proc.analyze()

            # 6) FPS
            frame_count += 1
            if frame_count % fps_interval == 0:
                elapsed = time.time() - fps_timer
                fps = fps_interval / max(elapsed, 1e-6)
                fps_timer = time.time()

            latency_ms = (time.time() - t_start) * 1000

            # 7) HUD
            annotated = hud.draw(
                annotated, detections, lane_offset, lane_status,
                threat, min_dist, zones, latency_ms, fps,
            )

            # Display
            cv2.imshow('SDV Perception — Standalone', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Print periodic status
            if frame_count % 60 == 0:
                obj_str = ', '.join(
                    f'{d.class_name}({d.distance_m:.1f}m)'
                    for d in detections
                ) or 'none'
                print(
                    f"[Frame {frame_count}] "
                    f"FPS={fps:.1f} | Lat={latency_ms:.1f}ms | "
                    f"Threat={threat} | Lane={lane_status} | "
                    f"Objects: {obj_str}"
                )

    finally:
        print("\n🛑 Shutting down sensors...")
        realsense.terminate()
        lidar.terminate()
        cv2.destroyAllWindows()
        print("✅ Clean shutdown complete.")


def run_synthetic():
    """Run pipeline with synthetic data (no hardware needed) to verify modules."""
    cfg = SDVConfig.get()
    cam = cfg.camera
    width, height = cam['width'], cam['height']

    print("=" * 60)
    print(" SDV Standalone Perception Test — SYNTHETIC MODE")
    print(" (No Quanser hardware detected — using test images)")
    print("=" * 60)

    print("[1/4] Loading YOLOv8 object detector (GPU)...")
    detector = ObjectDetector(cfg)

    print("[2/4] Initializing depth fusion...")
    depth_fuser = DepthFusion(cfg)

    print("[3/4] Initializing lane detector...")
    lane_det = LaneDetector(width, height, cfg)

    print("[4/4] Initializing LiDAR processor + HUD...")
    lidar_proc = LidarProcessor(cfg)
    hud = HUDRenderer(cfg)

    print("\n✅ All modules loaded! Running 30 synthetic frames...\n")

    # Synthetic test image: gray with some lines
    for i in range(30):
        t_start = time.time()

        # Create synthetic frame
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        # Draw fake lane lines
        cv2.line(frame, (width // 3, height), (width // 2 - 50, int(height * 0.6)),
                 (255, 255, 255), 3)
        cv2.line(frame, (2 * width // 3, height), (width // 2 + 50, int(height * 0.6)),
                 (255, 255, 255), 3)
        # Add some label text
        cv2.putText(frame, f'Synthetic Frame {i+1}/30', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Run pipeline
        detections = detector.detect(frame)
        annotated = ObjectDetector.draw_detections(frame, detections)
        annotated, lane_offset, left_ok, right_ok = lane_det.process(annotated)

        lane_status = ('BOTH' if (left_ok and right_ok)
                       else 'LEFT_ONLY' if left_ok
                       else 'RIGHT_ONLY' if right_ok
                       else 'NONE')

        threat, min_dist, zones = lidar_proc.analyze()
        latency_ms = (time.time() - t_start) * 1000

        annotated = hud.draw(
            annotated, detections, lane_offset, lane_status,
            threat, min_dist, zones, latency_ms, 30.0,
        )

        print(
            f"  Frame {i+1:2d}/30 | "
            f"Lat={latency_ms:6.1f}ms | "
            f"Dets={len(detections)} | "
            f"Lane={lane_status:10s} | "
            f"Threat={threat}"
        )

    print("\n✅ Synthetic test complete — all modules functional!")


def main():
    if HAS_QUANSER:
        run_with_quanser()
    else:
        run_synthetic()


if __name__ == '__main__':
    main()
