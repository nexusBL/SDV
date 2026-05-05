#!/usr/bin/env python3
"""
QCar2 Lane Detection — Headless Test Script

Runs the full detection pipeline on live camera frames WITHOUT requiring
a display (no cv2.imshow). Perfect for SSH testing.

Usage:
    python3 test_headless.py             # Capture 20 frames and analyse
    python3 test_headless.py --frames 50 # Capture 50 frames
    python3 test_headless.py --image path/to/image.jpg  # Test on saved image

Outputs:
    - Saves annotated debug images to ./headless_debug/
    - Prints detection metrics to console
"""
import sys
import os

# Ensure headless operation
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

import numpy as np
import cv2
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AppConfig
from perception.lane_cv import LaneDetector


def test_from_camera(cfg, num_frames=20):
    """Capture live frames and run detection."""
    from pal.products.qcar import QCarCameras

    print(f"[Test] Capturing {num_frames} frames from front CSI camera...")

    cameras = QCarCameras(
        frameWidth=cfg.camera.width,
        frameHeight=cfg.camera.height,
        frameRate=cfg.camera.fps,
        enableFront=True,
    )

    # Warmup
    print("[Test] Warming up camera...")
    for _ in range(cfg.camera.warmup_frames):
        cameras.readAll()
        time.sleep(0.03)

    detector = LaneDetector(cfg)
    frames = []

    print("[Test] Capturing frames...")
    for i in range(num_frames):
        cameras.readAll()
        img = cameras.csiFront.imageData
        if img is not None and img.max() > 10:
            frames.append(img.copy())
        time.sleep(0.05)

    cameras.terminate()
    print(f"[Test] Captured {len(frames)} valid frames.")
    return detector, frames


def test_from_image(cfg, image_path):
    """Test detection on a saved image file."""
    print(f"[Test] Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    # Resize to expected config resolution
    img = cv2.resize(img, (cfg.camera.width, cfg.camera.height))

    detector = LaneDetector(cfg)
    return detector, [img]


def run_detection(detector, frames, output_dir):
    """Run detection on all frames and save results."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" {'Frame':<8} {'Offset(m)':>10} {'Curv(m)':>10} {'Conf':>6} "
          f"{'L_px':>6} {'R_px':>6} {'Status':<10}")
    print(f"{'='*70}")

    for i, frame in enumerate(frames):
        t0 = time.time()
        error_m, hud = detector.process_frame(frame)
        dt = (time.time() - t0) * 1000  # ms

        status = "BOTH" if detector.left_pixel_count >= 300 and \
                           detector.right_pixel_count >= 300 else \
                 "PARTIAL" if detector.left_pixel_count >= 300 or \
                              detector.right_pixel_count >= 300 else "NONE"

        offset_str = f"{detector.lateral_offset_m:+.4f}" if error_m is not None else "  N/A  "
        curv_str = f"{detector.curvature_radius_m:.1f}" if detector.curvature_radius_m < 100 else "Straight"

        print(f" {i+1:<8} {offset_str:>10} {curv_str:>10} "
              f"{detector.confidence:>5.0%} "
              f"{detector.left_pixel_count:>6} {detector.right_pixel_count:>6} "
              f"{status:<10} ({dt:.0f}ms)")

        # Save annotated frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{i+1:03d}.jpg"), hud)

        # Save raw input for first frame
        if i == 0:
            cv2.imwrite(os.path.join(output_dir, "input_frame.jpg"), frame)

    print(f"{'='*70}")
    print(f"\n[Test] Saved {len(frames)} annotated frames to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="QCar2 Lane Detection Headless Test")
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of frames to capture (default: 20)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a saved image to test on')
    args = parser.parse_args()

    cfg = AppConfig()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "headless_debug")

    if args.image:
        detector, frames = test_from_image(cfg, args.image)
    else:
        detector, frames = test_from_camera(cfg, args.frames)

    if len(frames) == 0:
        print("[ERROR] No valid frames captured!")
        sys.exit(1)

    run_detection(detector, frames, output_dir)

    print("\n✅ Test complete! Review the images in headless_debug/")
    print("   Copy to your laptop with: scp nvidia@<ip>:~/Desktop/SDV_workspace/"
          "scripts/qcar2_lane_following/headless_debug/*.jpg .")


if __name__ == "__main__":
    main()
