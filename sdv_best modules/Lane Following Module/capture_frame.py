#!/usr/bin/env python3
"""
capture_frame.py - Camera Calibration Utility for QCar2
=======================================================
Captures a single frame from the front CSI camera and saves it as a PNG.
Use this to calibrate your perspective transform points in config.py.

Usage:
  python3 capture_frame.py                    # Saves to ./calibration_frame.png
  python3 capture_frame.py my_track_image     # Saves to ./my_track_image.png

Workflow:
  1. Place the QCar2 on the track in a straight section
  2. Run this script to capture a frame
  3. Open the saved image in an image editor (e.g., GIMP, Paint)
  4. Identify the 4 corners of the lane region:
     - Top-left, Top-right (where lanes converge in the distance)
     - Bottom-left, Bottom-right (near the car bumper)
  5. Update config.py CVConfig.src_points with those pixel coordinates
"""

import os
import sys

# Fix nvarguscamerasrc EGL authorization
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

import cv2
import numpy as np
import time

try:
    from pal.products.qcar import QCarCameras
    MOCK_MODE = False
except ImportError:
    print("[WARNING] QCarCameras not available. Using mock frame.")
    MOCK_MODE = True


def capture():
    filename = sys.argv[1] if len(sys.argv) > 1 else "calibration_frame"
    filename = f"{filename}.png"

    if MOCK_MODE:
        # Generate synthetic frame for development
        frame = np.zeros((616, 820, 3), dtype=np.uint8)
        cv2.putText(frame, "MOCK - No Camera Hardware",
                    (180, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(filename, frame)
        print(f"[MockCapture] Saved mock frame to: {filename}")
        return

    print("[Capture] Initializing front CSI camera (820x616 @ 30fps)...")

    # Pop display vars for nvarguscamerasrc fix
    display_bak = os.environ.pop("DISPLAY", None)
    xauth_bak = os.environ.pop("XAUTHORITY", None)

    try:
        cameras = QCarCameras(
            frameWidth=820,
            frameHeight=616,
            frameRate=30,
            enableFront=True,
            enableRight=False,
            enableBack=False,
            enableLeft=False
        )
    finally:
        if display_bak:
            os.environ["DISPLAY"] = display_bak
        else:
            os.environ["DISPLAY"] = ":1"
        if xauth_bak:
            os.environ["XAUTHORITY"] = xauth_bak

    print("[Capture] Camera ready. Warming up (3 frames)...")

    # Warm up: discard first few frames (often blank/dark)
    for _ in range(3):
        cameras.readAll()
        time.sleep(0.1)

    # Capture actual frame
    cameras.readAll()
    frame = cameras.csi[2].imageData  # Front camera = ID 2

    if frame is not None and frame.max() > 10:
        cv2.imwrite(filename, frame)
        print(f"[Capture] ✓ Saved calibration frame to: {filename}")
        print(f"[Capture]   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"[Capture]   Now open this in an image editor and pick")
        print(f"[Capture]   your 4 perspective transform points for config.py")
    else:
        print("[Capture] ✗ Frame was blank. Try again.")

    cameras.terminate()
    print("[Capture] Camera terminated.")


if __name__ == "__main__":
    capture()
