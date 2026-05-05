#!/usr/bin/env python3
"""
QCar2 Lane Detection — Perspective Transform Calibration Tool

Usage:
    python3 calibrate.py              # Capture live frame and calibrate
    python3 calibrate.py <image.jpg>  # Calibrate from a saved image

Instructions:
    1. Place the QCar2 centered on a straight section of track
    2. Run this script to capture a frame
    3. Click 4 points on the frame to define the BEV source trapezoid:
       Point 1: Top-left of the road area (left lane, far away)
       Point 2: Top-right of the road area (right lane, far away)
       Point 3: Bottom-right of the road area (right lane, close)
       Point 4: Bottom-left of the road area (left lane, close)
    4. The BEV preview updates live so you can see the result
    5. Press 's' to save calibration values, 'r' to reset, 'q' to quit
"""
import sys
import os
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
import numpy as np
import cv2
import time

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────
# Globals for mouse callback
# ──────────────────────────────────────────────────────────────────────────
clicked_points = []
frame_display = None
original_frame = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"  Point {len(clicked_points)}: ({x}, {y})")
        redraw()


def redraw():
    global frame_display
    frame_display = original_frame.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    labels = ["TL", "TR", "BR", "BL"]

    for i, pt in enumerate(clicked_points):
        cv2.circle(frame_display, tuple(pt), 8, colors[i], -1)
        cv2.putText(frame_display, labels[i], (pt[0]+12, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(frame_display, tuple(clicked_points[i]),
                     tuple(clicked_points[i+1]), (0, 255, 255), 2)
    if len(clicked_points) == 4:
        cv2.line(frame_display, tuple(clicked_points[3]),
                 tuple(clicked_points[0]), (0, 255, 255), 2)


def capture_frame():
    """Capture a frame from the front CSI camera."""
    from pal.products.qcar import QCarCameras

    print("[Calibrate] Initializing front CSI camera...")
    cameras = QCarCameras(
        frameWidth=820,
        frameHeight=410,
        frameRate=30,
        enableFront=True
    )

    # Warmup
    print("[Calibrate] Warming up camera (15 frames)...")
    for _ in range(15):
        cameras.readAll()
        time.sleep(0.05)

    # Capture a good frame
    frame = None
    for attempt in range(30):
        cameras.readAll()
        img = cameras.csiFront.imageData
        if img is not None and img.max() > 10:
            frame = img.copy()
            break
        time.sleep(0.1)

    cameras.terminate()

    if frame is None:
        print("[ERROR] Could not capture a valid frame!")
        sys.exit(1)

    print(f"[Calibrate] Captured frame: {frame.shape}")
    return frame


def show_bev_preview(frame, src_pts):
    """Show bird's-eye view preview with the given source points."""
    h, w = frame.shape[:2]
    dst_pts = np.float32([
        [100, 0], [720, 0], [720, h], [100, h]
    ])
    M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
    bev = cv2.warpPerspective(frame, M, (w, h))

    # Also show edge detection on BEV
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 4)
    combined = cv2.bitwise_or(edges, adapt)
    combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    # Stack BEV and edge images
    panel = np.hstack([bev, combined_color])
    cv2.imshow("BEV Preview (Color | Edges)", panel)


def main():
    global original_frame, frame_display, clicked_points

    # Get frame
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        print(f"[Calibrate] Loading image: {sys.argv[1]}")
        original_frame = cv2.imread(sys.argv[1])
    else:
        original_frame = capture_frame()

    # Save the raw frame for reference
    ref_path = os.path.join(SAVE_DIR, "calibration_frame.jpg")
    cv2.imwrite(ref_path, original_frame)
    print(f"[Calibrate] Saved reference frame: {ref_path}")

    h, w = original_frame.shape[:2]
    print(f"\n[Calibrate] Frame size: {w}×{h}")
    print("Click 4 points: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("Press 'r' to reset, 's' to save, 'q' to quit\n")

    # Restore DISPLAY for GUI
    os.environ["DISPLAY"] = ":1"

    frame_display = original_frame.copy()
    cv2.namedWindow("Calibrate — Click 4 points")
    cv2.setMouseCallback("Calibrate — Click 4 points", mouse_callback)

    while True:
        cv2.imshow("Calibrate — Click 4 points", frame_display)

        if len(clicked_points) == 4:
            show_bev_preview(original_frame, clicked_points)

        key = cv2.waitKey(50) & 0xFF

        if key == ord('r'):
            clicked_points = []
            frame_display = original_frame.copy()
            cv2.destroyWindow("BEV Preview (Color | Edges)")
            print("[Calibrate] Reset — click 4 new points")

        elif key == ord('s') and len(clicked_points) == 4:
            pts = np.array(clicked_points)
            print("\n" + "="*60)
            print("  ✅ CALIBRATED src_points — Paste into config.py:")
            print("="*60)
            print(f"    src_points: np.ndarray = field(default_factory=lambda: np.float32([")
            print(f"        [{pts[0][0]:3d}, {pts[0][1]:3d}],   # Top-left")
            print(f"        [{pts[1][0]:3d}, {pts[1][1]:3d}],   # Top-right")
            print(f"        [{pts[2][0]:3d}, {pts[2][1]:3d}],   # Bottom-right")
            print(f"        [{pts[3][0]:3d}, {pts[3][1]:3d}],   # Bottom-left")
            print(f"    ]))")
            print("="*60)

            # Save calibration image
            cal_path = os.path.join(SAVE_DIR, "calibration_result.jpg")
            cv2.imwrite(cal_path, frame_display)
            print(f"Saved: {cal_path}")
            break

        elif key == ord('q'):
            print("[Calibrate] Cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
