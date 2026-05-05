#!/usr/bin/env python3
"""
qcar_lane_detection.py — QCar2 Lane Detection + LiDAR Safety
=============================================================
Pipeline:
    Camera Img -> Preprocess -> Detect Lane Lines -> Find Lane Centre
               -> Compare with Car Center -> Generate Steering Command

Steps:
    1. Grayscale + Gaussian Blur
    2. Canny Edge Detection
    3. ROI Trapezoid Mask
    4. Hough Line Detection
    5. Separate Left/Right lines by slope
    6. Fit average line per side
    7. Find lane center (midpoint at bottom of frame)
    8. steer = (lane_center - image_center) / image_center

Usage:
    python3 qcar_lane_detection.py
"""

import numpy as np
import time
import cv2
from pal.products.qcar import QCar, QCarRealSense, QCarLidar


# ══════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════

sampleRate  = 200
runTime     = 300.0
frame_size  = (320, 240)

FORWARD_SPEED = 0.12
MAX_STEER     = 0.5

LIDAR_STOP_DISTANCE = 1.0
LIDAR_STABLE_FRAMES = 3

ALPHA = 0.7

STATE_MOVE = "MOVE"
STATE_STOP = "STOP"


# ══════════════════════════════════════════════════════════════════
#  STEP 1+2: PREPROCESS
# ══════════════════════════════════════════════════════════════════

def preprocess(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


# ══════════════════════════════════════════════════════════════════
#  STEP 3: ROI TRAPEZOID MASK
# ══════════════════════════════════════════════════════════════════

def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    trapezoid = np.array([[
        (0,          h),
        (w,          h),
        (int(w*0.6), int(h*0.6)),
        (int(w*0.4), int(h*0.6)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trapezoid, 255)
    return cv2.bitwise_and(edges, mask)


# ══════════════════════════════════════════════════════════════════
#  STEP 4: HOUGH LINE DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_lines(roi):
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    return lines


# ══════════════════════════════════════════════════════════════════
#  STEP 5+6: FIT LANE LINES
# ══════════════════════════════════════════════════════════════════

def fit_lane_lines(lines, img_shape):
    h, w   = img_shape[:2]
    left_pts  = []
    right_pts = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            right_pts.extend([(x1, y1), (x2, y2)])
        else:
            left_pts.extend([(x1, y1), (x2, y2)])

    y_bottom = h
    y_top    = int(h * 0.6)

    def fit_line(pts):
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        try:
            a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        except Exception:
            return None
        x_bottom = int(a * y_bottom + b)
        x_top    = int(a * y_top    + b)
        return (x_bottom, y_bottom, x_top, y_top)

    return fit_line(left_pts), fit_line(right_pts)


# ══════════════════════════════════════════════════════════════════
#  STEP 7: FIND LANE CENTER
# ══════════════════════════════════════════════════════════════════

def compute_lane_center(left_line, right_line, img_shape):
    w          = img_shape[1]
    img_center = w // 2
    offset     = w // 4

    if left_line is not None and right_line is not None:
        center     = (left_line[0] + right_line[0]) // 2
        confidence = 1.0
    elif left_line is not None:
        center     = left_line[0] + offset
        confidence = 0.5
    elif right_line is not None:
        center     = right_line[0] - offset
        confidence = 0.5
    else:
        center     = img_center
        confidence = 0.0

    center = int(np.clip(center, 0, w - 1))
    return center, confidence


# ══════════════════════════════════════════════════════════════════
#  STEP 8: GENERATE STEERING
# ══════════════════════════════════════════════════════════════════

def compute_steering(lane_center, image_center, confidence):
    if confidence == 0.0:
        return 0.0
    deviation = lane_center - image_center
    steer     = deviation / image_center
    return float(np.clip(steer, -MAX_STEER, MAX_STEER))


# ══════════════════════════════════════════════════════════════════
#  LIDAR FRONT DISTANCE
# ══════════════════════════════════════════════════════════════════

def lidar_front_distance(angles, distances):
    arr   = np.array(distances)
    ang   = np.degrees(np.array(angles))
    front = arr[(ang > -15) & (ang < 15) & (arr > 0.05) & (arr < 10.0)]
    return float(np.min(front)) if len(front) > 0 else np.inf


# ══════════════════════════════════════════════════════════════════
#  DEBUG VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def draw_debug(frame, left_line, right_line, lane_center,
               image_center, steer, confidence, front_dist, state):
    vis = cv2.resize(frame, frame_size)
    h, w = vis.shape[:2]

    sx = frame_size[0] / frame.shape[1]
    sy = frame_size[1] / frame.shape[0]

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(vis,
                 (int(x1*sx), int(y1*sy)),
                 (int(x2*sx), int(y2*sy)),
                 (255, 0, 0), 2)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(vis,
                 (int(x1*sx), int(y1*sy)),
                 (int(x2*sx), int(y2*sy)),
                 (0, 0, 255), 2)

    lc = int(lane_center  * sx)
    ic = int(image_center * sx)
    cv2.circle(vis, (lc, h-15), 6, (0, 255, 0),  -1)
    cv2.circle(vis, (ic, h-15), 6, (0, 255, 255), -1)
    cv2.line(vis, (ic, h-15), (lc, h-15), (255, 255, 0), 2)

    sc = (0,255,0) if state == STATE_MOVE else (0,0,255)
    cc = (0,255,0) if confidence==1.0 else (0,200,255) if confidence==0.5 else (0,0,255)
    lc_col = (0,0,255) if front_dist < LIDAR_STOP_DISTANCE else (0,255,0)

    cv2.putText(vis, f"State : {state}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
    cv2.putText(vis, f"Steer : {steer:+.3f}",
                (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    cv2.putText(vis, f"Conf  : {confidence:.1f}",
                (5, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, 1)
    cv2.putText(vis, f"LiDAR : {front_dist*100:.0f}cm",
                (5, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lc_col, 1)
    cv2.putText(vis, "Blue=Left Red=Right Green=LaneC Cyan=ImgC",
                (5, h-3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1)

    return vis


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   QCar2 Lane Detection + LiDAR Safety               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Forward speed : {FORWARD_SPEED}")
    print(f"  Max steer     : {MAX_STEER}")
    print(f"  LiDAR stop    : {LIDAR_STOP_DISTANCE}m")
    print(f"  Steer smooth  : alpha={ALPHA}")
    print()

    with QCar(readMode=1, frequency=sampleRate) as myCar, \
         QCarRealSense(mode='RGB, Depth, IR') as cam:

        lidar = QCarLidar(
            numMeasurements=720,
            rangingDistanceMode=2,
            interpolationMode=0
        )

        print("[QCar] Hardware initialized ✅")

        t0         = time.time()
        prev_steer = 0.0
        state      = STATE_MOVE
        stop_count = 0
        move_count = 0

        while time.time() - t0 < runTime:

            # READ SENSORS
            myCar.read()
            cam.read_RGB()
            lidar.read()

            rgb_raw      = cam.imageBufferRGB.copy()
            image_center = rgb_raw.shape[1] // 2

            # LIDAR SAFETY
            front_dist = lidar_front_distance(lidar.angles, lidar.distances)

            if front_dist < LIDAR_STOP_DISTANCE:
                stop_count += 1
                move_count  = 0
            else:
                move_count += 1
                stop_count  = 0

            if stop_count >= LIDAR_STABLE_FRAMES:
                state = STATE_STOP
            elif move_count >= LIDAR_STABLE_FRAMES:
                state = STATE_MOVE

            # LANE DETECTION PIPELINE
            edges      = preprocess(rgb_raw)           # Step 1+2
            roi        = region_of_interest(edges)     # Step 3
            lines      = detect_lines(roi)             # Step 4
            left_line, right_line = fit_lane_lines(    # Step 5+6
                lines, rgb_raw.shape
            )
            lane_center, confidence = compute_lane_center(  # Step 7
                left_line, right_line, rgb_raw.shape
            )
            raw_steer = compute_steering(              # Step 8
                lane_center, image_center, confidence
            )

            # Smooth steering
            smooth_steer = ALPHA * prev_steer + (1 - ALPHA) * raw_steer
            prev_steer   = smooth_steer

            # ACTUATION
            throttle = 0.0
            steer    = 0.0
            LEDs     = np.zeros(8)

            if state == STATE_MOVE:
                throttle = FORWARD_SPEED
                steer    = smooth_steer
                if steer >  0.1: LEDs[0] = LEDs[2] = 1
                if steer < -0.1: LEDs[1] = LEDs[3] = 1

            elif state == STATE_STOP:
                throttle    = 0.0
                steer       = 0.0
                LEDs[4] = LEDs[5] = 1

            myCar.write(throttle, steer, LEDs)

            # TERMINAL PRINT
            print(f"[Lane] center={lane_center:4d}  img={image_center}  "
                  f"raw={raw_steer:+.3f}  smooth={smooth_steer:+.3f}  "
                  f"conf={confidence:.1f}  lidar={front_dist:.2f}m  "
                  f"state={state}")

            # VISUALIZATION
            vis = draw_debug(
                rgb_raw, left_line, right_line,
                lane_center, image_center,
                smooth_steer, confidence,
                front_dist, state
            )
            cv2.imshow("QCar Lane Detection", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        lidar.terminate()
        cv2.destroyAllWindows()
        print("\n[Done] ✅")


if __name__ == "__main__":
    main()
