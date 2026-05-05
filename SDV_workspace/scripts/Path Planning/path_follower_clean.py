#!/usr/bin/env python3
"""
path_follower_clean.py
======================
QCar2 Pure Pursuit path follower using a fully hardcoded path table.

All 120 valid paths were pre-computed from SDCSRoadMap(leftHandTraffic=True,
useSmallMap=True) and stored in hardcoded_paths.py — no live roadmap needed.

USAGE
-----
1. Put the physical car at Node 0, facing the correct way (Y axis / +90 deg).
2. Run:  python3 path_follower_clean.py
3. If the car does weird things physically, tweak the Hardware Knobs below.
"""

import time
import math
import numpy as np

from pal.products.qcar import QCar
from hal.content.qcar_functions import QCarEKF

# ─────────────────────────────────────────────────────────────────────────────
# USER RUNTIME SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
START_NODE   = 0    # node the car is physically placed at
END_NODE     = 5    # destination node

THROTTLE     = 0.12 # m/s command (keep ≤ 0.15 on small map)
LOOKAHEAD    = 0.35 # metres (smaller = tighter tracking)
WHEELBASE    = 0.256 # metres (QCar2 wheelbase)
GOAL_RADIUS  = 0.20 # metres (stop when this close to final point)

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE CALIBRATION KNOBS (QCar2 Specific)
# ─────────────────────────────────────────────────────────────────────────────
# The previous telemetry showed the car turning right when commanded negatively,
# but the EKF heading spun wildly and the Y position went backwards.
# This means the hardware sensors on your specific QCar are inverted.

# 1. PHYSICAL STEERING INVERSION
# Positive = LEFT in pure pursuit.
STEERING_CMD_SIGN = 1

# 2. BYPASS FAULTY SENSORS (Dead Reckoning Mode)
# We need closed-loop gyro feedback to stay on the path in reality.
USE_DEAD_RECKONING = False

# (Only used if USE_DEAD_RECKONING = False)
# The first run's logs showed EKF Y going backwards and EKF heading spinning left rapidly 
# even though the car drove straight/right! These sensors are inverted on this car.
ENCODER_SIGN = -1   # (we will just take abs() to be safe)
GYRO_SIGN    = -1   # flips the inverted Z-axis gyro
# ─────────────────────────────────────────────────────────────────────────────


def pure_pursuit(x, y, th, path, prev_idx, lookahead=LOOKAHEAD, wb=WHEELBASE):
    """
    Pure Pursuit controller.

    Returns
    -------
    steering_angle : float  (radians, mathematical positive = LEFT)
    new_closest_idx: int    
    local_y        : float  (diagnostic)
    """
    n = path.shape[1]

    # 1. Forward-only closest-index search.
    search_end = min(prev_idx + 150, n)
    min_dist   = float('inf')
    closest    = prev_idx
    for i in range(prev_idx, search_end):
        d = math.hypot(path[0, i] - x, path[1, i] - y)
        if d < min_dist:
            min_dist = d
            closest  = i

    # 2. Lookahead point.
    la_idx = n - 1
    for i in range(closest, n):
        if math.hypot(path[0, i] - x, path[1, i] - y) >= lookahead:
            la_idx = i
            break

    lx, ly = path[0, la_idx], path[1, la_idx]

    # 3. Transform lookahead into car's local frame.
    dx      =  lx - x
    dy      =  ly - y
    local_x =  math.cos(th) * dx + math.sin(th) * dy
    local_y = -math.sin(th) * dx + math.cos(th) * dy

    # 4. Pure-pursuit curvature.
    ld_sq = local_x**2 + local_y**2
    if ld_sq < 1e-6:
        return 0.0, closest, local_y

    curvature = 2.0 * local_y / ld_sq
    steer     = math.atan(curvature * wb)
    steer     = max(min(steer, 0.5), -0.5)
    
    return steer, closest, local_y


def main():
    # ── Load hardcoded path ───────────────────────────────────────────────
    try:
        from hardcoded_paths import ALL_PATHS, NODE_POSES
    except ImportError:
        print("ERROR: hardcoded_paths.py not found in the same folder!")
        return

    key = (START_NODE, END_NODE)
    if key not in ALL_PATHS:
        print(f"ERROR: No path exists for ({START_NODE} → {END_NODE}).")
        return

    path = ALL_PATHS[key]
    print("=" * 60)
    print(f"  PATH FOLLOWER  |  Node {START_NODE} → Node {END_NODE}")
    print(f"  Path points  : {path.shape[1]}")
    print(f"  Lookahead    : {LOOKAHEAD} m   Throttle: {THROTTLE}")
    print(f"  DEAD_RECKONING: {USE_DEAD_RECKONING}")
    print("=" * 60)

    # ── Start pose from node table ────────────────────────────────────────
    nx, ny, nth = NODE_POSES[START_NODE]
    
    # State variables for our mathematical odometry Tracker
    x, y, th = nx, ny, nth

    print(f"\n  Init pose: X={nx:.3f}  Y={ny:.3f}  Th={math.degrees(nth):.1f}°")

    # ── Hardware init ─────────────────────────────────────────────────────
    qcar = QCar(readMode=1, frequency=100)
    
    if not USE_DEAD_RECKONING:
        ekf = QCarEKF(x_0=np.array([[nx], [ny], [nth]]))

    steering_math  = 0.0
    closest_state  = 0

    print("\n🚀 Stabilising sensors (1 s) …")

    try:
        with qcar:
            t_last  = time.time()
            t_start = t_last
            loop    = 0

            while True:
                t_now = time.time()
                dt    = min(t_now - t_last, 0.1)
                t_last = t_now

                qcar.read()

                # ── Warm-up ───────────────────────────────────────────────
                if (t_now - t_start) < 1.0:
                    if not USE_DEAD_RECKONING:
                        ekf.update([qcar.motorTach, 0.0], dt, None, qcar.gyroscope[2])
                    qcar.write(0.0, 0.0)
                    time.sleep(0.01)
                    continue

                if loop == 0:
                    print("🚗 MOTORS ENGAGED  (Ctrl+C to stop)\n")

                # ── 1. UPDATE ODOMETRY ────────────────────────────────────
                if USE_DEAD_RECKONING:
                    speed = THROTTLE
                    d_theta = (speed * math.tan(steering_math)) / WHEELBASE
                    x  += speed * math.cos(th) * dt
                    y  += speed * math.sin(th) * dt
                    th += d_theta * dt
                    th = (th + math.pi) % (2 * math.pi) - math.pi
                else:
                    # QCar2 hardware telemetry from run 1 proved encoder returns negative for forward
                    # and gyro returns positive for right turns. We MUST fix these for EKF.
                    tach = abs(qcar.motorTach)  # we only drive forward
                    gyro = qcar.gyroscope[2] * GYRO_SIGN
                    
                    ekf.update([tach, steering_math], dt, None, gyro)
                    x  = ekf.x_hat[0, 0]
                    y  = ekf.x_hat[1, 0]
                    th = ekf.x_hat[2, 0]

                # ── 2. CHECK ARRIVAL ──────────────────────────────────────
                dist_goal = math.hypot(path[0, -1] - x, path[1, -1] - y)
                if dist_goal < GOAL_RADIUS:
                    print("\n🏁 ARRIVED!  Stopping.")
                    qcar.write(0.0, 0.0)
                    break

                # ── 3. PURE PURSUIT (Mathematical) ────────────────────────
                steering_math, closest_state, _ = pure_pursuit(
                    x, y, th, path, closest_state
                )

                # ── 4. HARDWARE WRITE (Physical Inversion) ────────────────
                # The QCar actual steering servo is wired backward relative 
                # to the mathematical positive=left convention.
                physical_steering = steering_math * STEERING_CMD_SIGN
                
                qcar.write(THROTTLE, physical_steering)

                # ── 5. TELEMETRY ──────────────────────────────────────────
                if loop % 10 == 0:
                    print(f"  X:{x:6.3f} Y:{y:6.3f} Th:{math.degrees(th):+6.1f}° "
                          f"| idx:{closest_state:4d}/{path.shape[1]-1} "
                          f"| goal:{dist_goal:.2f}m "
                          f"| steer_cmd:{physical_steering:+.3f}")

                loop += 1
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        try:
            qcar.write(0.0, 0.0)
        except Exception:
            pass
        print("Car stopped safely.")

if __name__ == '__main__':
    main()
