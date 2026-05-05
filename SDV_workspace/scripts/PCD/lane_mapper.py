#!/usr/bin/env python3
"""
lane_mapper.py — Drive QCar2 manually while building a global lane point cloud.
Shows live color+depth feed. WASD control via the camera window.
Press Q to save map and quit.
"""
import os, sys
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

import pyrealsense2 as rs
import numpy as np
import cv2
import time

from pal.products.qcar import QCar
from odometry import QCarOdometry
from lane_filter import LaneFilter, get_vertices_array, get_color_array


def main():
    # 1. Hardware Initialization
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    # 2. QCar & Odometry Initialization
    car = QCar(readMode=1, frequency=100)
    odo = QCarOdometry(wheelbase=0.256, cps_to_mps=6.866e-6)
    lane_filter = LaneFilter()

    # 3. Accumulated Map & Control
    global_points = []
    throttle = 0.0
    steering = 0.0

    print("=" * 50)
    print("  LANE MAPPER — Live Camera Feed")
    print("=" * 50)
    print("Click on the camera window, then use keys:")
    print("  W / S : Throttle Up / Down")
    print("  A / D : Steer Left / Right")
    print("  Space : Stop (Reset throttle/steer)")
    print("  Q     : Finish and Save Map")
    print("=" * 50)

    try:
        while True:
            # --- Read Sensor Data ---
            car.read()
            current_state = odo.update(
                car.motorTach,
                car.steeringBias if hasattr(car, 'steeringBias') else 0,
            )

            # --- Read Camera Data ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # --- Extract & Filter Local Points ---
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            vtx = get_vertices_array(points)
            colors = get_color_array(color_frame)

            road_vtx = lane_filter.filter_road_points(vtx, z_min=0.3, z_max=1.2)
            lane_vtx, _ = lane_filter.extract_lane_candidates(vtx, colors)

            # --- Transform to Global frame ---
            T_world_car = odo.get_transform_matrix()

            for p in lane_vtx[::10]:  # Downsample by 10
                local_p = np.array([p[2], -p[0], 0, 1])
                global_p = T_world_car @ local_p
                global_points.append(global_p[:3])

            # --- Live Visualization ---
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_cm = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            # HUD overlay on color image
            hud = color_image.copy()
            cv2.putText(hud, f"Throttle: {throttle:+.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hud, f"Steer: {steering:+.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hud, f"Lane Pts: {len(global_points)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(hud, f"Pos: ({current_state[0]:.2f}, {current_state[1]:.2f})",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            combined = np.hstack((hud, depth_cm))
            cv2.imshow('Lane Mapper (Color | Depth)', combined)

            # --- Keyboard Control (from cv2 window) ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                throttle += 0.01
            elif key == ord('s'):
                throttle -= 0.01
            elif key == ord('a'):
                steering += 0.05
            elif key == ord('d'):
                steering -= 0.05
            elif key == 32:  # Space
                throttle = 0.0
                steering = 0.0
            elif key == ord('q'):
                print("\nFinishing...")
                break

            # Safety limits
            throttle = np.clip(throttle, -0.15, 0.15)
            steering = np.clip(steering, -0.5, 0.5)

            # Send commands to QCar
            car.write(throttle, steering, [0, 0, 0, 0, 0, 0, 0, 0])

    finally:
        # Save Global Map
        if global_points:
            print(f"\nSaving final map with {len(global_points)} points...")
            with open("global_lane_map.txt", "w") as f:
                for p in global_points:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
            print("Map saved to global_lane_map.txt")
        else:
            print("\nNo points collected.")

        car.terminate()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Done. Exited cleanly.")


if __name__ == "__main__":
    main()
