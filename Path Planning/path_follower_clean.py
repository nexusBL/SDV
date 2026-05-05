#!/usr/bin/env python3
import time
import math
import numpy as np

# Use native Quanser libraries directly!
from pal.products.qcar import QCar
from hal.products.mats import SDCSRoadMap
from hal.content.qcar_functions import QCarEKF

def pure_pursuit_steering(x, y, th, path, lookahead_distance=0.8, wheelbase=0.256):
    """Calculates steering angle directly using raw geometry without custom wrappers"""
    
    # 1. Find the closest point index on the path
    min_dist = float('inf')
    closest_idx = 0
    for i in range(path.shape[1]):
        px, py = path[0, i], path[1, i]
        dist = math.hypot(px - x, py - y)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            
    # 2. Find the lookahead point
    lookahead_idx = path.shape[1] - 1
    for i in range(closest_idx, path.shape[1]):
        px, py = path[0, i], path[1, i]
        if math.hypot(px - x, py - y) >= lookahead_distance:
            lookahead_idx = i
            break
            
    lx, ly = path[0, lookahead_idx], path[1, lookahead_idx]
    
    # 3. Transform the lookahead point into the car's local coordinate frame
    dx = lx - x
    dy = ly - y
    local_x = math.cos(th) * dx + math.sin(th) * dy
    local_y = -math.sin(th) * dx + math.cos(th) * dy
    
    # 4. Calculate pure pursuit curvature
    ld_sq = local_x**2 + local_y**2
    if ld_sq == 0.0:
        return 0.0
    curvature = 2.0 * local_y / ld_sq
    
    # 5. Ackerman steering command
    steering_angle = math.atan(curvature * wheelbase)
    
    # Limit steering angle to physical bounds (about 30 degrees = 0.5 rad)
    return max(min(steering_angle, 0.5), -0.5)


def main():
    print("==================================================")
    print(" NATIVE QUANSER PATH DRIVING DEMO ")
    print("==================================================")
    
    # The route nodes
    nodeSequence = [0, 5, 11]
    
    print("1. Generating SDCSRoadMap path...")
    roadmap = SDCSRoadMap(leftHandTraffic=True, useSmallMap=True)
    path = roadmap.generate_path(nodeSequence=nodeSequence)
    
    if path is None:
        print("Error: Could not calculate route.")
        return

    # Extract the exact physical starting coordinates for Node 0
    start_pose = roadmap.get_node_pose(nodeSequence[0]).flatten()
    print(f"   Start Coordinate: X={start_pose[0]:.2f}, Y={start_pose[1]:.2f}, Angle={math.degrees(start_pose[2]):.0f}deg")
    
    # Instead of the buggy KinematicTracker, we use Quanser's native
    # mathematically-perfect Extended Kalman Filter (EKF) to track the car!
    print("2. Initializing Hardware & QCarEKF Tracker...")
    qcar = QCar(readMode=1, frequency=100)
    
    # Initialize EKF at Node 0 coordinates
    ekf = QCarEKF(x_0=start_pose)
    
    # Initialize control variables
    throttle = 0.12  # Slow, safe driving speed
    steering = 0.0
    
    print("\\n🚀 INITIALIZING SENSORS (Please wait 1 second...)")
    
    try:
        with qcar:
            last_t = time.time()
            start_t = last_t
            loop_count = 0
            
            while True:
                current_t = time.time()
                dt = current_t - last_t
                last_t = current_t
                
                # Prevent huge integration jumps
                if dt > 0.1:
                    dt = 0.01  
                
                # --- READ HARDWARE SENSORS ---
                qcar.read()
                
                # --- UPDATE ODOMETRY ---
                # NOTE: For the first 1 second, we pass 0 steering/throttle to let the EKF stabilize
                if (current_t - start_t) < 1.0:
                    ekf.update([qcar.motorTach, 0.0], dt, None, qcar.gyroscope[2])
                    qcar.write(0.0, 0.0)
                    time.sleep(0.01)
                    continue

                if loop_count == 0:
                    print("\\n🚗 SENSORS STABLE - ENGAGING MOTORS! (Press Ctrl+C to stop)\\n")
                    
                ekf.update([qcar.motorTach, steering], dt, None, qcar.gyroscope[2])
                
                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                
                # --- CHECK DESTINATION ---
                end_x, end_y = path[0, -1], path[1, -1]
                dist_to_goal = math.hypot(end_x - x, end_y - y)
                if dist_to_goal < 0.20:
                    print("\\n🏁 ARRIVED AT DESTINATION! Stopping motors.")
                    qcar.write(0.0, 0.0)
                    break
                    
                # --- CALCULATE STEERING AND OVERRIDE ---
                # Find the closest point index on the path (re-implemented here for debugging prints)
                min_dist = float('inf')
                closest_idx = 0
                for i in range(path.shape[1]):
                    px, py = path[0, i], path[1, i]
                    dist = math.hypot(px - x, py - y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                        
                steering = pure_pursuit_steering(x, y, th, path)
                
                # INVERT THE STEERING AUTOMATICALLY - usually QCar uses inverted steering coordinates!
                steering = -steering  
                
                qcar.write(throttle, steering)
                
                if loop_count % 10 == 0:
                    print(f"Pose:[X:{x:6.2f} Y:{y:6.2f} Th:{math.degrees(th):5.0f}°] "
                          f"| Closest_Idx: {closest_idx:3d}/{path.shape[1]-1} "
                          f"| Dist-to-Goal: {dist_to_goal:5.2f}m "
                          f"| CmdSteer: {steering:5.2f} rad")
                
                loop_count += 1
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\\n🛑 Manual Emergency Stop Triggered.")
    finally:
        # Guarantee hardware shutdown on exit
        try:
            qcar.write(0.0, 0.0)
        except Exception:
            pass
        print("Car stopped safely.")

if __name__ == '__main__':
    main()
