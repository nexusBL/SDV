import sys
import os
import time
import math
import numpy as np
import cv2

# Headless EGL Setup for Jetson
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

# Add Quanser libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
# Add Path Planning directory to access PurePursuitController
sys.path.insert(0, '/home/nvidia/Desktop/SDV_workspace/scripts/Path Planning')

from pal.products.qcar import QCar
from hal.content.qcar_functions import QCarEKF
from route_planner import RoutePlanner
from control.pure_pursuit import PurePursuitController

# Mock configuration to instantiate Pure Pursuit
class MockPurePursuitConfig:
    lookahead_distance = 0.40        # Tighter lookahead for strict lane following
    base_velocity = 0.15             # Cruise speed slightly more confident
    min_velocity = 0.10              # Turn speed (higher to overcome motor friction)
    curve_slowdown_factor = 0.3      # Braking aggressiveness in turns
    steering_alpha = 0.3             # EMA Smoothing
    max_steering_rate = 0.3          # Limit jerky steering moves
    wheelbase = 0.256                # QCar axle distance

class MockAppConfig:
    pure_pursuit = MockPurePursuitConfig()

def main():
    # 1. Initialize Hardware & Modules
    print("--- Initializing QCar Pure Pursuit Navigator ---")
    my_car = QCar(readMode=1, frequency=100)
    planner = RoutePlanner()
    
    # User Input
    print("\nAvailable Nodes:", [n[0] for n in planner.get_all_nodes()])
    try:
        start_node = int(input("Enter Start Node ID: "))
        dest_node = int(input("Enter Destination Node ID: "))
    except ValueError:
        print("Invalid input. Defaulting to 0 -> 10")
        start_node, dest_node = 0, 10

    # Calculate shortest path
    path = planner.calculate_route(start_node, dest_node)
    if path is None: 
        print("Fatal: Could not find path.")
        return
        
    print(f"Route Planned: {planner.node_sequence}")
    print(f"Path Waypoints count: {path.shape[1]}")
    
    # 2. Initialize Controllers
    start_x = path[0, 0]
    start_y = path[1, 0]
    next_x = path[0, 1]
    next_y = path[1, 1]
    start_theta = math.atan2(next_y - start_y, next_x - start_x)
    
    # Initialize Dead-Reckoning pose variables
    current_x = start_x
    current_y = start_y
    current_th = start_theta
    
    pursuit = PurePursuitController(MockAppConfig())
    pursuit.set_path(path)
    
    end_x = path[0, -1]
    end_y = path[1, -1]
    
    print(f"Spawn Pose Set: X:{start_x:.3f}, Y:{start_y:.3f}, Th:{math.degrees(start_theta):.1f} deg")
    print("Starting Autonomous Drive in 3 seconds...")
    time.sleep(3.0)

    # 3. Main Control Loop
    state = "DRIVING"
    steer = 0.0 
    speed = 0.0
    
    t_last = time.time()
    t_start_loop = time.time()
    loop_count = 0
    
    try:
        while True:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            
            my_car.read()     # Clear hardware queues
            
            # Allow initialization delay before driving
            if (t_now - t_start_loop) < 1.0:
                my_car.read_write_std(0.0, 0.0, np.zeros(8, dtype=np.float64))
                continue
            
            # --- Dead-Reckoning Bypass ---
            # Bypassing the faulty hardware encoders and IMU completely.
            # Integrating the commanded speed and steering mathematically.
            actual_speed = speed if state == "DRIVING" else 0.0
            d_theta = (actual_speed * math.tan(steer)) / pursuit.cfg.wheelbase
            
            current_x += actual_speed * math.cos(current_th) * dt
            current_y += actual_speed * math.sin(current_th) * dt
            current_th += d_theta * dt
            
            # Normalize heading to [-pi, pi]
            current_th = (current_th + math.pi) % (2 * math.pi) - math.pi
            
            current_pose = (current_x, current_y, current_th)
            
            dist_to_goal = math.hypot(end_x - current_x, end_y - current_y)
            
            if state == "DRIVING":
                if dist_to_goal < 0.2:
                    print("\n🏁 FINAL DESTINATION REACHED!")
                    state = "STOPPED"
                    speed = 0.0
                    steer = 0.0
                    break
                else:
                    steer, speed, lookahead_pt = pursuit.compute(current_pose)
            else:
                speed = 0.0
                steer = 0.0
                break

            LEDS = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=np.float64)
            # Physical QCar servo is wired backward: mathematically negative (Right) must be sent as positive to HW
            my_car.read_write_std(speed, steer * -1.0, LEDS)
            
            # Debug Telemetry
            if loop_count % 10 == 0:
                print(f"X:{current_x:6.2f}  Y:{current_y:6.2f}  Th:{math.degrees(current_th):+6.1f}° | Cmd V:{speed:.2f} Str:{steer:+.2f}")
                
            loop_count += 1
            time.sleep(max(0, 0.033 - (time.time() - t_now)))

    except KeyboardInterrupt:
        print("User Aborted.")
    finally:
        my_car.read_write_std(0.0, 0.0, np.zeros(8))
        my_car.terminate()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
