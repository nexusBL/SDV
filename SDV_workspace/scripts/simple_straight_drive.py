"""
Simple QCar2 Straight Drive with Basic Obstacle Avoidance
- Drives in a straight line by default
- Only deviates when obstacles are detected
- Uses simple straight-line avoidance maneuvers
- Minimal complexity for initial testing
"""

# Fix nvarguscamerasrc EGL authorization issues by forcing headless NVMM capture:
import os
import sys
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
import numpy as np
import time

from pal.products.qcar import QCar, QCarLidar, IS_PHYSICAL_QCAR

# QLabs setup if simulated
if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()

def check_front_obstacle(angles, distances, max_distance=0.8):
    """
    Check for obstacles directly in front of the vehicle
    Returns: (obstacle_detected, min_distance, obstacle_side)
    obstacle_side: 'left', 'right', or 'center'
    """
    if len(distances) == 0:
        return False, 5.0, 'center'
    
    # Convert to numpy arrays for easier processing
    # Based on hardware test, front is at approx 3.27 radians (~pi).
    raw_angles = np.array(angles)
    angles = raw_angles - np.pi
    
    # Normalize angles to [-pi, pi]
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    
    distances = np.array(distances)
    
    # Filter valid measurements
    valid_mask = (distances > 0.05) & (distances < 5.0)
    valid_angles = angles[valid_mask]
    valid_distances = distances[valid_mask]
    
    if len(valid_distances) == 0:
        return False, 5.0, 'center'
    
    # Check front arc (±45 degrees)
    front_arc = np.deg2rad(45)
    front_mask = np.abs(valid_angles) < front_arc
    front_distances = valid_distances[front_mask]
    front_angles = valid_angles[front_mask]
    
    if len(front_distances) == 0:
        return False, 5.0, 'center'
    
    # Find minimum distance in front
    min_dist_idx = np.argmin(front_distances)
    min_distance = front_distances[min_dist_idx]
    min_angle = front_angles[min_dist_idx]
    
    # Determine which side the obstacle is on
    if min_angle > np.deg2rad(10):
        obstacle_side = 'left'  # Obstacle on left, we should go right
    elif min_angle < np.deg2rad(-10):
        obstacle_side = 'right'  # Obstacle on right, we should go left
    else:
        obstacle_side = 'center'
    
    obstacle_detected = min_distance < max_distance
    
    return obstacle_detected, min_distance, obstacle_side

def main():
    # ==== Settings ====
    sampleRate = 30
    runTime = 300.0  # 5 minutes
    
    # LiDAR settings
    numMeasurements = 1000
    lidarMeasurementMode = 2
    lidarInterpolationMode = 0
    
    # Control parameters
    BASE_SPEED = 0.08        # Normal forward speed (slowed down)
    AVOIDANCE_SPEED = 0.06   # Speed during avoidance (slowed down)
    TURN_STEERING = 0.5      # Steering angle for avoidance
    OBSTACLE_THRESHOLD = 1.2  # Distance to start avoiding (increased to 1.2m)
    
    # State machine
    state = "FORWARD"
    state_start_time = 0
    last_print_time = 0
    
    print("🚗 Simple Straight Drive with Obstacle Avoidance")
    print(f"- Forward speed: {BASE_SPEED}")
    print(f"- Obstacle threshold: {OBSTACLE_THRESHOLD}m")
    print(f"- Turn steering: {TURN_STEERING}")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Initialize LiDAR
        myLidar = QCarLidar(
            numMeasurements=numMeasurements,
            rangingDistanceMode=lidarMeasurementMode,
            interpolationMode=lidarInterpolationMode
        )
        print("✅ LiDAR initialized")
        
        with QCar(readMode=1, frequency=sampleRate) as myCar:
            print("✅ QCar initialized")
            print("🚀 Starting drive sequence...\n")
            
            t0 = time.time()
            
            while time.time() - t0 < runTime:
                current_time = time.time()
                
                # Read sensors
                myCar.read()
                myLidar.read()
                
                # Check for obstacles
                obstacle_detected, min_distance, obstacle_side = check_front_obstacle(
                    myLidar.angles, myLidar.distances, OBSTACLE_THRESHOLD
                )
                
                # State machine for obstacle avoidance
                throttle = 0.0
                steering = 0.0
                state_info = ""
                
                if state == "FORWARD":
                    if obstacle_detected:
                        # Start avoidance maneuver
                        if obstacle_side == 'left':
                            state = "AVOID_RIGHT"
                        elif obstacle_side == 'right':
                            state = "AVOID_LEFT"
                        else:  # center obstacle
                            state = "AVOID_LEFT"  # Default to left avoidance
                        
                        state_start_time = current_time
                        print(f"🚧 Obstacle detected at {min_distance:.2f}m on {obstacle_side} - starting {state}")
                    
                    # Normal forward driving
                    throttle = BASE_SPEED
                    steering = 0.0
                    state_info = f"FORWARD (clear: {min_distance:.2f}m)"
                    
                elif state == "AVOID_LEFT":
                    # Simple left avoidance: turn left, go straight, turn right back
                    elapsed = current_time - state_start_time
                    
                    if elapsed < 1.5:  # Turn left for 1.5 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = -TURN_STEERING
                        state_info = "TURNING_LEFT"
                    elif elapsed < 3.5:  # Go straight for 2 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = 0.0
                        state_info = "STRAIGHT_AVOID"
                    elif elapsed < 5.0:  # Turn right to return for 1.5 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = TURN_STEERING
                        state_info = "RETURNING_RIGHT"
                    else:
                        # Return to forward
                        state = "FORWARD"
                        print("✅ Left avoidance complete")
                        
                elif state == "AVOID_RIGHT":
                    # Simple right avoidance: turn right, go straight, turn left back
                    elapsed = current_time - state_start_time
                    
                    if elapsed < 1.5:  # Turn right for 1.5 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = TURN_STEERING
                        state_info = "TURNING_RIGHT"
                    elif elapsed < 3.5:  # Go straight for 2 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = 0.0
                        state_info = "STRAIGHT_AVOID"
                    elif elapsed < 5.0:  # Turn left to return for 1.5 seconds
                        throttle = AVOIDANCE_SPEED
                        steering = -TURN_STEERING
                        state_info = "RETURNING_LEFT"
                    else:
                        # Return to forward
                        state = "FORWARD"
                        print("✅ Right avoidance complete")
                
                # Set LEDs based on steering
                LEDs = np.zeros(8)
                if steering > 0.2:
                    LEDs[1] = 1  # Right turn signal
                    LEDs[3] = 1
                elif steering < -0.2:
                    LEDs[0] = 1  # Left turn signal
                    LEDs[2] = 1
                    
                if throttle == 0:
                    LEDs[4] = 1  # Brake lights
                    LEDs[5] = 1
                
                # Apply controls
                myCar.write(throttle, steering, LEDs)
                
                # Print telemetry every 0.5 seconds
                if current_time - last_print_time >= 0.5:
                    last_print_time = current_time
                    print(
                        f'⏱️  {current_time - t0:6.1f}s | '
                        f'{state_info:15s} | '
                        f'Obstacle: {"YES" if obstacle_detected else "NO":3s} '
                        f'({min_distance:.2f}m {obstacle_side:6s}) | '
                        f'T:{throttle:4.2f} S:{steering:5.2f} | '
                        f'Bat:{myCar.batteryVoltage:4.1f}V'
                    )
                    
                time.sleep(0.03)  # Small delay for stable operation
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user - performing safe shutdown...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🛑 Performing emergency stop...")
        
    finally:
        # Safe shutdown
        print("Shutting down...")
        try:
            if 'myCar' in locals():
                myCar.write(0.0, 0.0, np.zeros(8))  # Emergency stop
            if 'myLidar' in locals():
                myLidar.terminate()
            print("✅ Safe shutdown complete")
        except Exception as shutdown_error:
            print(f"⚠️  Shutdown error: {shutdown_error}")

if __name__ == "__main__":
    main()
