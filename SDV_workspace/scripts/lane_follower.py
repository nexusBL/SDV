#!/usr/bin/env python3
"""
lane_follower.py — QCar2 Lane Following System
================================================
Complete autonomous lane following with:
- Classical CV lane detection
- LiDAR obstacle avoidance
- Safe fail-safe behaviors

Based on Code 6 architecture with lane detection integrated.

Usage:
    python3 lane_follower.py
"""

import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime

# Add QCar libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

# QCar imports
from pal.products.qcar import QCar, QCarCameras, IS_PHYSICAL_QCAR

# Custom modules
from lane_detector_simple import SimpleLaneDetector
import config_lane as cfg


class LaneFollower:
    """
    Main lane following controller for QCar2.
    Integrates camera, LiDAR, and motor control.
    """
    
    def __init__(self):
        """Initialize all subsystems."""
        print("=" * 70)
        print("QCar2 LANE FOLLOWER — Initializing...")
        print("=" * 70)
        print()
        
        # Initialize QCar
        print("[1/3] Initializing QCar hardware...")
        self.qcar = QCar(
            readMode=1,  # RGB camera + LiDAR
            frequency=30
        )
        print("      QCar initialized!")
        
        # Initialize CSI camera using QCarCameras API
        print("[2/3] Initializing CSI Front Camera (index=2)...")
        try:
            self.camera = QCarCameras(
                frameWidth=cfg.CAMERA_WIDTH,
                frameHeight=cfg.CAMERA_HEIGHT,
                frameRate=30,
                enableFront=True,   # Front camera only
                enableLeft=False,
                enableRight=False,
                enableBack=False
            )
            print("      Camera initialized!")
            print(f"      Resolution: {cfg.CAMERA_WIDTH}x{cfg.CAMERA_HEIGHT}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CSI camera: {e}")
        
        # Initialize lane detector
        print("[3/3] Initializing lane detector...")
        self.detector = SimpleLaneDetector()
        print("      Lane detector ready!")
        print()
        
        # State variables
        self.running = False
        self.frames_without_lane = 0
        self.obstacle_detected = False
        
        # Performance tracking
        self.loop_times = []
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Create output directory for debug frames
        self.output_dir = "/home/nvidia/Desktop/SDV_workspace/scripts/lane_follower_logs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Initialization complete!")
        print(f"Debug logs will be saved to: {self.output_dir}")
        print()
    
    def check_lidar_obstacles(self, lidar_data):
        """
        Check LiDAR data for obstacles in front of the car.
        
        Args:
            lidar_data: Array of 360 distance measurements (in meters)
        
        Returns:
            bool: True if obstacle detected within stop distance
        """
        if lidar_data is None or len(lidar_data) == 0:
            return False
        
        # Check front cone (±LIDAR_FRONT_ANGLE degrees)
        # LiDAR index 0 = front, increases clockwise
        front_indices = list(range(360 - cfg.LIDAR_FRONT_ANGLE, 360)) + \
                       list(range(0, cfg.LIDAR_FRONT_ANGLE + 1))
        
        for idx in front_indices:
            if idx < len(lidar_data):
                distance = lidar_data[idx]
                if 0.0 < distance < cfg.LIDAR_STOP_DISTANCE:
                    return True
        
        return False
    
    def calculate_control(self, deviation, confidence):
        """
        Calculate throttle and steering based on lane deviation.
        
        Args:
            deviation: Lane center deviation (-1.0 to +1.0)
            confidence: Detection confidence (0.0 to 1.0)
        
        Returns:
            (throttle, steering): Control commands
        """
        # If no lane detected or low confidence
        if confidence < 0.3:
            self.frames_without_lane += 1
            
            # If lane lost for too long, stop
            if self.frames_without_lane > cfg.NO_LANE_STOP_FRAMES:
                return 0.0, cfg.FALLBACK_STEERING
            
            # Otherwise, maintain last direction briefly
            return cfg.FORWARD_SPEED * 0.5, cfg.FALLBACK_STEERING
        
        # Lane detected - reset counter
        self.frames_without_lane = 0
        
        # Calculate steering from deviation
        if abs(deviation) < cfg.STEERING_DEADZONE:
            steer = 0.0  # Go straight if within deadzone
        else:
            steer = deviation * cfg.STEERING_GAIN
            steer = np.clip(steer, -cfg.MAX_STEERING_ANGLE, cfg.MAX_STEERING_ANGLE)
        
        # Adjust speed based on steering angle (slow down for turns)
        if abs(steer) > cfg.TURN_THRESHOLD:
            throttle = cfg.TURN_SPEED
        else:
            throttle = cfg.FORWARD_SPEED
        
        return throttle, steer
    
    def run(self):
        """Main control loop."""
        print("=" * 70)
        print("STARTING LANE FOLLOWING")
        print("=" * 70)
        print()
        print("Safety Features:")
        print(f"  • LiDAR stop distance: {cfg.LIDAR_STOP_DISTANCE}m")
        print(f"  • Max frames without lane: {cfg.NO_LANE_STOP_FRAMES}")
        print(f"  • Max steering angle: ±{cfg.MAX_STEERING_ANGLE:.2f} rad")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Read camera frame from CSI (index 2 = front camera)
                self.camera.readAll()
                frame = self.camera.csi[2].imageData
                
                # Check if frame is valid
                if frame is None or frame.size == 0:
                    continue
                
                # Read QCar sensors (LiDAR)
                qcar_data = self.qcar.read()
                lidar_data = qcar_data if isinstance(qcar_data, np.ndarray) else None
                
                # Check for obstacles
                self.obstacle_detected = self.check_lidar_obstacles(lidar_data)
                
                # Process frame for lane detection
                lane_center_x, deviation, confidence, debug_frame = self.detector.process_frame(frame)
                
                # Calculate control commands
                if self.obstacle_detected:
                    # SAFETY: Stop if obstacle detected
                    throttle = 0.0
                    steering = 0.0
                    status = "OBSTACLE DETECTED - STOPPED"
                    status_color = (0, 0, 255)  # Red
                else:
                    # Normal lane following
                    throttle, steering = self.calculate_control(deviation, confidence)
                    
                    if confidence > 0.5:
                        status = "LANE FOLLOWING"
                        status_color = (0, 255, 0)  # Green
                    elif confidence > 0.0:
                        status = "LANE DETECTION WEAK"
                        status_color = (0, 255, 255)  # Yellow
                    else:
                        status = "NO LANE - STOPPING"
                        status_color = (0, 0, 255)  # Red
                
                # Send commands to QCar
                self.qcar.write([throttle, steering])
                
                # Add status overlay to debug frame
                cv2.putText(debug_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(debug_frame, f"Throttle: {throttle:.3f}", (10, debug_frame.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(debug_frame, f"Steering: {steering:+.3f} rad", (10, debug_frame.shape[0] - 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if self.obstacle_detected:
                    cv2.putText(debug_frame, "OBSTACLE!", (10, debug_frame.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Calculate FPS
                self.fps_frame_count += 1
                if self.fps_frame_count >= 10:
                    fps_end_time = time.time()
                    self.current_fps = self.fps_frame_count / (fps_end_time - self.fps_start_time)
                    self.fps_start_time = fps_end_time
                    self.fps_frame_count = 0
                
                cv2.putText(debug_frame, f"FPS: {self.current_fps:.1f}", (10, debug_frame.shape[0] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Lane Follower", debug_frame)
                
                # Handle keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested quit...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"lane_follower_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, debug_frame)
                    print(f"Saved frame: {filepath}")
                
                # Track loop time
                loop_time = time.time() - loop_start
                self.loop_times.append(loop_time)
                
                # Periodic status update
                frame_count += 1
                if frame_count % 100 == 0:
                    stats = self.detector.get_stats()
                    avg_loop_time = np.mean(self.loop_times[-100:]) if self.loop_times else 0
                    print(f"Frame {frame_count:5d} | "
                          f"Detection: {stats['detection_rate']:5.1f}% | "
                          f"FPS: {self.current_fps:5.1f} | "
                          f"Loop: {avg_loop_time*1000:5.1f}ms")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)...")
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stop()
    
    def stop(self):
        """Clean shutdown of all systems."""
        print()
        print("=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        print()
        
        # Stop the car
        print("Stopping QCar...")
        try:
            self.qcar.write([0.0, 0.0])
            time.sleep(0.1)
        except:
            pass
        
        # Print final statistics
        stats = self.detector.get_stats()
        print()
        print("Final Statistics:")
        print(f"  Frames Processed: {stats['frames_processed']}")
        print(f"  Lanes Detected: {stats['lanes_detected']}")
        print(f"  Detection Rate: {stats['detection_rate']:.1f}%")
        
        if self.loop_times:
            avg_loop = np.mean(self.loop_times) * 1000
            max_loop = np.max(self.loop_times) * 1000
            print(f"  Avg Loop Time: {avg_loop:.1f}ms")
            print(f"  Max Loop Time: {max_loop:.1f}ms")
        
        print()
        
        # Cleanup resources
        print("Releasing camera...")
        try:
            self.camera.terminate()
        except:
            pass
        
        print("Terminating QCar...")
        self.qcar.terminate()
        
        print("Closing windows...")
        cv2.destroyAllWindows()
        
        print()
        print("Shutdown complete. Goodbye!")
        print("=" * 70)


def main():
    """Entry point."""
    # Check if running on physical QCar
    if not IS_PHYSICAL_QCAR:
        print("ERROR: This script must run on a physical QCar (Jetson Nano)!")
        print("Cannot run in simulation mode.")
        return
    
    # Create and run lane follower
    follower = LaneFollower()
    follower.run()


if __name__ == "__main__":
    main()

