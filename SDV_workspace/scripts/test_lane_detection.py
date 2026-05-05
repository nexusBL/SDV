#!/usr/bin/env python3
"""
test_lane_detection.py — Standalone Lane Detection Tester
===========================================================
Test lane detection using CSI camera WITHOUT moving the car.
Press 'q' to quit, 's' to save current frame.

Usage:
    python3 test_lane_detection.py
"""

import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime

# Fix nvarguscamerasrc EGL authorization issues
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

# Add QCar libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.products.qcar import QCarCameras

# Import lane detector and config
from lane_detector_simple import SimpleLaneDetector
import config_lane as cfg


def main():
    """Main test loop."""
    print("=" * 70)
    print("LANE DETECTION TESTER")
    print("=" * 70)
    print()
    print("This script tests lane detection WITHOUT moving the car.")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics")
    print()
    print("Visualizations:")
    print("  Green trapezoid = Region of Interest (ROI)")
    print("  Red lines = Detected lane lines")
    print("  Cyan line = Calculated lane center")
    print("  Blue line = Image center (where car wants to be)")
    print()
    print("=" * 70)
    print()
    
    # Create output directory for saved frames
    output_dir = "/home/nvidia/Desktop/SDV_workspace/scripts/lane_test_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saved frames will be in: {output_dir}")
    print()
    
    # Initialize CSI camera using QCarCameras API
    print("Initializing CSI Front Camera (index=2)...")
    try:
        cameras = QCarCameras(
            frameWidth=cfg.CAMERA_WIDTH, 
            frameHeight=cfg.CAMERA_HEIGHT, 
            frameRate=30,
            enableFront=True,   # Front camera only
            enableLeft=False,
            enableRight=False,
            enableBack=False
        )
        print("Camera initialized successfully!")
        print(f"Resolution: {cfg.CAMERA_WIDTH}x{cfg.CAMERA_HEIGHT}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize CSI camera: {e}")
        print("Make sure:")
        print("  1. Camera ribbon cable is properly connected")
        print("  2. You're running on QCar2 Jetson system")
        print("  3. No other processes are using the camera")
        return
    
    # Camera warm-up: Wait for real frames (CSI cameras need time to start)
    print("Warming up camera (waiting for valid frames)...")
    frame_ready = False
    for attempt in range(30):
        cameras.readAll()
        frame = cameras.csi[2].imageData
        
        if frame is not None and frame.size > 0 and frame.max() > 10:
            frame_ready = True
            print(f"Camera ready! (attempt {attempt + 1})")
            break
        
        time.sleep(0.2)
    
    if not frame_ready:
        print("ERROR: Camera failed to produce valid frames after 30 attempts")
        print("Camera may not be connected or functioning properly")
        cameras.terminate()
        return
    
    print()
    
    # Initialize lane detector
    print("Initializing lane detector...")
    detector = SimpleLaneDetector()
    print("Lane detector ready!")
    print()
    
    # Performance tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    print("Starting lane detection... (Press 'q' to quit)")
    print()
    
    try:
        while True:
            # Read frame from CSI camera (index 2 = front camera)
            cameras.readAll()
            frame = cameras.csi[2].imageData
            
            # Check if frame is valid (skip black/empty frames)
            if frame is None or frame.size == 0 or frame.max() <= 10:
                continue
            
            # Process frame through lane detector
            lane_center_x, deviation, confidence, debug_frame = detector.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # Add FPS and additional info to debug frame
            cv2.putText(debug_frame, f"FPS: {current_fps:.1f}", (10, debug_frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if lane_center_x is not None:
                cv2.putText(debug_frame, f"Lane Center: {lane_center_x}px", 
                           (10, debug_frame.shape[0] - 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_frame, f"Deviation: {deviation:+.3f}", 
                           (10, debug_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Calculate steering that would be applied
                if abs(deviation) < cfg.STEERING_DEADZONE:
                    steer = 0.0
                else:
                    steer = deviation * cfg.STEERING_GAIN
                    steer = np.clip(steer, -cfg.MAX_STEERING_ANGLE, cfg.MAX_STEERING_ANGLE)
                
                cv2.putText(debug_frame, f"Steering: {steer:+.3f} rad", 
                           (10, debug_frame.shape[0] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(debug_frame, "NO LANE DETECTED", 
                           (10, debug_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Lane Detection Test", debug_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lane_test_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, debug_frame)
                print(f"Saved frame: {filepath}")
            
            elif key == ord('r'):
                # Reset statistics
                detector.frames_processed = 0
                detector.lanes_detected = 0
                print("Statistics reset!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    
    finally:
        # Print final statistics
        stats = detector.get_stats()
        print()
        print("=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"Lanes Detected: {stats['lanes_detected']}")
        print(f"Detection Rate: {stats['detection_rate']:.1f}%")
        print()
        
        # Cleanup
        try:
            cameras.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()

