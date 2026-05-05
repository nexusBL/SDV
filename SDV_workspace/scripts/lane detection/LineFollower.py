from pal.products.qcar import QCar
import os
import numpy as np 
import cv2
import time

from lines import LaneDetect
from keys import KeyListener
from lidar import LidarProcessor
from camera import CameraProcessor
from control import ControlSystem

# ----------------------------------------------------------------------------
# System Initialization
# ----------------------------------------------------------------------------
print("Booting QCar 2 Autonomous Systems...")

# Hardware Abstraction Layer
myCar = QCar()

# Perception and Control Modules
lanes = LaneDetect()
lidar = LidarProcessor()
camera = CameraProcessor()
control = ControlSystem(kp=0.00225) 
key_listener = KeyListener()
key_listener.start()

# State Machine Definitions
STATE_IDLE = 0
STATE_DRIVING = 1
STATE_OBSTACLE = 2
current_state = STATE_IDLE

# Control Variables
throttle_base = 0.08  # Typical safe cruise throttle for QCar 2
resume_counter = 0
RESUME_TIME_FRAMES = 10  # Gives the LIDAR a moment to confirm path is clear 

# HUD configuration
font = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------------------------------------------------------
# Main Control Loop
# ----------------------------------------------------------------------------
print("Systems Ready. Press 'a' to engage Autopilot. Press 's' to Stop. Press 'q' to Quit.")

try:
    while not key_listener.should_exit:
        # 1. Perception Step (Camera & LIDAR)
        frame = camera.take_frame()
        # Process the visual pipeline
        lines_frame = lanes.find_lines(frame) 
        
        # Poll LIDAR for immediate forwards threats
        obstacle_detected = lidar.detect_object()

        # 2. Input Handling Step
        key = key_listener.last_key_pressed
        if key == 'a' and current_state == STATE_IDLE:
            current_state = STATE_DRIVING
            control.reset()
            print("Autopilot ENGAGED.")
            key_listener.last_key_pressed = None
            
        elif key == 's' and current_state != STATE_IDLE:
            current_state = STATE_IDLE
            print("Autopilot DISENGAGED.")
            key_listener.last_key_pressed = None

        # 3. State Machine Step
        throttle_axis = 0.0
        steering_axis = 0.0
        LEDs = np.zeros(8)
        
        if current_state == STATE_DRIVING:
            if obstacle_detected:
                # Immediate transition to avoidance mode
                current_state = STATE_OBSTACLE
                resume_counter = RESUME_TIME_FRAMES
                print("OBSTACLE DETECTED! Emergency Stop.")
            else:
                # Normal Driving behavior
                # Invert PID output: QCar requires negative steering for left turns!
                steering_axis = -control.control_pid(lanes.error)
                steering_axis = control.saturate(steering_axis, 0.6, -0.6)
                throttle_axis = throttle_base
                LEDs = np.array([0, 0, 0, 0, 1, 1, 0, 0]) # Headlights on

        elif current_state == STATE_OBSTACLE:
            # Active emergency stop condition
            LEDs = np.array([1, 0, 1, 0, 1, 0, 1, 0]) # Hazard lights flash
            
            if not obstacle_detected:
                resume_counter -= 1
                if resume_counter <= 0:
                    current_state = STATE_DRIVING
                    control.reset()
                    print("Path Cleared. Resuming Autopilot.")
            else:
                resume_counter = RESUME_TIME_FRAMES # Keep resetting internal timer

        # IDLE state just sets throttle and steering to 0 (already 0.0 default)

        # 4. Actuation Step
        myCar.read_write_std(throttle_axis, steering_axis, LEDs)

        # 5. Diagnostic / HUD Step
        status_text = {
            STATE_IDLE: ("IDLE (Press A)", (200, 200, 200)),
            STATE_DRIVING: ("AUTOPILOT ACTIVE", (0, 255, 0)),
            STATE_OBSTACLE: ("OBSTACLE STOP", (0, 0, 255))
        }
        
        txt, color = status_text[current_state]
        cv2.putText(lines_frame, txt, (10, 130), font, 0.7, color, 2)
        cv2.putText(lines_frame, f"Speed: {throttle_axis:.2f}", (10, 160), font, 0.6, (255, 255, 255), 1)
        cv2.putText(lines_frame, f"Steer: {steering_axis:.2f}", (10, 190), font, 0.6, (255, 255, 255), 1)

        # 6. Display / Logging Step
        try:
            # Check if we have an active display to prevent X11 core dumps over SSH
            if os.environ.get("DISPLAY"):
                cv2.imshow('QCar 2 Autonomous Tracking', lines_frame)
                cv2.waitKey(1)
            else:
                # When running headless over SSH, save the visualizer output to disk occasionally
                if int(time.time() * 10) % 5 == 0:
                    cv2.imwrite('lane_headless_debug.jpg', lines_frame)
        except Exception as e:
            print(f"Display Warning: {e}")
except KeyboardInterrupt:
    print("\nKeyboard Interrupt caught! Shutting down.")

finally:
    # Critical Safety Teardown
    print("Initiating Hardware Teardown...")
    try:
        # Guarantee motors are neutral
        myCar.read_write_std(0.0, 0.0, np.zeros(8))
    except:
        pass
    
    myCar.terminate()
    lidar.end_lidar()
    camera.end_camera()
    cv2.destroyAllWindows()
    print("All Systems Terminated Securely. Goodbye.")
