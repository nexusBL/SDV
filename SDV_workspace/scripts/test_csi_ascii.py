#!/usr/bin/env python3
"""
SDV Live CSI Camera Matrix — ASCII Edition
Renders all 4 CSI cameras in real-time ASCII characters directly in the terminal!
No GUI or X11 forwarding required.
"""

import time
import sys
import os
import cv2
import numpy as np

# Force headless NVMM capture:
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.products.qcar import QCarCameras

# Brightness to ASCII character mapping
ASCII_CHARS = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

def frame_to_ascii(frame, cols=60, rows=20):
    """Converts an OpenCV BGR frame into a multiline ASCII string."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to fit terminal dimensions
    resized = cv2.resize(gray, (cols, rows))
    
    # Map pixel intensities (0-255) to ASCII string indices (0-9)
    # We use vectorization for high speed
    ascii_indices = (resized / 255.0 * (len(ASCII_CHARS) - 1)).astype(np.int32)
    
    # Build lines
    ascii_lines = []
    for r in range(rows):
        line = "".join([ASCII_CHARS[idx] for idx in ascii_indices[r, :]])
        ascii_lines.append(line)
        
    return ascii_lines

def main():
    print("🎥 Initializing QCar CSI Matrix — ASCII Edition...")
    
    # We MUST use the native hardware pipeline resolution for Quanser QCar2 NVMM (820x616 at 80fps)
    try:
        cameras = QCarCameras(
            frameWidth=820, frameHeight=616, frameRate=80,
            enableBack=True, enableFront=True, enableLeft=True, enableRight=True
        )
    except Exception as e:
        print(f"Failed to initialize cameras: {e}")
        return

    # Wait for NVMM GStreamer buffers to lock
    print("Locking memory hooks... (Wait 3 seconds)")
    time.sleep(3.0)
    
    # Figure out terminal size
    try:
        tw, th = os.get_terminal_size()
    except:
        tw, th = 120, 40
        
    # We want a 2x2 grid. So each frame gets half the terminal width and half the height.
    # Terminal characters are roughly 2x as tall as they are wide, so we compensate.
    cam_cols = max(20, (tw // 2) - 4)
    cam_rows = max(10, (th // 2) - 3)

    print("\033[2J") # Clear screen
    
    try:
        while True:
            cameras.readAll()
            
            # Quanser index: Right=0, Rear=1, Front=2, Left=3
            frames = []
            for i in range(4):
                cam = cameras.csi[i]
                if cam is not None and getattr(cam, 'imageData', None) is not None and cam.imageData.size > 0:
                    frames.append(cam.imageData)
                else:
                    frames.append(np.zeros((308, 410, 3), dtype=np.uint8))
            
            # Convert Frames to ASCII Line Arrays
            asc_right = frame_to_ascii(frames[0], cam_cols, cam_rows)
            asc_rear  = frame_to_ascii(frames[1], cam_cols, cam_rows)
            asc_front = frame_to_ascii(frames[2], cam_cols, cam_rows)
            asc_left  = frame_to_ascii(frames[3], cam_cols, cam_rows)
            
            # Move cursor to top-left of terminal instead of clearing (prevents flickering)
            sys.stdout.write("\033[H")
            
            sys.stdout.write("╔" + "═"*(cam_cols+2) + "╦" + "═"*(cam_cols+2) + "╗\n")
            sys.stdout.write(f"║ {'LEFT CSI':^{cam_cols}} ║ {'FRONT CSI':^{cam_cols}} ║\n")
            sys.stdout.write("╠" + "═"*(cam_cols+2) + "╬" + "═"*(cam_cols+2) + "╣\n")
            
            # Top row lines
            for i in range(cam_rows):
                sys.stdout.write(f"║ {asc_left[i]} ║ {asc_front[i]} ║\n")
                
            sys.stdout.write("╠" + "═"*(cam_cols+2) + "╬" + "═"*(cam_cols+2) + "╣\n")
            sys.stdout.write(f"║ {'REAR CSI':^{cam_cols}} ║ {'RIGHT CSI':^{cam_cols}} ║\n")
            sys.stdout.write("╠" + "═"*(cam_cols+2) + "╬" + "═"*(cam_cols+2) + "╣\n")
            
            # Bottom row lines
            for i in range(cam_rows):
                sys.stdout.write(f"║ {asc_rear[i]} ║ {asc_right[i]} ║\n")
                
            sys.stdout.write("╚" + "═"*(cam_cols+2) + "╩" + "═"*(cam_cols+2) + "╝\n")
            sys.stdout.write("Press CTRL+C to quit.\n")
            sys.stdout.flush()
            
            time.sleep(0.05) # ~20 FPS limit so SSH terminal doesn't lag out
            
    except KeyboardInterrupt:
        sys.stdout.write("\033[2J\033[H")
        print("\n🛑 Exited ASCII Camera Matrix.")
    finally:
        cameras.terminate()

if __name__ == '__main__':
    main()
