#!/usr/bin/env python3
"""
SDV Live CSI Camera Grid
Displays all 4 CSI cameras simultaneously in a 2x2 grid using OpenCV.
"""

import time
import sys
import os
import cv2
import numpy as np

# Force headless NVMM capture:
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

# Ensure Quanser API is accessible
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.products.qcar import QCarCameras

def main():
    print("🎥 Initializing all 4 QCar CSI Cameras...")
    print("Please wait a few seconds for the NVMM hardware buffers to lock...")
    
    # We use half-resolution (410x308) to make rendering 4 streams much faster and fit on a standard monitor
    cameras = QCarCameras(
        frameWidth=410, frameHeight=308, frameRate=30,
        enableBack=True, enableFront=True, enableLeft=True, enableRight=True
    )
    
    print("\n✅ Cameras Online!")
    print("Press 'q' or 'ESC' in the Video Window to exit.")
    print("Note: Run this directly on the car's monitor or via VNC to see the pop-up window.")
    
    try:
        while True:
            # Tell hardware to pull new frames into memory
            cameras.readAll()
            
            # Quanser CSI Index mapping: Right=0, Rear=1, Front=2, Left=3
            frames = []
            for i in range(4):
                cam = cameras.csi[i]
                if cam is not None and getattr(cam, 'imageData', None) is not None and cam.imageData.size > 0:
                    frames.append(cam.imageData)
                else:
                    # Black placeholder if camera drops a frame
                    frames.append(np.zeros((308, 410, 3), dtype=np.uint8))
            
            # Create a 2x2 Grid Visualization
            # Top row: Left Camera (3), Front Camera (2)
            top_row = np.hstack((frames[3], frames[2]))
            
            # Bottom row: Rear Camera (1), Right Camera (0)
            bottom_row = np.hstack((frames[1], frames[0]))
            
            # Stack Top and Bottom rows vertically
            grid = np.vstack((top_row, bottom_row))
            
            # Display
            cv2.imshow("QCar CSI Cameras - [Top: Left, Front] [Bottom: Rear, Right]", grid)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("🛑 User requested exit.")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ CTRL+C Detected. Shutting down safely...")
    finally:
        cameras.terminate()
        cv2.destroyAllWindows()
        print("🧹 Hardware interfaces released. Goodbye!")

if __name__ == '__main__':
    main()
