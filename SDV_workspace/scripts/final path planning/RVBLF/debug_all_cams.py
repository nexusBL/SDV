#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np

# Resolve headless GPU authorization
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

# Add QUANSER libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

try:
    from pal.products.qcar import QCarCameras
except ImportError:
    print("FATAL: QCar library not loaded.")
    sys.exit(1)

def main():
    # Initialize Cameras
    my_cams = QCarCameras(
        frameWidth=640,
        frameHeight=480,
        frameRate=30,
        enableRight=True,
        enableLeft=True,
        enableFront=True,
        enableBack=False
    )
    
    print("Capturing frames in 2 seconds...")
    time.sleep(2.0)
    
    my_cams.readAll()
    
    if my_cams.csiFront.imageData is not None:
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/debug_front.jpg", my_cams.csiFront.imageData)
    if my_cams.csiLeft.imageData is not None:
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/debug_left.jpg", my_cams.csiLeft.imageData)
    if my_cams.csiRight.imageData is not None:
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/debug_right.jpg", my_cams.csiRight.imageData)
        
    print("Frames saved.")
    my_cams.terminate()

if __name__ == "__main__":
    main()
