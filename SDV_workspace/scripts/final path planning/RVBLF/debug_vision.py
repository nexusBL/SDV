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
    # Initialize Camera
    my_cams = QCarCameras(
        frameWidth=640,
        frameHeight=480,
        frameRate=30,
        enableRight=False,
        enableLeft=False,
        enableFront=True,
        enableBack=False
    )
    
    print("Capturing frame in 2 seconds...")
    time.sleep(2.0)
    
    my_cams.readAll()
    frame = my_cams.csiFront.imageData
    
    if frame is not None:
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/debug_frame.jpg", frame)
        print("Frame saved to RVBLF/debug_frame.jpg")
    else:
        print("Failed to capture frame.")
        
    my_cams.terminate()

if __name__ == "__main__":
    main()
