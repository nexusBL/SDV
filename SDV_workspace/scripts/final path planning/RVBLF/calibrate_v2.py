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

from rvblf_vision import RVBLFVisionV2

def main():
    W, H = 820, 410
    vc = RVBLFVisionV2(width=W, height=H)
    
    my_cams = QCarCameras(
        frameWidth=W,
        frameHeight=H,
        frameRate=30,
        enableFront=True
    )
    
    print("Capturing BEV frame in 2 seconds...")
    time.sleep(2.0)
    
    my_cams.readAll()
    frame = my_cams.csiFront.imageData
    
    if frame is not None:
        # Save Original
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/cal_raw.jpg", frame)
        
        # Save BEV
        bev = cv2.warpPerspective(frame, vc.M, (W, H))
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/cal_bev.jpg", bev)
        
        # Save Edge Mask (for debugging detection)
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 25, 4)
        cv2.imwrite("/home/nvidia/Desktop/SDV_workspace/scripts/final path planning/RVBLF/cal_edges.jpg", adapt)
        
        print("Calibration frames saved to RVBLF/cal_*.jpg")
    else:
        print("Failed to capture frame.")
        
    my_cams.terminate()

if __name__ == "__main__":
    main()
