#!/usr/bin/env python3
"""
Separate script to capture a frame from the Front CSI camera.
Saved to /home/nvidia/Desktop/SDV_workspace/scripts/captured_csi_front.jpg
"""
import sys
import os
import time
import cv2

# Fix nvarguscamerasrc EGL authorization issues by forcing headless NVMM capture:
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.products.qcar import QCarCameras

def capture():
    print("[CSI] Initializing camera...")
    cameras = None
    try:
        cameras = QCarCameras(
            frameWidth=820,
            frameHeight=410,
            frameRate=30,
            enableFront=True
        )
        print("[CSI] Warming up...")
        time.sleep(2.0)
        
        for i in range(10):
            cameras.readAll()
            img = cameras.csiFront.imageData
            if img is not None and img.max() > 10:
                save_path = "/home/nvidia/Desktop/SDV_workspace/scripts/captured_csi_front.jpg"
                cv2.imwrite(save_path, img)
                print(f"[SUCCESS] Captured and saved to {save_path}")
                return
            time.sleep(0.1)
        print("[ERROR] Could not capture a valid frame (check lighting/connections)")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if cameras:
            cameras.terminate()
            print("[CSI] Terminated safely.")

if __name__ == "__main__":
    capture()
