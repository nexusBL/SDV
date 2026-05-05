import sys
import os

# Fix nvarguscamerasrc EGL authorization issues by forcing headless NVMM capture:
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
import numpy as np
import cv2
import time

from pal.products.qcar import QCarCameras

# Template for initializing cameras 
def setup_cameras():
    print("Initializing QCar Cameras in headless mode...")
    # Initialize the camera setup (Modify parameters as per your specific physical setup!)
    camParams = {
        'frameWidth': 640,
        'frameHeight': 480,
        'frameRate': 30.0,
        'enableRight': True,
        'enableBack': False,
        'enableLeft': True,
        'enableFront': True
    }
    
    # Standard PAL wrapper for CSI cameras on Jetson physically
    cameras = QCarCameras(
        frameWidth=camParams['frameWidth'],
        frameHeight=camParams['frameHeight'],
        frameRate=camParams['frameRate'],
        enableRight=camParams['enableRight'],
        enableBack=camParams['enableBack'],
        enableLeft=camParams['enableLeft'],
        enableFront=camParams['enableFront']
    )
    
    return cameras

if __name__ == "__main__":
    my_cams = setup_cameras()
    time.sleep(1.0)
    print("Cameras ready. Attempting to capture frames.")
    
    try:
        while True:
            # Poll cameras
            my_cams.readAll()
            
            # Example access to image front buffer
            if my_cams.csiFront.imageData is not None:
                front_frame = my_cams.csiFront.imageData
                # Perform path planning / vision analytics here
                
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        my_cams.terminate()
