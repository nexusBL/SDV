import os
import sys
import time

# Quanser path setup
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
QUANSER_LIB_PATH = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.exists(QUANSER_LIB_PATH) and QUANSER_LIB_PATH not in sys.path:
    sys.path.insert(0, QUANSER_LIB_PATH)

try:
    print("Initializing QCar...")
    from pal.products.qcar import QCar, QCarCameras, QCarRealSense
    qcar = QCar()
    print("QCar initialized.")
    
    print("Initializing Cameras...")
    cameras = QCarCameras(
        frameWidth=640, frameHeight=480, frameRate=30,
        enableFront=True, enableLeft=True, enableRight=True, enableBack=True
    )
    print("Cameras initialized.")
    
    print("Reading All Cameras once...")
    cameras.readAll()
    print("Read successful.")
    
    print("Terminating...")
    cameras.terminate()
    qcar.terminate()
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
