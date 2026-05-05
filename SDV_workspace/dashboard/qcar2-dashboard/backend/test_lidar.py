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
    from pal.products.qcar import QCarLidar
    print("Initializing QCarLidar...")
    lidar = QCarLidar()
    print("QCarLidar initialized. Waiting 2s for spin up...")
    time.sleep(2)
    
    for i in range(10):
        lidar.read()
        print(f"[{i}] LiDAR distances length: {len(lidar.distances)}")
        if len(lidar.distances) > 0:
            print(f"[{i}] First 5 distances: {lidar.distances[:5]}")
        time.sleep(0.5)
    
    print("Terminating...")
    lidar.terminate()
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
