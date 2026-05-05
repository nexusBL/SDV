import sys
import os
import time

# Quanser path setup
QUANSER_LIB_PATH = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.exists(QUANSER_LIB_PATH) and QUANSER_LIB_PATH not in sys.path:
    sys.path.insert(0, QUANSER_LIB_PATH)

from pal.products.qcar import QCarLidar

print("Attempting to force stop the QCar2 LiDAR hardware...")

try:
    # Initialize briefly to gain control
    myLidar = QCarLidar(
        numMeasurements=720,
        rangingDistanceMode=2,
        interpolationMode=0
    )
    
    # Give a small delay for initialization
    time.sleep(0.5)
    
    # Properly terminate to stop the motor
    myLidar.terminate()
    print("LiDAR motor should now be STOPPED.")

except Exception as e:
    print(f"Error during force stop: {e}")
    # If standard terminate fails, try brute force with system processes
    import os
    os.system("pkill -9 -f lidar_radar.py")
    os.system("pkill -9 -f python3")
    print("Brute force pkill triggered as fallback.")
