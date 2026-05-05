import cv2
import time
import numpy as np
import os

from hardware.depth_monitor import DepthMonitor
from config import AppConfig

# Headless verification of DepthMonitor IR stream
cfg = AppConfig()
dm = DepthMonitor(cfg)

try:
    dm.initialize()
    if dm._mock_mode:
        print("FAIL: DepthMonitor in MOCK mode. Check RealSense connection.")
        exit(1)
        
    print("Capturing IR frame...")
    # Wait for a few frames to stabilize
    for i in range(10):
        reading = dm.get_obstacle()
        if reading.ir_frame is not None:
            print(f"✓ IR Frame captured ({reading.ir_frame.shape})")
            cv2.imwrite("verif_raw_ir.png", reading.ir_frame)
            
            # Save a colorized version for better visibility
            colored = cv2.applyColorMap(reading.ir_frame, cv2.COLORMAP_JET)
            cv2.imwrite("verif_colored_ir.png", colored)
            break
        time.sleep(0.5)
    else:
        print("FAIL: No IR frame received after 10 attempts.")

finally:
    dm.terminate()
