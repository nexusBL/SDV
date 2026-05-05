#!/usr/bin/env python3
"""
Standalone Control Hardware Test
Simple script to physically test the hardware wrapper without ROS2 or the camera.
"""
import time
import numpy as np
import sys
import os

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/sdv_control')))
from sdv_control.hardware_controller import HardwareController

def main():
    print("🚗 SDV Standalone Control Hardware Test")
    print("Testing Control Abstraction...")
    
    hw = HardwareController(sample_rate=100)
    hw.start()

    if not hw.is_active:
        print("Hardware MOCK test passed. (Quanser PAL API not detected as physical car)")
        return

    try:
        print("⚠️ Hardware ACTIVE. Keep car on blocks. Spinning motors in 3 seconds...")
        time.sleep(3)
        
        print("➡️ Forward...")
        t0 = time.time()
        while time.time() - t0 < 1.0:
            hw.write(0.1, 0.0, np.zeros(8))
            time.sleep(0.01)

        print("🛑 Sudden Brake...")
        t0 = time.time()
        while time.time() - t0 < 0.5:
            # Active brake + reverse/brake lights
            leds = np.zeros(8)
            leds[4] = 1; leds[5] = 1 
            hw.write(-1.0, 0.0, leds)
            time.sleep(0.01)

        print("↩️ Turn wheels left...")
        t0 = time.time()
        while time.time() - t0 < 1.0:
            leds = np.zeros(8)
            leds[1] = 1; leds[3] = 1
            hw.write(0.0, -0.3, leds)
            time.sleep(0.01)

        hw.write(0.0, 0.0, np.zeros(8))
        print("✅ Control test Done.")

    finally:
        hw.stop()

if __name__ == '__main__':
    main()
