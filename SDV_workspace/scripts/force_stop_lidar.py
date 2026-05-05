#!/usr/bin/env python3
"""
Force stop QCar2 LiDAR — kills all processes first, then stops hardware.
"""

import os
import time
import subprocess

print("=" * 50)
print("QCar2 LiDAR Force Stop")
print("=" * 50)

# ══════════════════════════════════════════════════════════════════
#  STEP 1: Kill all processes that might be using LiDAR
# ══════════════════════════════════════════════════════════════════
print("\n[Step 1] Killing processes holding LiDAR GPIO...")

kill_commands = [
    "pkill -9 -f 'qcar2_nodes'",
    "pkill -9 -f 'lidar'",
    "pkill -9 -f 'ros2'",
    "pkill -9 -f 'QCarLidar'",
    "pkill -9 -f 'sdv_autonomous'",
    "pkill -9 -f 'sensor_fusion'",
]

for cmd in kill_commands:
    result = subprocess.run(cmd, shell=True, capture_output=True)
    print(f"  {cmd}")

# Wait for processes to fully die
time.sleep(2)

# ══════════════════════════════════════════════════════════════════
#  STEP 2: Unexport GPIO pins (release sysfs lock)
# ══════════════════════════════════════════════════════════════════
print("\n[Step 2] Releasing GPIO pins...")

# Common GPIO pins used by QCar LiDAR (try multiple)
gpio_pins = [216, 217, 50, 51, 52, 53, 79, 80]

for pin in gpio_pins:
    try:
        # Check if pin is exported
        if os.path.exists(f"/sys/class/gpio/gpio{pin}"):
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(pin))
            print(f"  Unexported GPIO {pin}")
    except Exception as e:
        pass  # Pin might not be exported, that's fine

time.sleep(1)

# ══════════════════════════════════════════════════════════════════
#  STEP 3: Now try to initialize and stop LiDAR properly
# ══════════════════════════════════════════════════════════════════
print("\n[Step 3] Stopping LiDAR hardware...")

try:
    from pal.products.qcar import QCarLidar
    
    myLidar = QCarLidar(
        numMeasurements=720,
        rangingDistanceMode=2,
        interpolationMode=0
    )
    time.sleep(0.5)
    myLidar.terminate()
    print("  LiDAR terminated via PAL ✅")
    
except Exception as e:
    print(f"  PAL termination failed: {e}")
    print("  Trying alternative method...")

# ══════════════════════════════════════════════════════════════════
#  STEP 4: Kill any remaining Python processes (last resort)
# ══════════════════════════════════════════════════════════════════
print("\n[Step 4] Final cleanup...")

# Only kill OTHER python processes, not this script
my_pid = os.getpid()
result = subprocess.run(
    f"pgrep -f python3 | grep -v {my_pid} | xargs -r kill -9",
    shell=True, capture_output=True
)

# Also try stopping via ROS2 if running
subprocess.run("ros2 daemon stop", shell=True, capture_output=True)

print("\n" + "=" * 50)
print("LiDAR stop sequence complete.")
print("If motor still spinning, try: sudo reboot")
print("=" * 50)
