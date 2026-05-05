#!/usr/bin/env python3
"""
Test all 4 CSI cameras with dummy HDMI connected.
Right=0, Rear=1, Front=2, Left=3
"""
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

print("╔══════════════════════════════════════════════════════╗")
print("║   QCar2 CSI Camera Test — All 4 Cameras            ║")
print("╚══════════════════════════════════════════════════════╝")
print()

cameras = None
print("[CSI] Initializing all 4 cameras...")
try:
    cameras = QCarCameras(
        frameWidth=820,
        frameHeight=616,
        frameRate=80,
        enableRight=True,
        enableBack=True,
        enableFront=True,
        enableLeft=True
    )
    print("[CSI] Cameras initialized ✅")
    print()

    names = {0: 'Right', 1: 'Rear', 2: 'Front', 3: 'Left'}

    for attempt in range(30):
        cameras.readAll()
        time.sleep(0.2)

        all_ok = True
        for i, name in names.items():
            img = cameras.csi[i].imageData
            max_val = img.max() if img is not None else 0
            status = "✅ WORKING" if max_val > 10 else "❌ BLANK"
            if max_val <= 10:
                all_ok = False
            print(f"  Camera {i} ({name:5s}): {status}  max_pixel={max_val}")

        print()

        if all_ok:
            print("🔥 ALL 4 CSI CAMERAS WORKING!")
            for i, name in names.items():
                img = cameras.csi[i].imageData
                if img is not None and img.max() > 10:
                    path = f"/home/nvidia/Desktop/SDV_workspace/scripts/csi_{name.lower()}.jpg"
                    cv2.imwrite(path, img)
                    print(f"  Saved: csi_{name.lower()}.jpg")
            break

        if attempt < 29:
            print(f"Retrying... ({attempt+1}/30)")
            time.sleep(0.5)
    else:
        print("❌ Some cameras still blank after 30 attempts")

except KeyboardInterrupt:
    print("\n[CSI] Interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
finally:
    if cameras is not None:
        try:
            cameras.terminate()
        except Exception:
            pass
        print("[CSI] Terminated safely.")
