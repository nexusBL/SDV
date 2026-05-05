import os, sys, cv2, time, numpy as np

os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from pal.products.qcar import QCarCameras

print("Initializing front camera (820x616 @ 30fps)...")
cams = QCarCameras(enableFront=True, frameWidth=820, frameHeight=616, frameRate=30)

print("Warming up (60 frames)...")
for _ in range(60):
    cams.readAll()
    time.sleep(0.033)

# Try both access methods
frame_a = cams.csi[2].imageData
frame_b = cams.csiFront.imageData if cams.csiFront else None

for label, frame in [("csi[2]", frame_a), ("csiFront", frame_b)]:
    if frame is not None:
        print(f"\n[{label}] Shape: {frame.shape}, max pixel: {frame.max()}")
        roi = frame[int(frame.shape[0]*0.6):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 50, 100]), np.array([45, 255, 255]))
        yellow_px = cv2.countNonZero(mask)
        print(f"[{label}] Yellow pixels in bottom 40%: {yellow_px}")
        cv2.imwrite(f"debug_{label.replace('[','').replace(']','')}.png", frame)
        cv2.imwrite(f"debug_{label.replace('[','').replace(']','')}_roi.png", roi)
        print(f"[{label}] Saved debug image.")
    else:
        print(f"\n[{label}] NONE - camera slot is empty!")

cams.terminate()
print("\nDone. Check the debug_*.png images.")
