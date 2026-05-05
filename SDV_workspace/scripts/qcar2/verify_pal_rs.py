from pal.products.qcar import QCarRealSense
import cv2
import time
import numpy as np

try:
    # Initialize with IR and Depth
    # Note: frameWidthIR/HeightIR defaults to 1280x720, frameRateIR defaults to 15
    print("Initializing QCarRealSense...")
    car = QCarRealSense(mode='DEPTH, IR')
    
    print("Capturing IR and Depth...")
    for _ in range(30):
        car.read_IR(lens='L') # Read Left IR
        car.read_depth(dataMode='M') # Read Depth in meters
        
        ir = car.imageBufferIR
        depth = car.imageBufferDepth
        
        if ir is not None and ir.max() > 0:
            print(f"✓ Success! IR shape: {ir.shape}")
            cv2.imwrite("verif_pal_ir.png", ir)
            break
        else:
            print("Waiting for IR data...")
        time.sleep(0.5)
    else:
        print("FAIL: No IR data received via PAL API.")

    if depth is not None:
        print(f"✓ Success! Depth shape: {depth.shape}")
        # Normalize depth for visibility
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("verif_pal_depth.png", d_norm)

finally:
    try:
        car.terminate()
    except:
        pass
