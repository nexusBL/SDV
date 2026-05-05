import os, sys

# Fixes "No EGL Display / nvbufsurftransform: Could not get EGL display"
# error on physical QCar2 (Jetson) when running headless (no monitor).
os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)

# Add Quanser library path (matches your QCar2 setup)
_QUANSER_LIB = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.isdir(_QUANSER_LIB) and _QUANSER_LIB not in sys.path:
    sys.path.insert(0, _QUANSER_LIB)

import cv2
import time
import numpy as np
from ultralytics import RTDETR
from pal.products.qcar import QCar, QCarRealSense

# Mapping from notebook classes
CLASS_NAMES = [
    'Stop Sign', 'Turn Left Ahead', 'Turn Right Ahead',
    'Straight Ahead', 'Give Way', 'No Entry',
    'Keep Left', 'Keep Right', 'Round About',
    'Pedestrian Crossing', 'Traffic Signal',
    'U Turn', 'No U Turn'
]

def main():
    # 1. Initialize the QCar Hardware and Camera
    try:
        car = QCar(readMode=1, frequency=30)
        cam = QCarRealSense(mode='RGB')
        print("✅ QCar and RealSense initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize QCar/RealSense: {e}")
        return

    # 2. Load the trained RT-DETR model 
    model_path = '/home/nvidia/Desktop/SDV_workspace/scripts/BEST_MODEL_QCAR2.pt'  
    try:
        model = RTDETR(model_path)
        print(f"✅ Loaded RT-DETR model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        car.terminate()
        cam.terminate()
        return

    print("🚦 Starting real-time inference. Press 'q' to quit.")
    
    try:
        while True:
            # 3. Read sensors
            car.read()
            cam.read_RGB()
            frame = cam.imageBufferRGB
            
            if frame is None or frame.size == 0:
                print("Failed to grab frame. Retrying...")
                time.sleep(0.01)
                continue
                
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 4. Perform inference with the loaded model
            results = model(frame_bgr, conf=0.5, verbose=False)
            
            # Default control values
            throttle = 0.1  # Base forward speed
            steering = 0.0  # Center steering
            
            # 5. Process detections and apply control logic
            if len(results[0].boxes) > 0:
                # Get the highest-confidence detection
                best_box = results[0].boxes[0]
                class_id = int(best_box.cls[0].item())
                confidence = best_box.conf[0].item()
                class_name = CLASS_NAMES[class_id]
                
                print(f"Detected: {class_name} ({confidence:.2f})")
                
                # Command mapping based on traffic signs
                if class_id == 0:  # Stop Sign
                    throttle = 0.0
                    steering = 0.0
                    print("-> Action: STOPPING")
                    
                elif class_id in [1, 6]:  # Turn Left Ahead / Keep Left
                    throttle = 0.1
                    steering = -0.5  # Steer left
                    print("-> Action: TURNING LEFT")
                    
                elif class_id in [2, 7]:  # Turn Right Ahead / Keep Right
                    throttle = 0.1
                    steering = 0.5   # Steer right
                    print("-> Action: TURNING RIGHT")
                    
                elif class_id == 3:  # Straight Ahead
                    throttle = 0.15  # Slightly increase speed
                    steering = 0.0
                    print("-> Action: GOING STRAIGHT")
                    
                elif class_id in [11]: # U Turn
                    throttle = 0.1
                    steering = -1.0 # Max steer left
                    print("-> Action: U-TURN")
                    
                else:
                    # Default / Ignore other signs
                    pass

            # 6. Apply commands to the QCar motors/steering
            car.write(throttle, steering)
            
            # 7. Visualization
            annotated_frame = results[0].plot()
            cv2.imwrite("/tmp/qcar2_detection_latest.jpg", annotated_frame)
            
            # Note: without waitKey, the loop runs as fast as possible. 
            # You can stop the script using Ctrl+C.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # 8. Safe shutdown
        print("Shutting down safely...")
        car.write(0.0, 0.0)
        time.sleep(0.1)
        car.terminate()
        cam.terminate()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
