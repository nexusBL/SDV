import time
import cv2
import numpy as np
from pal.products.qcar import QCar
from pal.utilities.vision import Camera2D

def main():
    print("Starting Clean Lane Follower from scratch...")
    
    # 1. Initialize Hardware
    myCar = QCar()
    # Use standard Orin/Qcar2 CSI resolution. Camera 3 is front camera.
    cam = Camera2D(cameraId='3', frameWidth=820, frameHeight=410, frameRate=60)
    
    # 2. PID Control Parameters
    Kp = 0.003
    Kd = 0.001
    previous_error = 0
    
    # Base throttle speed
    throttle = 0.08
    
    print("System Ready! Press CTRL+C in the terminal to stop.")
    
    try:
        while True:
            # 3. Read Frame
            cam.read()
            frame = cam.imageData
            if frame is None:
                continue
                
            height, width, _ = frame.shape
            
            # 4. Computer Vision: Crop to Bottom 50% (where the road is)
            roi = frame[int(height/2):height, 0:width]
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define Yellow color range (adjust if needed for your room's lighting)
            lower_yellow = np.array([20, 80, 80])
            upper_yellow = np.array([40, 255, 255])
            
            # Create a mask of the yellow lane
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Clean up noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 5. Lane Calculation (Center of Mass)
            M = cv2.moments(mask)
            if M["m00"] > 0:
                # Calculate the center X pixel of all the yellow blobs
                cx = int(M["m10"] / M["m00"])
                
                # Image center offset (camera is slightly unaligned on QCar)
                target_x = (width / 2) + 20 
                
                # Calculate Error
                error = target_x - cx
                
                # PID Steering Calculation
                steering = (Kp * error) + (Kd * (error - previous_error))
                previous_error = error
                
                # Clamp steering to physical limits
                steering = max(-0.5, min(0.5, steering))
            else:
                # If no yellow is found, drive straight slowly
                steering = 0.0
                error = 0.0

            # 6. Actuate Motors
            myCar.read_write_std(throttle, steering, np.array([0,0,0,0, 1,1,0,0]))
            
            # 7. (Optional) Visual Debugging
            # Draw targeting reticle
            if M["m00"] > 0:
                cv2.circle(roi, (cx, int(M["m01"] / M["m00"])), 10, (0, 0, 255), -1)
            cv2.line(roi, (int(target_x), 0), (int(target_x), roi.shape[0]), (0, 255, 0), 2)
            cv2.putText(roi, f"Error: {error:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            try:
                cv2.imshow("Lane Tracking", roi)
                cv2.waitKey(1)
            except:
                pass # Ignores popup failures if running via SSH 

    except KeyboardInterrupt:
        print("\nCTRL+C Detected. Shutting down safely...")
    except Exception as e:
        print(f"\nError encountered: {e}")
    finally:
        # 8. Guaranteed Hardware Shutdown
        print("Stopping motors and releasing camera.")
        try:
            myCar.read_write_std(0.0, 0.0, np.zeros(8))
        except:
            pass
        myCar.terminate()
        cam.terminate()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
