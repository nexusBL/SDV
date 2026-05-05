#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Resolve headless GPU authorization
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

# Add QUANSER libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

try:
    from pal.products.qcar import QCar, QCarCameras
except ImportError:
    print("FATAL: QCar library not loaded. Check paths.")
    sys.exit(1)

from rvblf_vision import RVBLFVisionV2

class FilteredPID:
    """PID Controller with Low-Pass Filtered Derivative for smooth steering."""
    def __init__(self, kp=2.5, ki=0.2, kd=0.8, d_alpha=0.3, max_steer=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.d_alpha = d_alpha
        self.max_steer = max_steer
        
        self.prev_error = 0.0
        self.integrated_error = 0.0
        self.filtered_d = 0.0
        self.last_output = 0.0

    def compute(self, error, dt):
        if error is None: return self.last_output
        
        # P
        p_val = self.kp * error
        
        # I
        self.integrated_error += error * dt
        self.integrated_error = np.clip(self.integrated_error, -0.5, 0.5)
        i_val = self.ki * self.integrated_error
        
        # D (Filtered)
        raw_d = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.filtered_d = (self.d_alpha * self.filtered_d) + (1 - self.d_alpha) * raw_d
        d_val = self.kd * self.filtered_d
        
        self.prev_error = error
        
        output = p_val + i_val + d_val
        self.last_output = np.clip(output, -self.max_steer, self.max_steer)
        return self.last_output

def main():
    print("=========================================")
    print(" Robust RVBLF V2 (BEV + Sliding Windows) ")
    print("=========================================")

    # Native resolution for QCar2
    W, H = 820, 410
    
    vc = RVBLFVisionV2(width=W, height=H)
    # Reduced KP for less twitchy response, maintained KD for damping
    pid = FilteredPID(kp=1.5, ki=0.1, kd=1.0, d_alpha=0.5)
    
    BASE_SPEED = 0.12
    
    # Init Hardware
    my_car = QCar(readMode=0)
    my_cams = QCarCameras(
        frameWidth=W,
        frameHeight=H,
        frameRate=30,
        enableFront=True
    )
    
    print("\nWaiting 2 seconds for Camera to initialize...")
    time.sleep(2.0)
    
    print("Engaging RVBLF V2! Bird's Eye View active.")
    
    # State Machine
    STATE_FOLLOWING = "FOLLOWING"
    STATE_BLIND_GAP = "BLIND_GAP"
    
    current_state = STATE_FOLLOWING
    blind_start_time = 0.0
    BLIND_TIMEOUT = 1.5 
    
    t_last = time.time()
    loop_count = 0
    
    try:
        while True:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            
            my_cams.readAll()
            frame = my_cams.csiFront.imageData
            if frame is None: continue
            
            # Perception: error in meters
            error_m, found = vc.process_frame(frame)
            
            if current_state == STATE_FOLLOWING:
                if found:
                    steer_cmd = pid.compute(error_m, dt)
                    speed = BASE_SPEED
                else:
                    print("\n[RVBLF V2] ⚠️ Gap detected. Entering Blind Mode.")
                    current_state = STATE_BLIND_GAP
                    blind_start_time = t_now
                    steer_cmd = pid.last_output
                    speed = BASE_SPEED
                    
            elif current_state == STATE_BLIND_GAP:
                if found:
                    print("[RVBLF V2] ✓ Re-locked! Resuming Following.")
                    current_state = STATE_FOLLOWING
                    steer_cmd = pid.compute(error_m, dt)
                    speed = BASE_SPEED
                else:
                    if (t_now - blind_start_time) > BLIND_TIMEOUT:
                        print("[RVBLF V2] 🛑 Safe Halt: No lane markers found.")
                        speed, steer_cmd = 0.0, 0.0
                    else:
                        steer_cmd = pid.last_output
                        speed = BASE_SPEED
            
            # Command Hardware
            LEDS = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=np.float64) if found else \
                   np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
            
            my_car.read_write_std(speed, steer_cmd, LEDS)
            
            if loop_count % 15 == 0:
                sys.stdout.write(f"\r[RVBLF V2] State: {current_state:10s} | Err: {error_m:+.3f}m | Steer: {steer_cmd:+.3f}   ")
                sys.stdout.flush()
                
            loop_count += 1
            # Maintain ~30Hz
            time.sleep(max(0.005, 0.033 - (time.time() - t_now)))

    except KeyboardInterrupt:
        print("\nUser Abort.")
    finally:
        my_car.read_write_std(0.0, 0.0, np.zeros(8))
        my_car.terminate()
        my_cams.terminate()
        print("Hardware Release Complete.")

if __name__ == "__main__":
    main()
