#!/usr/bin/env python3
from pal.products.qcar import QCar
import time

def diag():
    print("--- QCar2 Deep Hardware Diagnostic ---")
    try:
        car = QCar(readMode=1, frequency=100)
        car.read()
        
        v = car.batteryVoltage
        i = car.motorCurrent
        print(f"Battery Voltage: {v:.2f}V")
        print(f"Motor Current: {i:.2f}A")
        
        if v < 10.0:
            print("!!! WARNING: Battery voltage is critically low or Motor Switch is OFF !!!")
            print("Please ensure BOTH power switches on the QCar2 are in the ON position.")
            
        print("\n--- Testing Motor Pulse (0.30 Throttle) ---")
        # Pulse at 0.3 (higher torque) and monitor current
        for i in range(20):
            car.read()
            # Command: throttle, steer, LEDs
            car.write(0.3, 0.0, [1]*8)
            if i % 5 == 0:
                print(f"  [Pulse {i}] Current: {car.motorCurrent:.2f}A")
            time.sleep(0.01)
            
        car.write(0.0, 0.0, [0]*8)
        car.terminate()
        print("\nDiagnostic complete.")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")

if __name__ == "__main__":
    diag()
