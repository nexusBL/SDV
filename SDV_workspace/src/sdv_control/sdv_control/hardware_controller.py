#!/usr/bin/env python3
"""
Hardware Controller — Abstraction layer for Quanser QCar2 APIs.
Gracefully handles physical connection or falls back to Mock mode for testing.
"""

import numpy as np

try:
    from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
    HAS_QUANSER = True
except ImportError:
    HAS_QUANSER = False
    IS_PHYSICAL_QCAR = False


class HardwareController:
    """Wrapper for the QCar hardware, offering a clean start/stop interface."""
    
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
        self.car = None
        self._is_active = False

    def start(self):
        """Initialize the QCar connection."""
        if HAS_QUANSER and IS_PHYSICAL_QCAR:
            self.car = QCar(readMode=1, frequency=self.sample_rate)
            self.car.__enter__()
            self._is_active = True
            print("[HardwareController] ✅ Connected to physical QCar2.")
        else:
            print("[HardwareController] ⚠️ Quanser PAL not found or not physical car. Using MOCK hardware.")
            self._is_active = False
            self.car = None

    def stop(self):
        """Safely terminate the QCar connection and ensure zero velocity."""
        if self._is_active and self.car is not None:
            self.car.write(0.0, 0.0, np.zeros(8))
            self.car.__exit__(None, None, None)
            self._is_active = False
            print("🛑 [HardwareController] Connection terminated cleanly. Car stopped.")

    def read(self):
        """Read sensor data from the car."""
        if self._is_active and self.car is not None:
            self.car.read()

    def write(self, throttle: float, steering: float, leds: np.ndarray):
        """Write motor, steering, and LED commands to the hardware."""
        if self._is_active and self.car is not None:
            self.car.write(throttle, steering, leds)

    @property
    def is_active(self):
        return self._is_active

    @property
    def battery_voltage(self):
        return self.car.batteryVoltage if self._is_active else 12.0

    @property
    def motor_current(self):
        return self.car.motorCurrent if self._is_active else 0.0
