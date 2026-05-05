"""
car_controller.py - QCar2 Motor & LED Hardware Abstraction
==========================================================
Maps logical throttle/steering commands to the QCar2 PAL write API.
Includes strict safety clamping to prevent runaway scenarios and
LED state management for headlights, braking, and hazard indicators.
"""

import numpy as np

try:
    from pal.products.qcar import QCar
except ImportError:
    print("[WARNING] QCar PAL library not found. CarController runs in MOCK mode.")
    QCar = None


class CarController:
    """
    Controls the QCar2 drive motors, steering servo, and LED array.

    The QCar2 write API expects:
        car.write(throttle, steering, LEDs)
    where:
        throttle: float [-1.0, 1.0] (clamped to ±0.3 for safety)
        steering: float [-0.5, 0.5] radians
        LEDs:     list of 8 ints (0 or 1)

    Methods:
        initialize()              - Opens QCar hardware node
        drive(throttle, steering) - Sends movement command with headlights
        stop()                    - Zero throttle/steering with brake LEDs
        hazard_stop()             - Zero throttle with alternating hazard LEDs
        terminate()               - Guarantees zero-command and releases hardware
    """

    def __init__(self, config):
        """
        Args:
            config: AppConfig instance containing control and LED parameters.
        """
        self.cfg = config
        self.car = None
        self._mock_mode = (QCar is None)

    def initialize(self):
        """Opens the QCar2 hardware communication node."""
        if self._mock_mode:
            print("[CarController] MOCK mode - no physical car.")
            return

        print("[CarController] Initializing QCar2 hardware node...")
        try:
            self.car = QCar(readMode=1, frequency=100)
            print("[CarController] ✓ QCar2 hardware ready.")
        except Exception as e:
            print(f"[CarController] ✗ FATAL: QCar init failed: {e}")
            raise

    def read(self):
        """
        Polls the hardware for the latest sensor samples (encoders, etc).
        CRITICAL: Many PAL implementations require a read() call to keep the
        hardware watchdog alive and synchronize the write() commands.
        """
        if self._mock_mode or self.car is None:
            return
        try:
            self.car.read()
        except Exception:
            pass

    def drive(self, throttle, steering):
        """
        Sends throttle and steering commands with safety clamping.

        Args:
            throttle: float - desired throttle [-1.0, 1.0]
            steering: float - desired steering angle in radians
        """
        if self._mock_mode:
            return

        # Safety hard clamp: prevent high-speed runaway
        throttle = max(min(throttle, 0.3), -0.3)
        steering = max(min(steering, self.cfg.control.max_steering),
                       -self.cfg.control.max_steering)

        # DEBUG: Only print if significant command or first few loops
        if abs(throttle) > 0.01:
            print(f"[HW_CMD] Throttle: {throttle:.3f}, Steering: {steering:.3f}")

        self.car.write(throttle, steering, self.cfg.leds.headlights)

    def stop(self):
        """Sends zero-command with brake indicator LEDs."""
        if self._mock_mode:
            return
        self.car.write(0.0, 0.0, self.cfg.leds.braking)

    def avoid(self, throttle, steering):
        """
        Sends a commanded avoidance move with right blinker LEDs.
        Used during the STATE_AVOIDING maneuver.
        """
        if self._mock_mode:
            return
        throttle = max(min(throttle, 0.3), -0.3)
        steering = max(min(steering, self.cfg.control.max_steering),
                       -self.cfg.control.max_steering)
        # DEBUG
        print(f"[HW_CMD:AVOID] Throttle: {throttle:.3f}, Steering: {steering:.3f}")
        self.car.write(throttle, steering, self.cfg.leds.right_blinker)

    def reverse(self, throttle):
        """Sends a slow reverse command with brake LEDs."""
        if self._mock_mode:
            return
        throttle = max(min(throttle, 0.0), -0.2)  # only allow negative
        # DEBUG
        print(f"[HW_CMD:REVERSE] Throttle: {throttle:.3f}")
        self.car.write(throttle, 0.0, self.cfg.leds.braking)

    def hazard_stop(self):
        """Sends zero-command with alternating hazard warning LEDs."""
        if self._mock_mode:
            return
        self.car.write(0.0, 0.0, self.cfg.leds.hazard)

    def terminate(self):
        """
        Guarantees a zero-command is sent before releasing hardware.
        This prevents the car from continuing to drive if the script crashes.
        """
        if self.car is not None:
            try:
                # CRITICAL: Always send stop before terminating
                self.car.write(0.0, 0.0, self.cfg.leds.off)
                self.car.terminate()
                print("[CarController] ✓ QCar2 hardware terminated safely.")
            except Exception as e:
                print(f"[CarController] Warning during termination: {e}")
