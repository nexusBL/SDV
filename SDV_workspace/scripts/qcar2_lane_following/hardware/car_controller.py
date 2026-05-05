import numpy as np

try:
    from pal.products.qcar import QCar
except ImportError:
    print("[WARNING] QCar PAL libraries not found. Running CarController in MOCK Mode.")
    QCar = None


class CarController:
    """Low-level motor driver for the QCar2."""

    def __init__(self, config):
        self.cfg = config
        self.car = None
        self._mock_mode = (QCar is None)

        # QCar2 uses 16 int8 LEDs (not 8 float64 like QCar1)
        # QCar2 PAL write mapping expects 8 values and maps them to 16 digital channels
        # LEDs[0:4] -> Indicators, [4]->Brake, [5]->Reverse, [6]->HeadL, [7]->HeadR
        self.led_normal = np.zeros(8, dtype=np.int8)
        self.led_normal[6:8] = 1  # Headlamps on (mapped to 10:16)

        self.led_brake = np.zeros(8, dtype=np.int8)
        self.led_brake[4] = 1     # Brake lights on (mapped to 4:8)
        self.led_brake[6:8] = 1   # Headlamps on

    def initialize(self):
        if self._mock_mode:
            print("[CarController] Simulated controller initialized.")
            return

        print("[CarController] Initializing QCar2 hardware node...")
        self.car = QCar(readMode=1, frequency=100)
        print("[CarController] QCar2 hardware ready ✅")

    def drive(self, throttle, steering):
        """Send throttle and steering commands to the car."""
        if self._mock_mode:
            return

        # Clamp values
        steering = np.clip(steering, -self.cfg.control.max_steering,
                           self.cfg.control.max_steering)
        throttle = np.clip(throttle, -0.3, 0.3)

        self.car.write(throttle, steering, self.led_normal)

    def stop(self):
        """Immediately stop the car."""
        if self._mock_mode:
            return
        self.car.write(0.0, 0.0, self.led_brake)

    def terminate(self):
        if self.car is not None:
            self.stop()
            self.car.terminate()
            print("[CarController] Hardware terminated safely.")
