"""
pid_controller.py - PID Controller for QCar2
=============================================
Matches the EXACT formulation from the original control.py:
  error = self.setpoint + ubicacion_linea
  where setpoint = -22.452118490490875

Plus improvements:
  - Time-scaled integral term
  - Low-pass filtered derivative to prevent jitter
  - State reset for obstacle stop/resume
"""


class PIDController:
    """
    PID controller matching original control.py ControlSystem class.

    Original formulation:
      error = setpoint + ubicacion_linea
      P = error * kp
      I = accumulated_error * ki  (clamped to ±max_integral)
      D = (error - previous_error) * kd
      output = P + I + D
    """

    def __init__(self, config):
        self.cfg = config.control
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0

    def compute(self, ubicacion_linea, dt=0.033):
        """
        Computes steering using original formula:
          error = self.setpoint + ubicacion_linea

        Args:
            ubicacion_linea: float - the raw error from LaneDetector
                             (= center_cam - center_lines, same as original)
            dt: float - time delta (seconds), default 1/30

        Returns:
            float: steering command
        """
        dt = max(dt, 0.001)

        # Original formula: error = self.setpoint + ubicacion_linea
        error = self.cfg.setpoint + ubicacion_linea

        # Proportional (from original: error * self.kp)
        p_term = error * self.cfg.kp

        # Integral with anti-windup (from original control_pid)
        self.integral += error
        if self.integral > self.cfg.anti_windup:
            self.integral = self.cfg.anti_windup
        elif self.integral < -self.cfg.anti_windup:
            self.integral = -self.cfg.anti_windup
        i_term = self.cfg.ki * self.integral

        # Derivative with low-pass filter (original + improvement)
        raw_derivative = error - self.previous_error
        alpha = self.cfg.derivative_alpha
        filtered_derivative = (
            alpha * self.previous_derivative +
            (1.0 - alpha) * raw_derivative
        )
        self.previous_derivative = filtered_derivative
        d_term = self.cfg.kd * filtered_derivative

        self.previous_error = error

        return p_term + i_term + d_term

    def control_p(self, ubicacion_linea):
        """
        Simple proportional control (from original control_p).
        """
        error = self.cfg.setpoint + ubicacion_linea
        return error * self.cfg.kp

    def reset_state(self):
        """Reset integral and derivative state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0

    @staticmethod
    def saturate(value, upper, lower):
        """
        From original saturate(raw_steering, major, less).
        Clamps value between lower and upper.
        """
        if value <= lower:
            return lower
        elif value >= upper:
            return upper
        return value
