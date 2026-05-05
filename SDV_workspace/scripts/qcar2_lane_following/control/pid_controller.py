class PIDController:
    """
    PID controller for lateral steering control.

    Input:  error in METERS (lateral offset from lane center)
    Output: steering angle in RADIANS
    """

    def __init__(self, config):
        self.cfg = config
        self.integrated_error = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.last_steering = 0.0

    def compute(self, error_m, dt=0.033):
        """
        Compute steering command from lateral offset error.

        Args:
            error_m: Lateral offset in meters. Positive = car right of center.
                     None = no detection, hold last command.
            dt:      Time step in seconds.

        Returns:
            steering: Steering angle in radians, clamped to max_steering.
        """
        if error_m is None:
            # No lane detected — hold last steering command
            return self.last_steering

        # Proportional
        p = self.cfg.control.kp * error_m

        # Integral with anti-windup clamping
        self.integrated_error += error_m * dt
        self.integrated_error = max(
            min(self.integrated_error, self.cfg.control.anti_windup),
            -self.cfg.control.anti_windup
        )
        i = self.cfg.control.ki * self.integrated_error

        # Derivative with low-pass filter to reduce noise
        raw_d = (error_m - self.previous_error) / max(dt, 0.001)
        alpha = self.cfg.control.d_filter_alpha
        self.filtered_derivative = alpha * self.filtered_derivative + (1 - alpha) * raw_d
        d = self.cfg.control.kd * self.filtered_derivative

        self.previous_error = error_m

        # Sum and clamp
        steering = p + i + d
        max_s = self.cfg.control.max_steering
        steering = max(min(steering, max_s), -max_s)

        self.last_steering = steering
        return steering

    def reset_state(self):
        """Reset all PID state. Call when car stops or mode changes."""
        self.integrated_error = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.last_steering = 0.0
