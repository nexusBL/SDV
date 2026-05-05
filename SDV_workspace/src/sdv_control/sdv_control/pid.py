#!/usr/bin/env python3
"""
PID Controller for SDV Lane Centering
"""

import time

class PIDController:
    """Discrete PID Controller with anti-windup for steering."""
    
    def __init__(self, kp, ki, kd, max_integral):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, current_error: float) -> float:
        """Calculate the PID response for a given current error."""
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 1e-4

        # Proportional
        p_term = self.kp * current_error

        # Integral
        self.integral += current_error * dt
        # Anti-windup
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral

        # Derivative
        derivative = (current_error - self.prev_error) / dt
        d_term = self.kd * derivative

        self.prev_error = current_error
        self.prev_time = current_time

        output = p_term + i_term + d_term
        return output

    def reset(self):
        """Reset internal accumulator (used after recovery or long stops)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
