"""
kinematic_tracker.py - Dead-Reckoning Position Estimator
========================================================
A software mock for localization. Since OptiTrack overhead cameras aren't 
active to provide ground-truth [X, Y, Yaw], this script integrates the 
vehicle's commanded speed and steering over time using a standard 
Bicycle Kinematic Model.

Warning: This will drift over time! Only use for testing routing logic.
"""

import math
import time

class KinematicTracker:
    """
    Estimates the 2D position and orientation of the QCar.
    """
    def __init__(self, config, start_pose=(0.0, 0.0, 0.0)):
        """
        Args:
            config: AppConfig
            start_pose: tuple (x, y, theta_rad)
        """
        self.cfg = config.pure_pursuit
        self.x, self.y, self.theta = start_pose
        self.last_time = time.time()
        
    def reset(self, pose):
        """Forces the tracker to a new starting position."""
        self.x, self.y, self.theta = pose
        self.last_time = time.time()
        
    def get_pose(self):
        return [self.x, self.y, self.theta]

    def update_from_sensors(self, gyro_yaw_rate, motor_tach, commanded_velocity):
        """
        Updates pose using REAL sensor feedback from the QCar2 HAL.
        
        Uses the IMU gyroscope for heading (eliminates steering model errors)
        and the motor encoder tachometer for speed estimation (eliminates
        throttle-to-velocity guesswork).
        
        Args:
            gyro_yaw_rate: float - Gyroscope z-axis reading in rad/s.
            motor_tach: float - Motor encoder tachometer value.
            commanded_velocity: float - The commanded velocity, used as speed
                                        estimate scaled by tach direction.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Prevent huge jumps if the loop stalls
        if dt > 0.1:
            dt = 0.033  # Assume 30Hz nominal if stalled
            
        self.last_time = current_time

        # Use gyroscope for heading change (much more accurate than steering model)
        self.theta += gyro_yaw_rate * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # Use REAL sensor feedback (tachometer outputs m/s directly)
        speed = motor_tach
        
        if abs(speed) < 0.01:
            return

        # Integrate position using real heading
        self.x += speed * math.cos(self.theta) * dt
        self.y += speed * math.sin(self.theta) * dt

    def update(self, velocity_ms, steering_rad):
        """
        FALLBACK: Iterates the Bicycle Kinematic Model by one timestep.
        Only used when sensor feedback is unavailable.
        
        Args:
            velocity_ms: float - Commanded speed in meters per second.
            steering_rad: float - Commanded steering angle in radians.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Prevent huge jumps if the loop stalls
        if dt > 0.1:
            dt = 0.033 # Assume 30Hz nominal if stalled
            
        self.last_time = current_time

        # If we are stopped, position doesn't change
        if abs(velocity_ms) < 0.01:
            return

        # Bicycle model integration
        d_theta = (velocity_ms * math.tan(steering_rad)) / self.cfg.wheelbase
        
        self.x += velocity_ms * math.cos(self.theta) * dt
        self.y += velocity_ms * math.sin(self.theta) * dt
        self.theta += d_theta * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
