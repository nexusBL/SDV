import numpy as np
import time

class QCarOdometry:
    def __init__(self, wheelbase=0.256, cps_to_mps=6.866e-6):
        self.L = wheelbase
        self.cps_to_mps = cps_to_mps
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        self.last_ts = time.time()

    def update(self, motor_tach, steering_angle):
        """
        Update state using Ackermann kinematics.
        motor_tach: Speed in counts/sec
        steering_angle: Steering angle in radians
        """
        current_ts = time.time()
        dt = current_ts - self.last_ts
        self.last_ts = current_ts

        # 1. Calculate Velocity (v)
        v = motor_tach * self.cps_to_mps

        # 2. Kinematic equations
        x, y, theta = self.state
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / self.L) * np.tan(steering_angle)

        # 3. Integrate
        self.state += np.array([dx, dy, dtheta]) * dt
        
        # Keep theta within [-pi, pi]
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        return self.state

    def get_transform_matrix(self):
        """Returns 4x4 Homegenous Transform from World to Car frame."""
        x, y, theta = self.state
        c, s = np.cos(theta), np.sin(theta)
        
        # Transformation matrix: [R | T ; 0 | 1]
        # Note: Point clouds have (X=Right, Y=Down, Z=Forward)
        # Global map has (X=Forward, Y=Left, Z=Up)
        # We need to map RealSense local to Global World.
        
        # Rotation around Z (up)
        # World X = Forward, World Y = Left
        return np.array([
            [c, -s, 0, x],
            [s,  c, 0, y],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

