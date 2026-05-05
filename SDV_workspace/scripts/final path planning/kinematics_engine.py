import numpy as np

class KinematicsEngine:
    def __init__(self, x0=0.0, y0=0.0, th0=0.0):
        """
        Initializes the car's state: x, y (meters) and theta (radians).
        """
        self.x = x0
        self.y = y0
        self.theta = th0
        
        # Track previous encoder counts for incremental movement
        self.prev_counts = None
        
        # Physical constants for QCar
        self.counts_per_rev = 720 # Quad encoder 
        self.gear_ratio = 13.0 # Approximate
        self.wheel_diameter = 0.1 # 10 cm in meters
        self.wheel_base = 0.256 # Distance between axles
        
        # Calibration constant: Meters per encoder count
        self.m_per_cnt = (np.pi * self.wheel_diameter) / (self.counts_per_rev * self.gear_ratio)

    def update_state(self, current_counts, steering_angle, dt):
        """
        Updates [x, y, theta] using a basic Ackermann kinematic model.
        current_counts: Current raw encoder value from myCar.read_write_std()
        steering_angle: Commanded steering angle in radians
        dt: Time delta since last update
        """
        if self.prev_counts is None:
            self.prev_counts = current_counts
            return self.x, self.y, self.theta
        
        # Calculate distance traveled since last frame
        delta_counts = current_counts - self.prev_counts
        self.prev_counts = current_counts
        
        dist = delta_counts * self.m_per_cnt 
        
        # Simple Ackermann motion model
        # dx = v * cos(theta) * dt
        # dy = v * sin(theta) * dt
        # dth = (v / L) * tan(delta) * dt
        
        # Since we use discrete distance (dist = v * dt):
        self.x += dist * np.cos(self.theta)
        self.y += dist * np.sin(self.theta)
        
        # Change in heading
        d_theta = (dist / self.wheel_base) * np.tan(steering_angle)
        self.theta += d_theta
        
        return self.x, self.y, self.theta

    def get_distance_to(self, target_x, target_y):
        """Returns the Euclidean distance to a target point."""
        return np.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)

if __name__ == '__main__':
    # Test
    ke = KinematicsEngine()
    # Simulate move 1000 counts forward at 0 steer
    pos = ke.update_state(1000, 0, 0.1)
    print(f"Position (meters): x={pos[0]:.3f}, y={pos[1]:.3f}, theta={pos[2]:.3f}")
