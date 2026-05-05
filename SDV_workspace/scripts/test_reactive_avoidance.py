import sys
import numpy as np
import math

# Mocking the QCar library parts to allow local testing
class MockLidar:
    def __init__(self, angles, distances):
        self.angles = angles
        self.distances = distances

# Import the controller from the implemented script
# We'll need to mock the imports inside that script or structure it carefully.
# Since I can't easily mock the imports in the existing script without editing it, 
# I'll just copy the class definition for this verification test.

class Config:
    LOOP_HZ                 = 30
    DT                      = 1.0 / LOOP_HZ
    D_SAFE                  = 1.5
    D_CAUTION               = 1.0
    D_CRITICAL              = 0.6
    D_STOP                  = 0.4
    D_RESUME                = 0.5
    THROTTLE_CRUISE         = 0.12
    THROTTLE_MIN            = 0.05
    TTC_THRESHOLD           = 1.5
    MAX_STEER               = 0.5
    STEER_P_GAIN            = 1.5
    STEER_DEADZONE          = 0.02
    STEER_SMOOTH_ALPHA      = 1.0   # Set to 1.0 for testing raw output
    THROTTLE_SMOOTH_ALPHA   = 1.0   # Set to 1.0 for testing raw output
    FRONT_ARC               = 45.0
    MIN_LIDAR_DIST          = 0.05
    MAX_LIDAR_DIST          = 5.0
    MIN_VALID_POINTS        = 5

# ReactiveController implementation (copied for verification)
class ReactiveController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.direction_persistence = 0.0
        self.is_emergency_stop = False

    def process_lidar(self, angles, distances):
        raw_angles = np.array(angles)
        raw_distances = np.array(distances)
        adj_angles = (raw_angles + math.pi) % (2 * math.pi) - math.pi
        mask = (np.abs(np.degrees(adj_angles)) <= self.cfg.FRONT_ARC) & \
               (raw_distances > self.cfg.MIN_LIDAR_DIST) & \
               (raw_distances < self.cfg.MAX_LIDAR_DIST)
        return adj_angles[mask], raw_distances[mask]

    def compute_control(self, angles, distances):
        if angles is None or len(angles) < self.cfg.MIN_VALID_POINTS:
            return self.cfg.THROTTLE_MIN, 0.0, "SPARSE_DATA"
        min_dist = np.min(distances)
        if min_dist < self.cfg.D_STOP: self.is_emergency_stop = True
        elif min_dist > self.cfg.D_RESUME: self.is_emergency_stop = False
        if self.is_emergency_stop: return 0.0, 0.0, "EMERGENCY_STOP"
        weights = (np.maximum(0, self.cfg.D_SAFE - distances)**2) * np.cos(angles)
        v_repulse_x = np.sum(weights * -np.cos(angles))
        v_repulse_y = np.sum(weights * -np.sin(angles))
        left_mask = angles > 0.02
        right_mask = angles < -0.02
        sum_left = np.sum(weights[left_mask]) if np.any(left_mask) else 0.0
        sum_right = np.sum(weights[right_mask]) if np.any(right_mask) else 0.0
        # High sum_left -> Negative bias (Right)
        imbalance_bias = 0.3 * (sum_right - sum_left)
        v_repulse_y += imbalance_bias
        v_total_x = 1.0 + v_repulse_x
        v_total_y = v_repulse_y
        if abs(v_total_y) < 0.1 and abs(self.direction_persistence) > 0.1:
            v_total_y += 0.2 * self.direction_persistence
        target_delta = math.atan2(v_total_y, v_total_x)
        speed_factor = 1.0 - (self.prev_throttle / self.cfg.THROTTLE_CRUISE) * 0.3
        target_steer = np.clip(self.cfg.STEER_P_GAIN * target_delta * speed_factor, 
                               -self.cfg.MAX_STEER, self.cfg.MAX_STEER)
        if abs(target_steer) > 0.1: self.direction_persistence = np.sign(target_steer)
        if min_dist > self.cfg.D_SAFE: target_throttle = self.cfg.THROTTLE_CRUISE
        elif min_dist > self.cfg.D_CRITICAL:
            ratio = (min_dist - self.cfg.D_CRITICAL) / (self.cfg.D_SAFE - self.cfg.D_CRITICAL)
            target_throttle = self.cfg.THROTTLE_MIN + ratio * (self.cfg.THROTTLE_CRUISE - self.cfg.THROTTLE_MIN)
        else: target_throttle = self.cfg.THROTTLE_MIN
        est_v = max(self.prev_throttle * 2.0, 0.05)
        ttc = min_dist / est_v
        if ttc < self.cfg.TTC_THRESHOLD: target_throttle = min(target_throttle, self.cfg.THROTTLE_MIN * 1.2)
        if abs(target_steer) < self.cfg.STEER_DEADZONE: target_steer = 0.0
        self.prev_steer = (self.cfg.STEER_SMOOTH_ALPHA * target_steer) + (1.0 - self.cfg.STEER_SMOOTH_ALPHA) * self.prev_steer
        self.prev_throttle = (self.cfg.THROTTLE_SMOOTH_ALPHA * target_throttle) + (1.0 - self.cfg.THROTTLE_SMOOTH_ALPHA) * self.prev_throttle
        return self.prev_throttle, self.prev_steer, "OK"

def run_test():
    cfg = Config()
    ctrl = ReactiveController(cfg)
    
    # Test Case 1: CLEAR PATH
    print("--- Test 1: Clear Path ---")
    angles = np.radians(np.linspace(-45, 45, 50))
    distances = np.ones(50) * 3.0
    ang, dist = ctrl.process_lidar(angles, distances)
    th, st, info = ctrl.compute_control(ang, dist)
    print(f"Result: Throttle={th:.3f}, Steering={st:.3f} (Expected: ~0.12, 0.0)")

    # Test Case 2: OBSTACLE ON LEFT (positive angles)
    print("\n--- Test 2: Obstacle on Left ---")
    distances = np.ones(50) * 3.0
    distances[30:50] = 0.8  # Left side is closer
    ang, dist = ctrl.process_lidar(angles, distances)
    th, st, info = ctrl.compute_control(ang, dist)
    print(f"Result: Throttle={th:.3f}, Steering={st:.3f} (Expected: Steer Right, negative st)")

    # Test Case 3: OBSTACLE ON RIGHT (negative angles)
    print("\n--- Test 3: Obstacle on Right ---")
    distances = np.ones(50) * 3.0
    distances[0:20] = 0.8  # Right side is closer
    ang, dist = ctrl.process_lidar(angles, distances)
    th, st, info = ctrl.compute_control(ang, dist)
    print(f"Result: Throttle={th:.3f}, Steering={st:.3f} (Expected: Steer Left, positive st)")

    # Test Case 4: EMERGENCY STOP
    print("\n--- Test 4: Emergency Stop ---")
    distances = np.ones(50) * 0.3
    ang, dist = ctrl.process_lidar(angles, distances)
    th, st, info = ctrl.compute_control(ang, dist)
    print(f"Result: Throttle={th:.3f}, Steering={st:.3f} (Expected: 0.0, 0.0)")

    # Test Case 5: CORRIDOR (Left and Right balanced)
    print("\n--- Test 5: Symmetric Corridor ---")
    distances = np.ones(50) * 0.8
    # Add a tiny noise to break symmetry or check imbalance bias
    distances[30:50] = 0.79 # Slightly closer on left
    ang, dist = ctrl.process_lidar(angles, distances)
    th, st, info = ctrl.compute_control(ang, dist)
    print(f"Result: Throttle={th:.3f}, Steering={st:.3f} (Expected: Should steer right due to imbalance bias)")

if __name__ == "__main__":
    run_test()

