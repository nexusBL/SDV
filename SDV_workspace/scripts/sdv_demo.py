#!/usr/bin/env python3
"""
SDV Demo — Lane Following + Distance-based Auto Stop
Usage: python3 sdv_demo.py --stop 30   (stop at 30cm)
       python3 sdv_demo.py --stop 50   (stop at 50cm)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import LaserScan
import numpy as np
import argparse
import json
import time

try:
    from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
    HAS_QUANSER = True
except ImportError:
    HAS_QUANSER = False
    IS_PHYSICAL_QCAR = False

class PID:
    def __init__(self, kp=0.8, ki=0.05, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt=0.1):
        self.integral  += error * dt
        self.integral   = np.clip(self.integral, -0.3, 0.3)
        derivative      = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return np.clip(self.kp*error + self.ki*self.integral + self.kd*derivative, -0.3, 0.3)

class SDVDemo(Node):
    def __init__(self, stop_distance_cm=30):
        super().__init__('sdv_demo')
        self.stop_distance_m = stop_distance_cm / 100.0
        self.get_logger().info(f'🚗 SDV Demo — Stop at {stop_distance_cm}cm')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # State
        self.lane_offset  = 0.0
        self.left_ok      = False
        self.right_ok     = False
        self.front_dist_m = 99.0
        self.stopped      = False
        self.state        = 'WAITING'
        self.pid          = PID(kp=0.6, ki=0.02, kd=0.05)
        self.leds         = np.zeros(8)

        # Hardware
        if HAS_QUANSER and IS_PHYSICAL_QCAR:
            self.car = QCar(readMode=1, frequency=100)
            self.car.__enter__()
            self.get_logger().info('✅ Physical QCar2 connected!')
        else:
            self.car = None
            self.get_logger().warn('⚠️ Mock mode — no hardware!')

        # Subscribers
        self.create_subscription(Float32MultiArray, '/sdv/perception/lane', self.lane_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, qos)

        # Publisher
        self.pub_status = self.create_publisher(String, '/sdv/demo/status', 10)

        # Control loop 10Hz
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('✅ Ready! Lane following + auto stop active.')

    def lane_cb(self, msg):
        if len(msg.data) >= 3:
            self.lane_offset = float(msg.data[0])
            self.left_ok     = bool(msg.data[1])
            self.right_ok    = bool(msg.data[2])

    def lidar_cb(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(np.degrees(msg.angle_min),
                             np.degrees(msg.angle_max), len(ranges))
        front = (np.abs(angles) < 15) & (ranges > 0.05) & (ranges < 10.0)
        self.front_dist_m = float(np.min(ranges[front])) if front.any() else 99.0

    def control_loop(self):
        lane_detected = self.left_ok or self.right_ok
        throttle  = 0.0
        steering  = 0.0

        # ── State Machine ──
        if self.stopped:
            self.state = 'STOPPED'
            throttle   = 0.0
            steering   = 0.0
            self.leds  = np.array([0,0,1,1,1,1,0,0], dtype=float)  # brake lights

        elif self.front_dist_m <= self.stop_distance_m:
            self.stopped = True
            self.state   = 'STOPPED'
            throttle     = 0.0
            steering     = 0.0
            self.get_logger().info(
                f'🛑 STOPPED! Object at {self.front_dist_m*100:.1f}cm '
                f'(target: {self.stop_distance_m*100:.0f}cm)')

        elif not lane_detected:
            self.state = 'LOST_LANE'
            throttle   = 0.0
            steering   = 0.0

        else:
            self.state = 'DRIVING'
            throttle   = 0.10   # slow safe speed
            # PID steering — negative because left offset = steer right
            steering   = self.pid.compute(-self.lane_offset)
            self.leds  = np.array([1,0,0,0,0,0,0,1], dtype=float)  # headlights

        # ── Send to hardware ──
        if self.car is not None:
            self.car.write(throttle, steering, self.leds)

        # ── Publish status ──
        status = {
            'state':         self.state,
            'throttle':      round(throttle, 3),
            'steering':      round(steering, 3),
            'lane_offset':   round(self.lane_offset, 3),
            'left_ok':       self.left_ok,
            'right_ok':      self.right_ok,
            'front_dist_cm': round(self.front_dist_m * 100, 1),
            'stop_target_cm': self.stop_distance_m * 100,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

        self.get_logger().info(
            f'{self.state} | Lane: {"BOTH" if (self.left_ok and self.right_ok) else "PARTIAL" if (self.left_ok or self.right_ok) else "NONE"} '
            f'| Offset: {self.lane_offset:+.3f} | Steer: {steering:+.3f} '
            f'| Throttle: {throttle:.2f} | Front: {self.front_dist_m*100:.1f}cm'
        )

    def destroy_node(self):
        # Safety — stop car on exit
        if self.car is not None:
            self.car.write(0.0, 0.0, np.zeros(8))
            self.car.__exit__(None, None, None)
            self.get_logger().info('🛑 Car stopped safely.')
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop', type=float, default=30.0,
                        help='Stop distance in cm (default: 30)')
    args = parser.parse_args()

    rclpy.init()
    node = SDVDemo(stop_distance_cm=args.stop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Demo stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
