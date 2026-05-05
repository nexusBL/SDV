#!/usr/bin/env python3
"""
SDV Control ROS2 Node — Hardware actuations.
Subscribes to perception topics and runs a dedicated hard 100Hz hardware loop
to drive the physical Quanser QCar2 safely, isolated from ROS2 callback jitter.
"""

import time
import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32MultiArray

from sdv_perception.config import SDVConfig
from .hardware_controller import HardwareController
from .pid import PIDController
from .state_machine import StateMachine

class SDVControlNode(Node):
    def __init__(self):
        super().__init__('sdv_control')
        self.get_logger().info('🕹️ SDV Control Node Starting...')

        self.cfg = SDVConfig.get()
        topics = self.cfg.topics

        # ── Hardware & Control Logic ──
        self.hw = HardwareController(sample_rate=self.cfg.hardware['loop_rate_hz'])
        self.sm = StateMachine(self.cfg)
        
        pid_cfg = self.cfg.pid
        self.pid = PIDController(
            kp=pid_cfg['kp'], ki=pid_cfg['ki'], kd=pid_cfg['kd'],
            max_integral=pid_cfg['max_integral']
        )
        
        self.steering_max = self.cfg.hardware['steering_max']
        self.steering_threshold = self.cfg.hardware['steering_threshold']

        # ── Shared State (Thread Protected) ──
        self._lock = threading.Lock()
        self.latest_lane_offset = 0.0
        self.latest_lane_status = 'NONE'
        self.latest_threat = 'UNKNOWN'
        self.last_msg_time = time.time()
        
        # ── Subscribers ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Float32MultiArray, topics['lane_data'], self.lane_callback, qos)
        self.create_subscription(String, topics['threat_level'], self.threat_callback, qos)

        # ── Publishers ──
        self.pub_telemetry = self.create_publisher(String, '/sdv/control/telemetry', 10)

        # ── Hardware 100Hz Thread ──
        self.hw.start()
        self.is_running = True
        self.hw_thread = threading.Thread(target=self._hardware_loop, daemon=True)
        self.hw_thread.start()

        self.get_logger().info('✅ SDV Control Node READY and armed!')

    def lane_callback(self, msg):
        """Update lane state safely."""
        with self._lock:
            self.latest_lane_offset = msg.data[0]
            left_ok = bool(msg.data[1])
            right_ok = bool(msg.data[2])
            
            if left_ok and right_ok: self.latest_lane_status = 'BOTH'
            elif left_ok:            self.latest_lane_status = 'LEFT_ONLY'
            elif right_ok:           self.latest_lane_status = 'RIGHT_ONLY'
            else:                    self.latest_lane_status = 'NONE'

            self.last_msg_time = time.time()

    def threat_callback(self, msg):
        """Update LiDAR threat level safely."""
        with self._lock:
            self.latest_threat = msg.data
            self.last_msg_time = time.time()

    def _hardware_loop(self):
        """100Hz dedicated hardware actuation loop ensuring absolute real-time control."""
        rate = 1.0 / self.cfg.hardware['loop_rate_hz']
        
        while self.is_running:
            t0 = time.time()
            
            with self._lock:
                lane_offset = self.latest_lane_offset
                lane_status = self.latest_lane_status
                threat = self.latest_threat
                msg_age = t0 - self.last_msg_time
            
            # Watchdog Timer: Emergency break if Perception AI freezes or ROS bridge dies
            if msg_age > 0.5:
                # E-STOP overriding AI
                throttle = 0.0
                steering = 0.0
                state_str = 'WATCHDOG_TIMEOUT'
                self.pid.reset()
            else:
                # 1. State Machine determines maximum safe Throttle
                throttle = self.sm.evaluate(threat, lane_status)
                state_str = self.sm.current_state
                
                # 2. PID Control determines accurate Steer (only if moving)
                if state_str in [StateMachine.STATE_DRIVING, StateMachine.STATE_AVOIDING]:
                    pid_out = self.pid.compute(lane_offset)
                    # Hard clamping for steering rack safety
                    steering = max(-self.steering_max, min(self.steering_max, pid_out))
                else:
                    steering = 0.0
                    self.pid.reset()

            # 3. Vehicle Indicator Lights Logic
            leds = np.zeros(8)
            if steering > self.steering_threshold:    # Turning Right
                leds[0] = 1; leds[2] = 1
            elif steering < -self.steering_threshold: # Turning Left
                leds[1] = 1; leds[3] = 1
                
            if throttle < 0: # Reversing or Active Braking
                leds[4] = 1; leds[5] = 1

            # 4. Dispatch Commands Physically
            self.hw.read()
            self.hw.write(throttle, float(steering), leds)
            
            # 5. Output sparse debug telemetry at 10Hz
            if int(t0 * 100) % 10 == 0:
                tel = {
                    'state': state_str,
                    'throttle': round(throttle, 3),
                    'steering': round(float(steering), 3),
                    'battery': round(self.hw.battery_voltage, 2),
                    'watchdog': round(msg_age, 3)
                }
                msg = String()
                msg.data = json.dumps(tel)
                self.pub_telemetry.publish(msg)

            # Strictly regulate 100Hz timing loop
            elapsed = time.time() - t0
            time.sleep(max(0.0, rate - elapsed))

    def destroy_node(self):
        """Teardown connections gracefully on ROS interrupt."""
        self.is_running = False
        if self.hw_thread.is_alive():
            self.hw_thread.join(timeout=1.0)
        self.hw.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SDVControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Control node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
