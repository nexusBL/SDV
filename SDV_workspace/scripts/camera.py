
"""

camera.py — RealSense D435 wrapper for QCar2 (replaces Quanser Camera3D)

"""

import numpy as np

import rclpy

from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

import threading



class SafeCamera3D:

    def __init__(self, verbose=True):

        self.verbose = verbose

        self.rgb   = None

        self.depth = None

        self._lock = threading.Lock()

        

        rclpy.init(args=None)

        self._node = rclpy.create_node('camera_reader')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,

                         history=HistoryPolicy.KEEP_LAST, depth=1)

        

        self._node.create_subscription(Image, '/camera/color_image', self._rgb_cb, qos)

        self._node.create_subscription(Image, '/camera/depth_image', self._depth_cb, qos)

        

        self._thread = threading.Thread(target=self._spin, daemon=True)

        self._thread.start()

        if verbose:

            print("[SafeCamera3D] RealSense D435 connected via ROS2!")



    def _spin(self):

        rclpy.spin(self._node)



    def _rgb_cb(self, msg):

        if len(msg.data) == 0: return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        with self._lock:

            self.rgb = frame.copy()



    def _depth_cb(self, msg):

        if len(msg.data) == 0: return

        frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

        with self._lock:

            self.depth = (frame.astype(np.float32) * 0.001)



    def read(self):

        with self._lock:

            if self.rgb is None:

                return None, None

            return self.rgb.copy(), self.depth.copy() if self.depth is not None else None



    def force_reset(self):

        pass



    def terminate(self):

        self._node.destroy_node()

        print("[SafeCamera3D] Terminated.")

