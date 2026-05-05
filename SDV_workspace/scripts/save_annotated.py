
import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2

import numpy as np



class Saver(Node):

    def __init__(self):

        super().__init__('saver')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,

                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.saved = 0

        self.sub = self.create_subscription(

            Image, '/sdv/perception/annotated', self.cb, qos)



    def cb(self, msg):

        if len(msg.data) == 0:

            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        self.saved += 1

        path = f'/home/nvidia/Desktop/SDV_workspace/scripts/annotated_{self.saved}.jpg'

        cv2.imwrite(path, frame)

        print(f'Saved: {path}')

        if self.saved >= 3:

            raise SystemExit



rclpy.init()

node = Saver()

try:

    rclpy.spin(node)

except SystemExit:

    print('Done!')

