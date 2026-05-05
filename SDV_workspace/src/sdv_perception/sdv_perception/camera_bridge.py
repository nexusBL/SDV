#!/usr/bin/env python3
"""
SDV Camera Bridge Node - Threaded Version
Official Quanser QCar2 camera IDs (from documentation):
  Right = 0, Rear = 1, Front = 2, Left = 3
Resolution: 820x616 @ 80fps (best balance)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import sys
import os
import threading
import time

# Force headless NVMM capture to prevent EGL authorization crashes over SSH:
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
from pal.utilities.vision import Camera2D


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('sdv_camera_bridge')
        self.get_logger().info('🎥 SDV Camera Bridge Starting...')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.pubs = {
            'front': self.create_publisher(Image, '/sdv/camera/front', qos),
            'rear':  self.create_publisher(Image, '/sdv/camera/rear',  qos),
            'left':  self.create_publisher(Image, '/sdv/camera/left',  qos),
            'right': self.create_publisher(Image, '/sdv/camera/right', qos),
        }

        # Official Quanser QCar2 camera IDs from documentation
        # Right=0, Rear=1, Front=2, Left=3
        self.cam_ids = {
            'front': '2',
            'rear':  '1',
            'left':  '3',
            'right': '0',
        }

        # Resolution: 820x616 @ 80fps (best for autonomous driving)
        self.CAM_WIDTH  = 820
        self.CAM_HEIGHT = 616
        self.CAM_FPS    = 80

        # Start one thread per camera
        self.running = True
        for name, cam_id in self.cam_ids.items():
            t = threading.Thread(
                target=self.camera_thread,
                args=(name, cam_id),
                daemon=True
            )
            t.start()
            self.get_logger().info(f'  ✅ Thread started → CSI {name} (id={cam_id})')
            time.sleep(2.0)  # wait between each camera

        self.get_logger().info('✅ Camera Bridge READY!')
        self.get_logger().info(f'   Resolution: {self.CAM_WIDTH}x{self.CAM_HEIGHT} @ {self.CAM_FPS}fps')
        self.get_logger().info('   Topics:')
        self.get_logger().info('   /sdv/camera/front  → Front  (id=2)')
        self.get_logger().info('   /sdv/camera/rear   → Rear   (id=1)')
        self.get_logger().info('   /sdv/camera/left   → Left   (id=3)')
        self.get_logger().info('   /sdv/camera/right  → Right  (id=0)')

    def camera_thread(self, name, cam_id):
        """Each camera runs independently in its own thread."""
        try:
            cam = Camera2D(
                cameraId=cam_id,
                frameWidth=self.CAM_WIDTH,
                frameHeight=self.CAM_HEIGHT,
                frameRate=self.CAM_FPS
            )
        except Exception as e:
            self.get_logger().error(f'Failed to open {name} camera (id={cam_id}): {e}')
            return

        pub  = self.pubs[name]
        rate = 1.0 / self.CAM_FPS

        while self.running and rclpy.ok():
            t0 = time.time()
            try:
                cam.read()
                if cam.imageData is not None:
                    msg = Image()
                    msg.header.stamp    = self.get_clock().now().to_msg()
                    msg.header.frame_id = f'csi_{name}'
                    msg.height          = cam.imageData.shape[0]
                    msg.width           = cam.imageData.shape[1]
                    msg.encoding        = 'bgr8'
                    msg.step            = cam.imageData.shape[1] * 3
                    msg.data            = cam.imageData.tobytes()
                    pub.publish(msg)
            except Exception:
                pass

            # Sleep to maintain target fps
            elapsed = time.time() - t0
            sleep_t = rate - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        del cam

    def destroy_node(self):
        self.running = False
        time.sleep(0.3)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Camera Bridge shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
