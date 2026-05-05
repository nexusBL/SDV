#!/usr/bin/env python3
"""
SDV Perception ROS2 Node — Main orchestrator.
Subscribes to camera + LiDAR topics, runs the full perception pipeline,
and publishes annotated results.
"""

import time
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32MultiArray

import torch

from .config import SDVConfig
from .object_detector import ObjectDetector
from .depth_fusion import DepthFusion
from .lane_detector import LaneDetector
from .lidar_processor import LidarProcessor
from .visualization import HUDRenderer


class SDVPerceptionNode(Node):
    """ROS2 node that orchestrates the full SDV perception pipeline."""

    def __init__(self):
        super().__init__('sdv_perception')
        self.get_logger().info('🚀 SDV Perception Node v2.0 Starting...')

        self.cfg = SDVConfig.get()

        # ── QoS for sensor data ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        topics = self.cfg.topics

        # ── Subscribers ──
        # 1. RealSense Camera (used for YOLO + Depth Fusion)
        self.create_subscription(
            Image, topics['rgb_image_yolo'],
            self.yolo_image_callback, sensor_qos,
        )
        self.create_subscription(
            Image, topics['depth_image'],
            self.depth_callback, sensor_qos,
        )
        # 2. CSI Wide Camera (used exclusively for Lane Centering)
        self.create_subscription(
            Image, topics['rgb_image_lane'],
            self.lane_image_callback, sensor_qos,
        )
        # 3. LiDAR
        self.create_subscription(
            LaserScan, topics['lidar_scan'],
            self.lidar_callback, sensor_qos,
        )

        # ── Publishers ──
        self.pub_annotated = self.create_publisher(
            Image, topics['annotated_image'], 10,
        )
        self.pub_detections = self.create_publisher(
            String, topics['detections_json'], 10,
        )
        self.pub_lane = self.create_publisher(
            Float32MultiArray, topics['lane_data'], 10,
        )
        self.pub_threat = self.create_publisher(
            String, topics['threat_level'], 10,
        )

        # ── Initialize perception modules ──
        self.get_logger().info('🔧 Loading YOLOv8 on GPU...')
        self.detector = ObjectDetector(self.cfg)
        self.get_logger().info('✅ YOLOv8 GPU warmed up!')

        cam = self.cfg.camera
        self.lane_detector = LaneDetector(
            cam['width'], cam['height'], self.cfg,
        )
        self.depth_fusion = DepthFusion(self.cfg)
        self.lidar_proc = LidarProcessor(self.cfg)
        self.hud = HUDRenderer(self.cfg)

        # ── State ──
        self.latest_depth = None
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps = 0.0

        perf = self.cfg.performance
        self.fps_interval = perf['fps_update_interval']

        self.get_logger().info('✅ SDV Perception Node v2.0 READY!')
        if torch.cuda.is_available():
            self.get_logger().info(
                f'   GPU: {torch.cuda.get_device_name(0)}'
            )
        self.get_logger().info(
            f'   Camera: {cam["width"]}x{cam["height"]}'
        )

    # ── Callbacks ──

    def lidar_callback(self, msg):
        self.lidar_proc.update(msg)

    def depth_callback(self, msg):
        self.latest_depth = msg

    def lane_image_callback(self, msg):
        """Processes high-speed CSI frames exclusively for Lane Detection."""
        if len(msg.data) == 0: return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        ).copy()
        
        # We don't draw detections on the lane frame since it's a completely different camera perspective
        annotated_lane, lane_offset, left_ok, right_ok = self.lane_detector.process(frame)
        
        lane_status = ('BOTH' if (left_ok and right_ok)
                       else 'LEFT_ONLY' if left_ok
                       else 'RIGHT_ONLY' if right_ok
                       else 'NONE')
                       
        self.latest_lane_data = {
            'offset': lane_offset,
            'status': lane_status,
            'left_ok': left_ok,
            'right_ok': right_ok
        }

    def yolo_image_callback(self, msg):
        """Processes RealSense frames for YOLO Object Detection & Fusion, acts as main Publisher Loop."""
        t_start = time.time()

        # Decode image
        if len(msg.data) == 0: return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        ).copy()

        # 1) Object Detection (GPU)
        detections = self.detector.detect(frame)

        # 2) Depth Fusion
        if getattr(self, 'latest_depth', None) is not None:
            depth_img = np.frombuffer(
                self.latest_depth.data, dtype=np.uint16
            ).reshape(self.latest_depth.height, self.latest_depth.width)
            detections = self.depth_fusion.fuse(detections, depth_img)

        # 3) Draw detections on RealSense frame
        frame_annotated = ObjectDetector.draw_detections(frame, detections)

        # 4) Retrieve latest Lane Data mathematically (we don't draw the bird's eye on the RealSense frame)
        lane = getattr(self, 'latest_lane_data', {
            'offset': 0.0, 'status': 'NONE', 'left_ok': False, 'right_ok': False
        })
        lane_offset = lane['offset']
        lane_status = lane['status']

        # 5) LiDAR Threat Analysis
        threat, min_dist, zones = self.lidar_proc.analyze()

        # 6) FPS
        self.frame_count += 1
        if self.frame_count % self.fps_interval == 0:
            elapsed = time.time() - self.fps_timer
            self.fps = self.fps_interval / max(elapsed, 1e-6)
            self.fps_timer = time.time()

        latency_ms = (time.time() - t_start) * 1000

        # 7) HUD Overlay (Drawing data onto the RealSense frame)
        frame_annotated = self.hud.draw(
            frame_annotated, detections, lane_offset, lane_status,
            threat, min_dist, zones, latency_ms, self.fps,
        )

        # ── Publish ──

        # Annotated image
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = frame_annotated.shape[0]
        out_msg.width = frame_annotated.shape[1]
        out_msg.encoding = 'bgr8'
        out_msg.step = frame_annotated.shape[1] * 3
        out_msg.data = frame_annotated.tobytes()
        self.pub_annotated.publish(out_msg)

        # Detections JSON
        det_dicts = []
        for d in detections:
            det_dicts.append({
                'class': d.class_name,
                'conf': round(d.confidence, 3),
                'bbox': d.bbox,
                'center': d.center,
                'distance_m': round(d.distance_m, 2),
            })

        det_msg = String()
        det_msg.data = json.dumps({
            'timestamp': msg.header.stamp.sec,
            'detections': det_dicts,
            'lane': {'offset': lane_offset, 'status': lane_status},
            'lidar': {
                'threat': threat,
                'min_dist': min_dist,
                'zones': zones,
            },
            'fps': round(self.fps, 1),
            'latency_ms': round(latency_ms, 1),
        })
        self.pub_detections.publish(det_msg)

        # Lane data
        lane_msg = Float32MultiArray()
        lane_msg.data = [lane_offset, float(left_detected), float(right_detected)]
        self.pub_lane.publish(lane_msg)

        # Threat level
        threat_msg = String()
        threat_msg.data = threat
        self.pub_threat.publish(threat_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SDVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Perception node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
