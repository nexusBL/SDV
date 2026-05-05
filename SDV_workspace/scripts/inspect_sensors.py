#!/usr/bin/env python3
"""
inspect_sensors.py — QCar2 Sensor Format Inspector
Prints output format of:
  1. CSI Front Camera (index 2)
  2. RealSense Depth Camera (via ROS2 /camera/depth_image)

Run:
    python3 inspect_sensors.py
"""

import sys, os, time, threading

# Fix CSI headless mode
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from pal.products.qcar import QCarCameras


# ══════════════════════════════════════════════════════════════════
#  Depth Subscriber Node
# ══════════════════════════════════════════════════════════════════

class DepthInspector(Node):
    def __init__(self):
        super().__init__('depth_inspector')
        self._depth = None
        self._depth_raw = None
        self._timestamp = None
        self._received = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, '/camera/depth_image', self._depth_cb, qos)
        print("[Depth] Subscribed to /camera/depth_image")

    def _depth_cb(self, msg):
        if len(msg.data) == 0:
            return

        # Save raw message info
        self._timestamp = time.time()

        # Raw uint16 frame
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self._depth_raw = raw.copy()

        # Converted to float32 meters (same as sdv_autonomous.py)
        self._depth = raw.astype(np.float32) * 0.001
        self._received = True

        # Print once
        if self._received:
            print("\n" + "="*55)
            print("  DEPTH CAMERA — ROS2 /camera/depth_image")
            print("="*55)
            print(f"  ROS2 msg encoding     : {msg.encoding}")
            print(f"  ROS2 msg height       : {msg.height} px")
            print(f"  ROS2 msg width        : {msg.width} px")
            print(f"  ROS2 msg step         : {msg.step} bytes/row")
            print(f"  Raw buffer dtype      : uint16")
            print(f"  Raw buffer shape      : {raw.shape}")
            print(f"  Raw value range       : min={raw.min()}, max={raw.max()} (in mm)")
            print(f"  Converted dtype       : {self._depth.dtype}")
            print(f"  Converted shape       : {self._depth.shape}")
            print(f"  Converted value range : min={self._depth.min():.3f}m, max={self._depth.max():.3f}m")
            print(f"  Timestamp (epoch)     : {self._timestamp:.4f}")

            # Sample center pixel
            cy, cx = msg.height // 2, msg.width // 2
            center_val = self._depth[cy, cx]
            print(f"  Center pixel [{cy},{cx}]  : {center_val:.3f} m")

            # NaN / zero analysis
            zeros = np.sum(self._depth == 0)
            total = self._depth.size
            print(f"  Zero/invalid pixels   : {zeros} / {total} ({100*zeros/total:.1f}%)")
            print("="*55)


# ══════════════════════════════════════════════════════════════════
#  CSI Front Camera Inspector
# ══════════════════════════════════════════════════════════════════

def inspect_csi():
    cameras = None
    try:
        print("\n[CSI] Initializing Front Camera (index=2)...")
        cameras = QCarCameras(
            frameWidth=820,
            frameHeight=616,
            frameRate=80,
            enableRight=False,
            enableBack=False,
            enableFront=True,
            enableLeft=False
        )
        print("[CSI] Camera initialized ✅")

        # Warm up — read a few frames
        for _ in range(5):
            cameras.readAll()
            time.sleep(0.05)

        # Measure FPS over 30 frames
        timestamps = []
        for i in range(30):
            t0 = time.time()
            cameras.readAll()
            img = cameras.csi[2].imageData
            timestamps.append(time.time())
            time.sleep(0.01)

        img = cameras.csi[2].imageData
        ts  = timestamps[-1]

        if img is None:
            print("[CSI] ❌ No image received from Front camera!")
            return

        # FPS estimate
        if len(timestamps) > 1:
            fps_est = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        else:
            fps_est = 0.0

        print("\n" + "="*55)
        print("  CSI FRONT CAMERA (index=2)")
        print("="*55)
        print(f"  dtype                 : {img.dtype}")
        print(f"  shape                 : {img.shape}  → (H, W, C)")
        print(f"  height                : {img.shape[0]} px")
        print(f"  width                 : {img.shape[1]} px")
        print(f"  channels              : {img.shape[2]} (BGR)")
        print(f"  value range           : min={img.min()}, max={img.max()}")
        print(f"  mean pixel value      : {img.mean():.2f}")
        print(f"  estimated FPS         : {fps_est:.1f}")
        print(f"  Timestamp (epoch)     : {ts:.4f}")

        # Sample center pixel
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        print(f"  Center pixel [{cy},{cx}]: BGR={img[cy, cx]}")
        print("="*55)

        return ts

    except Exception as e:
        print(f"[CSI ERROR] {e}")
        import traceback; traceback.print_exc()

    finally:
        if cameras is not None:
            try:
                cameras.terminate()
                print("[CSI] Terminated safely.")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════
#  Timestamp Sync Check
# ══════════════════════════════════════════════════════════════════

def check_sync(csi_ts, depth_ts):
    if csi_ts is None or depth_ts is None:
        print("\n[SYNC] Could not compare — one sensor missing.")
        return

    delta_ms = abs(csi_ts - depth_ts) * 1000
    print("\n" + "="*55)
    print("  TIMESTAMP SYNC ANALYSIS")
    print("="*55)
    print(f"  CSI timestamp         : {csi_ts:.4f}")
    print(f"  Depth timestamp       : {depth_ts:.4f}")
    print(f"  Delta                 : {delta_ms:.1f} ms")

    if delta_ms < 50:
        print("  Sync status           : ✅ GOOD (< 50ms)")
    elif delta_ms < 100:
        print("  Sync status           : ⚠️  OK (< 100ms)")
    else:
        print("  Sync status           : ❌ BAD (> 100ms) — needs alignment")
    print("="*55)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   QCar2 Sensor Format Inspector                     ║")
    print("║   CSI Front (2) + RealSense Depth                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: CSI Camera ──────────────────────────────────────
    print("─── [1/2] Inspecting CSI Front Camera ───")
    csi_ts = inspect_csi()

    # ── Step 2: Depth via ROS2 ──────────────────────────────────
    print("\n─── [2/2] Inspecting Depth Camera via ROS2 ───")
    print("[ROS2] Initializing...")
    rclpy.init()
    depth_node = DepthInspector()

    # Spin until we get one depth frame (max 10 seconds)
    deadline = time.time() + 10.0
    ros_thread = threading.Thread(target=rclpy.spin, args=(depth_node,), daemon=True)
    ros_thread.start()

    while not depth_node._received and time.time() < deadline:
        time.sleep(0.1)

    if not depth_node._received:
        print("[Depth] ❌ No depth frame received in 10s!")
        print("        → Is 'ros2 run qcar2_nodes rgbd' running?")

    depth_ts = depth_node._timestamp

    # ── Step 3: Sync check ──────────────────────────────────────
    check_sync(csi_ts, depth_ts)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY — What we are working with")
    print("="*55)
    print("  CSI Front  → numpy uint8  (H=616, W=820, C=3) BGR")
    print("  Depth      → numpy float32 (H, W) in METERS")
    print("  For fusion → need to ALIGN resolution + timestamp")
    print("="*55)
    print("\n[Done] ✅  Next step: Task 2 — define fusion confidence parameter\n")

    # Cleanup
    try:
        depth_node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
