#!/usr/bin/env python3

"""

qcar_multisensor_lane_drive.py

==============================

QCar2 lane following using ALL active sensors with clear roles:



  CSI Front Camera   -> main lane detection / steering

  RealSense Depth    -> forward-clearance confidence / obstacle veto

  LiDAR              -> hard emergency stop

  RealSense RGB      -> fallback/debug only



Design goals:

  - run on Quanser map

  - follow lane center and turn with the lane

  - stop safely on close obstacle

  - work in headless SSH mode by default

  - optional GUI with --gui



Usage:

    python3 qcar_multisensor_lane_drive.py --preview

    python3 qcar_multisensor_lane_drive.py

    python3 qcar_multisensor_lane_drive.py --gui

    python3 qcar_multisensor_lane_drive.py --preview --no-launch

"""



import os

import sys

import time

import math

import signal

import argparse

import threading

import subprocess



import numpy as np

import cv2



SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPTS_DIR)

sys.path.insert(0, "/home/nvidia/Documents/Quanser/0_libraries/python")



import rclpy

from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan



try:

    from pal.products.qcar import QCar, QCarLidar, QCarCameras

    HAS_QCAR = True

except ImportError:

    HAS_QCAR = False

    print("[QCar] PAL not found -> preview only")



from lane_detector import LaneDetector

from config import (

    CAMERA_FPS,

    SPEED_BASE,

    SPEED_MIN,

    SPEED_MAX,

    STEER_MAX_RAD,

    PID_KP,

    PID_KI,

    PID_KD,

    PID_INTEGRAL_MAX,

    NO_LANE_STOP_FRAMES,

    LEDS_DEFAULT,

)



# ============================================================

# TUNING

# ============================================================



CSI_W = 820

CSI_H = 616

CSI_FPS = 80



# LiDAR

LIDAR_STOP_DISTANCE_M = 0.50

LIDAR_RESUME_DISTANCE_M = 0.65

LIDAR_FRONT_DEG = 15.0



# Depth

DEPTH_STALE_MS = 200.0

DEPTH_MIN_VALID_M = 0.10

DEPTH_MAX_VALID_M = 4.00

DEPTH_OBSTACLE_M = 0.55

DEPTH_OBS_PIXELS = 80



# Depth ROI in native RealSense depth frame

# center-bottom region looking ahead

DEPTH_ROI_TOP = 0.45

DEPTH_ROI_BOTTOM = 0.85

DEPTH_ROI_LEFT = 0.35

DEPTH_ROI_RIGHT = 0.65



# Fusion

CONF_DRIVE = 0.60

CONF_CAUTION = 0.30

DEPTH_SCORE_STALE = 0.40



# Logging

LOG_EVERY_N = 10





# ============================================================

# ARGUMENTS

# ============================================================



def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--preview", action="store_true",

                   help="Run without motor commands")

    p.add_argument("--gui", action="store_true",

                   help="Show OpenCV window (only use locally, not headless SSH)")

    p.add_argument("--no-launch", action="store_true",

                   help="Do not launch ROS2 rgbd/lidar nodes internally")

    p.add_argument("--stop", type=float, default=50.0,

                   help="LiDAR hard stop distance in cm (default 50)")

    return p.parse_args()





# ============================================================

# ROS2 NODE LAUNCHER

# ============================================================



class ROSNodeLauncher:

    def __init__(self):

        self._procs = []

        self._source = (

            "source /opt/ros/humble/setup.bash && "

            "source /home/nvidia/Documents/Quanser/5_research/sdcs/qcar2/ros2/install/setup.bash && "

            "source /home/nvidia/ros2/install/setup.bash"

        )



    def launch(self, package, executable, env_prefix=""):

        cmd = f"{env_prefix}{self._source} && ros2 run {package} {executable}"

        proc = subprocess.Popen(

            cmd,

            shell=True,

            executable="/bin/bash",

            stdout=subprocess.DEVNULL,

            stderr=subprocess.DEVNULL,

            preexec_fn=os.setsid,

        )

        self._procs.append(proc)

        print(f"[Launcher] Started: ros2 run {package} {executable} (pid={proc.pid})")

        return proc



    def stop_all(self):

        for proc in self._procs:

            try:

                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

            except Exception:

                pass

        self._procs.clear()

        subprocess.run(["pkill", "-9", "-f", "qcar2_nodes"], capture_output=True)

        print("[Launcher] ROS2 nodes stopped.")





# ============================================================

# PID

# ============================================================



class PID:

    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD):

        self.kp = kp

        self.ki = ki

        self.kd = kd

        self._integral = 0.0

        self._prev_err = 0.0

        self._prev_time = None



    def reset(self):

        self._integral = 0.0

        self._prev_err = 0.0

        self._prev_time = None



    def update(self, error):

        now = time.time()

        dt = max((now - self._prev_time) if self._prev_time else 0.033, 1e-6)

        self._prev_time = now



        self._integral = np.clip(

            self._integral + error * dt,

            -PID_INTEGRAL_MAX,

            PID_INTEGRAL_MAX

        )



        deriv = (error - self._prev_err) / dt

        self._prev_err = error



        return float(np.clip(

            self.kp * error + self.ki * self._integral + self.kd * deriv,

            -STEER_MAX_RAD,

            STEER_MAX_RAD

        ))





# ============================================================

# ROS2 SENSOR NODE

# ============================================================



class SensorNode(Node):

    def __init__(self):

        super().__init__("qcar_multisensor_lane_drive")

        qos = QoSProfile(

            reliability=ReliabilityPolicy.BEST_EFFORT,

            history=HistoryPolicy.KEEP_LAST,

            depth=1

        )



        self._lock = threading.Lock()



        self._rgb = None

        self._rgb_ts = None



        self._depth = None

        self._depth_ts = None



        self._lidar_dist = 99.0

        self._lidar_ts = None



        self.create_subscription(Image, "/camera/color_image", self._rgb_cb, qos)

        self.create_subscription(Image, "/camera/depth_image", self._depth_cb, qos)

        self.create_subscription(LaserScan, "/scan", self._lidar_cb, qos)



    @staticmethod

    def _stamp_to_sec(msg):

        try:

            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        except Exception:

            return time.time()



    def _rgb_cb(self, msg):

        if len(msg.data) == 0:

            return



        try:

            channels = 3

            row_w = int(msg.step // channels)

            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, row_w, channels)

            frame = frame[:, :msg.width, :]



            enc = (msg.encoding or "").lower()

            if enc == "rgb8":

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



            with self._lock:

                self._rgb = frame.copy()

                self._rgb_ts = self._stamp_to_sec(msg)

        except Exception:

            pass



    def _depth_cb(self, msg):

        if len(msg.data) == 0:

            return



        try:

            enc = (msg.encoding or "").lower()



            if enc in ("16uc1", "mono16", ""):

                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

                depth = raw.astype(np.float32) * 0.001

            elif enc == "32fc1":

                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

            else:

                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

                depth = raw.astype(np.float32) * 0.001



            with self._lock:

                self._depth = depth.copy()

                self._depth_ts = self._stamp_to_sec(msg)

        except Exception:

            pass



    def _lidar_cb(self, msg):

        try:

            ranges = np.array(msg.ranges, dtype=np.float32)

            angles = np.linspace(

                np.degrees(msg.angle_min),

                np.degrees(msg.angle_max),

                len(ranges)

            )



            mask = (

                (np.abs(angles) < LIDAR_FRONT_DEG) &

                np.isfinite(ranges) &

                (ranges > 0.05) &

                (ranges < 10.0)

            )



            dist = float(np.min(ranges[mask])) if np.any(mask) else 99.0



            with self._lock:

                self._lidar_dist = dist

                self._lidar_ts = self._stamp_to_sec(msg)

        except Exception:

            pass



    def read(self):

        with self._lock:

            rgb = self._rgb.copy() if self._rgb is not None else None

            depth = self._depth.copy() if self._depth is not None else None

            return (

                rgb,

                self._rgb_ts,

                depth,

                self._depth_ts,

                self._lidar_dist,

                self._lidar_ts,

            )





# ============================================================

# UTILITIES

# ============================================================



def hardware_stop(car=None):

    print("[QCar] Stopping hardware...")

    try:

        if car is not None:

            try:

                car.write(0.0, 0.0, np.zeros(8))

                car.__exit__(None, None, None)

            except Exception:

                pass



        if HAS_QCAR:

            my_lidar = QCarLidar()

            my_car = QCar()

            my_car.terminate()

            my_lidar.terminate()

            print("[QCar] LiDAR stopped, QCar stopped")

    except Exception as e:

        print(f"[QCar] Hardware stop error: {e}")





def compute_depth_score(depth_frame, depth_ts, now_sec):

    """

    Depth is NOT pixel-fused with CSI.

    It is used as an independent forward-clearance score in its own frame.



    Returns:

        depth_score   : 0..1

        obstacle      : bool

        fresh         : bool

        age_ms        : float

        valid_ratio   : float

    """

    if depth_frame is None or depth_ts is None:

        return DEPTH_SCORE_STALE, False, False, 9999.0, 0.0



    age_ms = abs(now_sec - depth_ts) * 1000.0

    fresh = age_ms <= DEPTH_STALE_MS



    h, w = depth_frame.shape[:2]

    y1 = int(h * DEPTH_ROI_TOP)

    y2 = int(h * DEPTH_ROI_BOTTOM)

    x1 = int(w * DEPTH_ROI_LEFT)

    x2 = int(w * DEPTH_ROI_RIGHT)



    roi = depth_frame[y1:y2, x1:x2]

    if roi.size == 0:

        return DEPTH_SCORE_STALE, False, fresh, age_ms, 0.0



    valid = (

        np.isfinite(roi) &

        (roi > DEPTH_MIN_VALID_M) &

        (roi < DEPTH_MAX_VALID_M)

    )

    valid_ratio = float(np.sum(valid)) / float(max(roi.size, 1))



    obs = (

        np.isfinite(roi) &

        (roi > DEPTH_MIN_VALID_M) &

        (roi < DEPTH_OBSTACLE_M)

    )

    obstacle_pixels = int(np.sum(obs))

    obstacle = obstacle_pixels > DEPTH_OBS_PIXELS



    if obstacle:

        return 0.0, True, fresh, age_ms, valid_ratio



    if not fresh:

        return DEPTH_SCORE_STALE, False, False, age_ms, valid_ratio



    score = float(np.clip(valid_ratio / 0.35, 0.0, 1.0))

    return score, False, True, age_ms, valid_ratio





def speed_from_confidence(conf, steer_cmd):

    if conf >= CONF_DRIVE:

        base = SPEED_BASE

    elif conf >= CONF_CAUTION:

        base = SPEED_BASE * 0.50

    else:

        return 0.0



    return float(np.clip(

        base * max(0.5, math.cos(abs(steer_cmd))),

        SPEED_MIN,

        SPEED_MAX

    ))





def draw_hud(frame, state, speed, steer, lane_conf, depth_score,
             fusion_conf, lidar_dist, depth_age_ms, valid_ratio, source_name):
    out = frame.copy()
    cv2.rectangle(out, (8, 8), (720, 170), (20, 20, 20), cv2.FILLED)
    out = cv2.addWeighted(out, 0.55, frame, 0.45, 0)

    if state == "DRIVING":
        sc = (0, 255, 120)
    elif state == "CAUTION":
        sc = (0, 200, 255)
    else:
        sc = (0, 0, 255)

    lines = [
        (f"State: {state}", sc),
        (f"Source: {source_name}", (220, 220, 220)),
        (f"Speed: {speed:.3f}   Steer: {steer:+.3f}", (220, 220, 220)),
        (f"LaneConf: {lane_conf:.2f}   Depth: {depth_score:.2f}   Fusion: {fusion_conf:.2f}", (220, 220, 220)),
        (f"LiDAR: {lidar_dist*100:.0f} cm   DepthAge: {depth_age_ms:.0f} ms   DepthValid: {valid_ratio:.0%}", (220, 220, 220)),
    ]

    for i, (txt, col) in enumerate(lines):
        cv2.putText(
            out,
            txt,
            (18, 34 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            col,
            2,
            cv2.LINE_AA
        )

    return out





# ============================================================

# MAIN

# ============================================================



def main():

    args = parse_args()



    # Headless-safe default

    if not args.gui:

        os.environ.pop("DISPLAY", None)

        os.environ.pop("XAUTHORITY", None)

        os.environ["OPENCV_LOG_LEVEL"] = "SILENT"



    stop_dist_m = args.stop / 100.0

    preview = args.preview or not HAS_QCAR



    print("TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW")

    print("Q   QCar2 Multi-Sensor Lane Drive                     Q")

    print("ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]")

    print(f"  Mode            : {'PREVIEW' if preview else 'AUTONOMOUS'}")

    print(f"  GUI             : {'ON' if args.gui else 'OFF (headless-safe)'}")

    print(f"  LiDAR stop      : {stop_dist_m:.2f} m")

    print("  Sensor roles    : CSI=lane  Depth=forward-clearance  LiDAR=hard-stop")

    print()



    launcher = ROSNodeLauncher()



    try:

        if not args.no_launch:

            print("[ROS2] Launching RealSense rgbd...")

            launcher.launch(

                "qcar2_nodes",

                "rgbd",

                env_prefix="LD_PRELOAD=/opt/ros/humble/lib/aarch64-linux-gnu/librealsense2.so.2.54.1 "

            )

            time.sleep(2.0)



            print("[ROS2] Launching LiDAR...")

            launcher.launch("qcar2_nodes", "lidar")

            time.sleep(2.0)



        rclpy.init()

        sensors = SensorNode()

        ros_thread = threading.Thread(target=rclpy.spin, args=(sensors,), daemon=True)

        ros_thread.start()

        print("[ROS2] Subscribers ready")



        detector = LaneDetector()

        pid = PID()



        csi_cameras = None

        try:

            print("[CSI] Initializing front camera (index=2)...")

            csi_cameras = QCarCameras(

                frameWidth=CSI_W,

                frameHeight=CSI_H,

                frameRate=CSI_FPS,

                enableRight=False,

                enableBack=False,

                enableFront=True,

                enableLeft=False,

            )

            print("[CSI] Front camera ready")

        except Exception as e:

            print(f"[CSI] Failed: {e}")



        car = None

        if not preview:

            try:

                car = QCar(readMode=1, frequency=100)

                car.__enter__()

                print("[QCar] Hardware connected")

            except Exception as e:

                print(f"[QCar] Failed: {e} -> switching to preview")

                preview = True



        # Wait for RealSense RGB / depth and CSI

        print("[Wait] Waiting for sensor data...")

        rgb = None

        depth = None

        csi_ok = False



        t_wait = time.time()

        while time.time() - t_wait < 8.0:

            rgb, rgb_ts, depth, depth_ts, lidar_dist, lidar_ts = sensors.read()



            if csi_cameras is not None:

                try:

                    csi_cameras.readAll()

                    img = csi_cameras.csi[2].imageData

                    if img is not None and img.max() > 10:

                        csi_ok = True

                except Exception:

                    pass



            if depth is not None and (rgb is not None or csi_ok):

                break



            time.sleep(0.1)



        print(f"[Wait] CSI ready   : {'YES' if csi_ok else 'NO'}")

        print(f"[Wait] RGB ready   : {'YES' if rgb is not None else 'NO'}")

        print(f"[Wait] Depth ready : {'YES' if depth is not None else 'NO'}")



        if args.gui:

            cv2.namedWindow("QCar Multi-Sensor Lane Drive", cv2.WINDOW_NORMAL)

            cv2.resizeWindow("QCar Multi-Sensor Lane Drive", 1280, 720)



        frame_count = 0

        state = "WAITING"

        stopped_by_lidar = False



        while True:

            t0 = time.time()

            frame_count += 1



            rgb, rgb_ts, depth_frame, depth_ts, lidar_dist, lidar_ts = sensors.read()



            # PRIMARY CAMERA = CSI front

            csi_frame = None

            if csi_cameras is not None:

                try:

                    csi_cameras.readAll()

                    img = csi_cameras.csi[2].imageData

                    if img is not None and img.max() > 10:

                        csi_frame = img.copy()

                except Exception:

                    csi_frame = None



            source_name = "CSI"

            drive_frame = csi_frame



            # fallback only if CSI is not available

            if drive_frame is None and rgb is not None:

                drive_frame = rgb

                source_name = "RealSenseRGB"



            if drive_frame is None:

                time.sleep(0.01)

                continue



            # Lane detection on ONE primary visual source

            # IMPORTANT: do not pass RealSense depth into lane detector when using CSI.

            overlay = detector.process(drive_frame, None)

            offset = float(getattr(detector, "center_offset_m", 0.0))

            lane_conf = float(np.clip(getattr(detector, "confidence", 0.0), 0.0, 1.0))

            no_lane = int(getattr(detector, "no_lane_count", 0))



            # Depth score in native depth frame (not fake pixel-fused to CSI)

            now_sec = time.time()

            (
    depth_score,
    depth_obstacle,
    depth_fresh,
    depth_age_ms,
    valid_ratio,
) = compute_depth_score(depth_frame, depth_ts, now_sec)



            fusion_conf = float(np.clip(lane_conf * depth_score, 0.0, 1.0))



            # LiDAR hysteresis

            if lidar_dist <= stop_dist_m:

                stopped_by_lidar = True

            elif lidar_dist >= max(LIDAR_RESUME_DISTANCE_M, stop_dist_m + 0.10):

                stopped_by_lidar = False



            speed_cmd = 0.0

            steer_cmd = 0.0



            # State logic

            if stopped_by_lidar:

                state = "STOP-LIDAR"

                pid.reset()



            elif depth_obstacle:

                state = "STOP-DEPTH"

                pid.reset()



            elif no_lane >= NO_LANE_STOP_FRAMES or lane_conf <= 0.05:

                state = "STOP-NOLANE"

                pid.reset()



            else:

                # steering from lane offset

                steer_cmd = pid.update(-offset / 0.008)



                if fusion_conf >= CONF_DRIVE:

                    state = "DRIVING"

                    speed_cmd = speed_from_confidence(fusion_conf, steer_cmd)

                elif fusion_conf >= CONF_CAUTION:

                    state = "CAUTION"

                    speed_cmd = speed_from_confidence(fusion_conf, steer_cmd)

                else:

                    state = "STOP-LOWCONF"

                    speed_cmd = 0.0

                    steer_cmd = 0.0

                    pid.reset()



            leds = LEDS_DEFAULT.copy()

            if steer_cmd > 0.05:

                leds[0] = leds[2] = 1

            if steer_cmd < -0.05:

                leds[1] = leds[3] = 1

            if state.startswith("STOP"):

                leds[4] = leds[5] = 1

            if state == "CAUTION":

                leds[6] = leds[7] = 1



            if car is not None and not preview:

                try:

                    car.write(speed_cmd, steer_cmd, leds)

                except Exception as e:

                    print(f"[QCar] Write error: {e}")



            if frame_count % LOG_EVERY_N == 0:

                print(

                    f"[{state}] src={source_name:<12} "

                    f"lane={lane_conf:.2f} depth={depth_score:.2f} fusion={fusion_conf:.2f} "

                    f"offset={offset:+.4f} steer={steer_cmd:+.3f} speed={speed_cmd:.3f} "

                    f"lidar={lidar_dist:.2f}m depthAge={depth_age_ms:.0f}ms valid={valid_ratio:.0%}"

                )



            if args.gui:

                disp = draw_hud(

                    overlay, state, speed_cmd, steer_cmd, lane_conf, depth_score,

                    fusion_conf, lidar_dist, depth_age_ms, valid_ratio, source_name

                )

                cv2.imshow("QCar Multi-Sensor Lane Drive", disp)

                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q"):

                    break



            sleep_t = (1.0 / CAMERA_FPS) - (time.time() - t0)

            if sleep_t > 0:

                time.sleep(sleep_t)



    except KeyboardInterrupt:

        print("\n[Stopped] Ctrl+C")



    finally:

        print("\n[Shutdown] Cleaning up...")

        try:

            hardware_stop(car)

        except Exception:

            pass



        try:

            if csi_cameras is not None:

                csi_cameras.terminate()

        except Exception:

            pass



        try:

            sensors.destroy_node()

            rclpy.shutdown()

        except Exception:

            pass



        if args.gui:

            try:

                cv2.destroyAllWindows()

            except Exception:

                pass



        if not args.no_launch:

            launcher.stop_all()



        print("[Done] ")





if __name__ == "__main__":

    main()
