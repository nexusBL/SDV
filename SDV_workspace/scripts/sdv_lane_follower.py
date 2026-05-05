#!/usr/bin/env python3
"""
sdv_lane_follower.py — QCar2 Industry-Level Autonomous Lane Follower
ONE file. ONE command.

Controls: a=start  s=stop  q=quit
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
import cv2

# ── Save DISPLAY before stripping for CSI ────────────────────────────────────
_SAVED_DISPLAY    = os.environ.get("DISPLAY", ":1")
_SAVED_XAUTHORITY = os.environ.get("XAUTHORITY", "")

# Strip for CSI EGL headless init
os.environ.pop("DISPLAY",    None)
os.environ.pop("XAUTHORITY", None)

sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

try:
    from pal.products.qcar import QCar, QCarLidar, QCarCameras
    HAS_QCAR = True
except ImportError:
    HAS_QCAR = False
    print("[SDV] WARNING: QCar PAL not found — preview mode only.")

try:
    from pal.utilities.lidar import Lidar as QuanserLidar
    HAS_LIDAR = True
except ImportError:
    HAS_LIDAR = False

# ── Restore DISPLAY for OpenCV ────────────────────────────────────────────────
os.environ["DISPLAY"] = _SAVED_DISPLAY
if _SAVED_XAUTHORITY:
    os.environ["XAUTHORITY"] = _SAVED_XAUTHORITY


# ══════════════════════════════════════════════════════════════════════════════
#  CSI CAMERA
# ══════════════════════════════════════════════════════════════════════════════

class CSICamera:
    def __init__(self, width=820, height=616, fps=30):
        self._cam    = None
        self._active = False
        if not HAS_QCAR:
            return
        try:
            # Strip display ONLY for camera init
            os.environ.pop("DISPLAY",    None)
            os.environ.pop("XAUTHORITY", None)

            self._cam = QCarCameras(
                frameWidth=width,
                frameHeight=height,
                frameRate=fps,
                enableFront=True,
                enableRight=False,
                enableBack=False,
                enableLeft=False,
            )
            self._active = True
            print("[Camera] CSI front camera initialized ✅")
        except Exception as e:
            print(f"[Camera] Init failed: {e}")
        finally:
            # Always restore display
            os.environ["DISPLAY"] = _SAVED_DISPLAY
            if _SAVED_XAUTHORITY:
                os.environ["XAUTHORITY"] = _SAVED_XAUTHORITY

    def read(self):
        if not self._active or self._cam is None:
            return None
        try:
            self._cam.readAll()
            img = self._cam.csi[2].imageData
            if img is not None and img.max() > 5:
                return img.copy()
        except Exception as e:
            print(f"[Camera] Read error: {e}")
        return None

    def terminate(self):
        if self._cam is not None:
            try:
                self._cam.terminate()
                print("[Camera] Terminated ✅")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  LANE DETECTOR — RANSAC + Adaptive HSV + Sliding Window
# ══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self, frame_w=820, frame_h=616):
        w = frame_w // 2   # after 0.5x resize = 410
        h = frame_h // 2   #                    = 308

        # Perspective transform points — recalibrate on your track!
        self.Source = np.float32([
            [int(w*0.25), int(h*0.60)],
            [int(w*0.75), int(h*0.60)],
            [int(w*0.00), int(h*1.00)],
            [int(w*1.00), int(h*1.00)],
        ])
        self.Destination = np.float32([
            [int(w*0.25), 0],
            [int(w*0.75), 0],
            [int(w*0.25), h],
            [int(w*0.75), h],
        ])
        self._M     = cv2.getPerspectiveTransform(self.Source, self.Destination)
        self._M_inv = cv2.getPerspectiveTransform(self.Destination, self.Source)

        # Adaptive HSV
        self.min_hue = 10;  self.max_hue = 45
        self.min_sat = 50;  self.max_sat = 255
        self.min_val = 100; self.max_val = 255

        # RANSAC
        self.ransac_min_samples        = 10
        self.ransac_residual_threshold = 2.0
        self.ransac_max_trials         = 100

        # Polynomial state
        self.polyleft       = [0.0, 0.0, 0.0]
        self.polyright      = [0.0, 0.0, 0.0]
        self.polyleft_last  = [0.0, 0.0, 0.0]
        self.polyright_last = [0.0, 0.0, 0.0]
        self.left_points    = []
        self.right_points   = []

        # Error smoothing
        self.error         = 0.0
        self.error_history = []
        self.max_history   = 5
        self.process_time  = 0.0
        self._left_base    = None
        self._right_base   = None

    def _adaptive_hsv(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b    = np.mean(gray)
        if b < 80:
            self.min_val = max(50,  self.min_val - 10)
            self.min_sat = max(30,  self.min_sat - 10)
        elif b > 180:
            self.min_val = min(150, self.min_val + 10)
            self.min_sat = min(80,  self.min_sat + 10)
        return (np.array([self.min_hue, self.min_sat, self.min_val]),
                np.array([self.max_hue, self.max_sat, self.max_val]))

    def _transform(self, frame):
        small  = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        orig   = small.copy()
        bev    = cv2.warpPerspective(small, self._M, (small.shape[1], small.shape[0]))
        hsv    = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        lo, hi = self._adaptive_hsv(bev)
        binary = cv2.inRange(hsv, lo, hi)
        k      = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
        return orig, bev, binary

    def _histogram(self, binary):
        h, w  = binary.shape
        roi   = binary[h//2:, :]
        hist  = np.sum(roi, axis=0)
        mid   = w // 2
        lp    = int(np.argmax(hist[:mid]))
        rp    = int(np.argmax(hist[mid:])) + mid
        if hist[lp] < 50: lp = -1
        if hist[rp] < 50: rp = -1
        return lp, rp

    def _sliding_window(self, binary):
        self.left_points  = []
        self.right_points = []
        h, w = binary.shape

        lp, rp = self._histogram(binary)
        lx = lp if lp != -1 else (self._left_base  or w//4)
        rx = rp if rp != -1 else (self._right_base or w*3//4)
        if lp != -1: self._left_base  = lp
        if rp != -1: self._right_base = rp

        nwindows   = 10
        margin     = 40
        minpix     = 30
        win_h      = h // nwindows
        nz         = binary.nonzero()
        nzy        = np.array(nz[0])
        nzx        = np.array(nz[1])

        for win in range(nwindows):
            y_lo = h - (win+1)*win_h
            y_hi = h - win*win_h
            xl_lo = max(0, lx-margin); xl_hi = min(w-1, lx+margin)
            xr_lo = max(0, rx-margin); xr_hi = min(w-1, rx+margin)

            l_idx = ((nzy>=y_lo)&(nzy<y_hi)&(nzx>=xl_lo)&(nzx<xl_hi)).nonzero()[0]
            r_idx = ((nzy>=y_lo)&(nzy<y_hi)&(nzx>=xr_lo)&(nzx<xr_hi)).nonzero()[0]

            for i in l_idx: self.left_points.append((nzy[i], nzx[i]))
            for i in r_idx: self.right_points.append((nzy[i], nzx[i]))
            if len(l_idx) > minpix: lx = int(np.mean(nzx[l_idx]))
            if len(r_idx) > minpix: rx = int(np.mean(nzx[r_idx]))

    def _ransac_fit(self, points):
        if len(points) < self.ransac_min_samples:
            return None, False
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        best_c = None; best_n = 0
        for _ in range(self.ransac_max_trials):
            idx = np.random.choice(len(points), self.ransac_min_samples, replace=False)
            try:
                c    = np.polyfit(xs[idx], ys[idx], 2)
                errs = np.abs(ys - np.polyval(c, xs))
                n    = int(np.sum(errs < self.ransac_residual_threshold))
                if n > best_n:
                    best_n = n; best_c = c
                    if n > len(points)*0.8: break
            except Exception:
                continue
        if best_c is not None and best_n >= self.ransac_min_samples:
            errs = np.abs(ys - np.polyval(best_c, xs))
            inl  = errs < self.ransac_residual_threshold
            if int(np.sum(inl)) >= self.ransac_min_samples:
                try:
                    return np.polyfit(xs[inl], ys[inl], 2).tolist(), True
                except Exception:
                    pass
            return best_c.tolist(), True
        try:
            return np.polyfit(xs, ys, 2).tolist(), True
        except Exception:
            return None, False

    def _draw_and_calc(self, bev, orig):
        h, w       = bev.shape[:2]
        center_cam = w//2 + 22

        ok_l, ok_r = False, False
        cl, ok_l   = self._ransac_fit(self.left_points)
        cr, ok_r   = self._ransac_fit(self.right_points)
        self.left_points  = []
        self.right_points = []

        if ok_l: self.polyleft      = cl; self.polyleft_last  = cl
        if ok_r: self.polyright     = cr; self.polyright_last = cr

        pl = self.polyleft  if ok_l else self.polyleft_last
        pr = self.polyright if ok_r else self.polyright_last

        overlay = np.zeros_like(bev)
        cl_col  = (0,255,255) if ok_l else (0,0,255)
        cr_col  = (0,255,255) if ok_r else (0,0,255)

        # Draw lane lines + fill
        pts_l, pts_r = [], []
        for row in range(h-1, -1, -8):
            xl = int(np.clip(pl[2]+pl[1]*row+pl[0]*row*row, 0, w-1))
            xr = int(np.clip(pr[2]+pr[1]*row+pr[0]*row*row, 0, w-1))
            cv2.circle(overlay, (xl, row), 3, cl_col, -1)
            cv2.circle(overlay, (xr, row), 3, cr_col, -1)
            pts_l.append([xl, row])
            pts_r.append([xr, row])

        # Green fill between lanes
        if len(pts_l) >= 2 and len(pts_r) >= 2:
            pts_all = pts_l + pts_r[::-1]
            cv2.fillPoly(overlay, [np.array(pts_all, dtype=np.int32)], (0,180,0))

        # Ego line (center)
        for row in range(0, h, 8):
            xl = pl[2]+pl[1]*row+pl[0]*row*row
            xr = pr[2]+pr[1]*row+pr[0]*row*row
            xc = int(np.clip((xl+xr)/2, 0, w-1))
            cv2.circle(overlay, (xc, row), 3, (0,255,255), -1)

        # Center at bottom
        xl_b = pl[2]+pl[1]*(h-1)+pl[0]*(h-1)*(h-1)
        xr_b = pr[2]+pr[1]*(h-1)+pr[0]*(h-1)*(h-1)

        if ok_l and ok_r:
            center_lines = (xl_b + xr_b) / 2
        elif ok_l:
            center_lines = float(pl[2]) + 125
        elif ok_r:
            center_lines = float(pr[2]) - 125
        else:
            center_lines = (xl_b + xr_b) / 2

        center_lines = float(np.clip(center_lines, 0, w-1))

        # Error + smoothing
        raw = float(center_cam - center_lines)
        self.error_history.append(raw)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        self.error = float(np.mean(self.error_history))

        ne = self.error / (w/2)
        if abs(ne) < 0.1:  direction = "CENTER"
        elif ne > 0:        direction = "STEER LEFT"
        else:               direction = "STEER RIGHT"

        # Unwarp overlay back to camera view
        unwarped = cv2.warpPerspective(overlay, self._M_inv,
                                       (orig.shape[1], orig.shape[0]))
        result   = cv2.addWeighted(orig, 0.7, unwarped, 0.3, 0)

        # Reference lines
        cv2.line(result, (int(center_cam), int(h*0.25)),
                         (int(center_cam), int(h*0.75)), (0,255,0), 2)
        cv2.line(result, (int(center_lines), 0),
                         (int(center_cam),   h-1), (0,0,255), 2)

        cv2.putText(result, f"Dir: {direction}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(result, f"Error: {self.error:.2f}",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(result, f"T: {self.process_time*1000:.1f}ms",
                    (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return result

    def process(self, frame):
        t0 = time.time()
        orig, bev, binary = self._transform(frame)
        self._sliding_window(binary)
        result = self._draw_and_calc(bev, orig)
        self.process_time = time.time() - t0
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  PID CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class PIDController:
    def __init__(self, kp=0.00225, ki=0.00015, kd=0.00075,
                 setpoint=-22.45, max_integral=100.0, steer_limit=0.6):
        self.kp=kp; self.ki=ki; self.kd=kd
        self.setpoint=setpoint; self.integral=0.0
        self.prev_error=0.0; self.max_integral=max_integral
        self.steer_limit=steer_limit

    def compute(self, error):
        e         = self.setpoint + error
        p         = self.kp * e
        self.integral = np.clip(self.integral+e, -self.max_integral, self.max_integral)
        i         = self.ki * self.integral
        d         = self.kd * (e - self.prev_error)
        self.prev_error = e
        return float(np.clip(p+i+d, -self.steer_limit, self.steer_limit))

    def reset(self):
        self.integral=0.0; self.prev_error=0.0


# ══════════════════════════════════════════════════════════════════════════════
#  LIDAR PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class LidarProcessor:
    def __init__(self, max_distance=1.5):
        self.max_distance = max_distance
        self.side_len     = int(8 * 50)
        self.decay        = 0.9
        self.map          = np.zeros((self.side_len, self.side_len), dtype=np.float32)
        self._lidar       = None
        self._active      = False

        if HAS_LIDAR:
            try:
                self._lidar = QuanserLidar(
                    type='RPLidar',
                    numMeasurements=360,
                    rangingDistanceMode=2,
                    interpolationMode=0
                )
                self._active = True
                print("[LiDAR] Initialized ✅")
            except Exception as e:
                print(f"[LiDAR] Init failed: {e} — no obstacle detection.")

    def detect_obstacle(self):
        if not self._active or self._lidar is None:
            return False
        try:
            self.map *= self.decay
            self._lidar.read()
            angles = self._lidar.angles * -1 + np.pi/2
            dists  = self._lidar.distances
            idx    = [i for i,d in enumerate(dists) if d < self.max_distance]
            if not idx:
                return False
            x  = dists[idx] * np.cos(angles[idx])
            y  = dists[idx] * np.sin(angles[idx])
            pX = np.clip((self.side_len/2 - x*50).astype(int), 0, self.side_len-1)
            pY = np.clip((self.side_len/2 - y*50).astype(int), 0, self.side_len-1)
            filtered = [(px,py) for px,py in zip(pX,pY) if 190<=py<=210 and px<=190]
            if filtered:
                fpx,fpy = zip(*filtered)
                self.map[list(fpx), list(fpy)] = 1.0
                Y = [py for py in fpy if py > 165]
                return len(Y) > 10
        except Exception as e:
            print(f"[LiDAR] Read error: {e}")
        return False

    def terminate(self):
        if self._lidar is not None:
            try:
                self._lidar.terminate()
                print("[LiDAR] PAL Lidar terminated ✅")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER
# ══════════════════════════════════════════════════════════════════════════════

class KeyListener:
    def __init__(self):
        self.last_key    = None
        self.should_exit = False

    def _getch(self):
        import termios, tty
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def _listen(self):
        while not self.should_exit:
            try:
                ch = self._getch()
                self.last_key = ch
                if ch == 'q':
                    self.should_exit = True
            except Exception:
                time.sleep(0.1)

    def start(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def consume(self):
        k = self.last_key; self.last_key = None; return k


# ══════════════════════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, driving, throttle, steering, error,
             obstacle, resume_counter, fps):
    ov = frame.copy()
    cv2.rectangle(ov, (5,5), (500,215), (15,15,15), cv2.FILLED)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    if obstacle:         state="OBSTACLE"; sc=(0,165,255)
    elif driving:        state="DRIVING";  sc=(0,255,0)
    else:                state="STOPPED";  sc=(0,0,255)

    lines = [
        (f"State   : {state}   FPS: {fps}", sc),
        (f"Throttle: {throttle:.3f}   Steer: {steering:+.3f}", (220,220,220)),
        (f"Error   : {error:.2f} px", (220,220,220)),
        (f"Obstacle: {'YES - STOPPED' if obstacle else 'none'}", (0,0,255) if obstacle else (0,200,0)),
        (f"Resume in: {resume_counter} frames" if resume_counter>0 else "a=start  s=stop  q=quit", (150,150,150)),
    ]
    for i,(txt,col) in enumerate(lines):
        cv2.putText(frame, txt, (15,30+i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE STOP — Quanser PAL way (like QCar2_hardware_stop.py)
# ══════════════════════════════════════════════════════════════════════════════

def hardware_stop(car=None):
    print("[Shutdown] Stopping hardware via Quanser PAL...")
    if car:
        try:
            car.write(0.0, 0.0, np.zeros(8))
            car.__exit__(None, None, None)
            print("[QCar2] Stopped ✅")
        except Exception:
            pass
    try:
        from pal.products.qcar import QCar as _QCar, QCarLidar as _QL
        _QL().terminate()
        _QCar().terminate()
        print("[QCar2] QCarLidar + QCar terminated via PAL ✅")
    except Exception as e:
        print(f"[Hardware] PAL stop: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', action='store_true')
    args    = p.parse_args()
    preview = args.preview or not HAS_QCAR

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  QCar2 SDV — Industry-Level Lane Follower               ║")
    print("║  RANSAC + Adaptive HSV + PID + LiDAR + CSI Camera       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Mode   : {'PREVIEW (no motors)' if preview else 'AUTONOMOUS'}")
    print(f"  Display: {_SAVED_DISPLAY}")
    print()

    camera   = CSICamera(width=820, height=616, fps=30)
    detector = LaneDetector(frame_w=820, frame_h=616)
    pid      = PIDController()
    lidar    = LidarProcessor()
    keys     = KeyListener()
    keys.start()

    car = None
    if not preview:
        try:
            car = QCar(readMode=1, frequency=100)
            car.__enter__()
            print("[QCar2] Hardware connected ✅")
        except Exception as e:
            print(f"[QCar2] Failed: {e} — preview mode.")
            preview = True

    # Wait for first camera frame
    print("[Camera] Waiting for first frame...")
    frame = None
    for _ in range(50):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        print("[ERROR] No camera frames! Exiting.")
        hardware_stop(car)
        return
    print("[Camera] Ready ✅")

    # OpenCV window
    cv2.namedWindow("QCar2 Lane Follower", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar2 Lane Follower", 820, 616)

    print("\nPress 'a' to START, 's' to STOP, 'q' to QUIT\n")

    driving_active    = False
    obstacle_detected = False
    resume_counter    = 0
    RESUME_TIME       = 5
    throttle_cmd      = 0.0
    steering_cmd      = 0.0
    fps               = 0
    frame_count       = 0
    fps_timer         = time.time()

    try:
        while not keys.should_exit:
            t0    = time.time()
            frame = camera.read()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count; frame_count=0; fps_timer=now

            # Lane detection
            display = detector.process(frame)

            # Keyboard
            key = keys.consume()
            if key == 'a' and not driving_active:
                driving_active = True
                pid.reset()
                print("[SDV] Lane following STARTED! 🚗")
            elif key == 's' and driving_active:
                driving_active = False
                throttle_cmd   = 0.0
                steering_cmd   = 0.0
                print("[SDV] Lane following STOPPED!")

            # Obstacle detection
            cur_obs = lidar.detect_obstacle()
            if cur_obs != obstacle_detected:
                obstacle_detected = cur_obs
                if obstacle_detected:
                    print("[SDV] 🛑 OBSTACLE! Stopping.")
                    resume_counter = RESUME_TIME
                else:
                    print("[SDV] ✅ Path clear.")
            if not obstacle_detected and resume_counter > 0:
                resume_counter -= 1

            # Control
            if driving_active:
                steering_cmd = pid.compute(detector.error)
                throttle_cmd = 0.0 if (obstacle_detected or resume_counter>0) else 0.08
            else:
                throttle_cmd = 0.0
                steering_cmd = 0.0

            # LEDs
            if driving_active:
                leds = np.array([1,0,1,0,1,0,1,0] if obstacle_detected
                                else [0,0,0,0,1,1,0,0], dtype=np.float64)
            else:
                leds = np.zeros(8, dtype=np.float64)

            if car and not preview:
                try:
                    car.write(throttle_cmd, steering_cmd, leds)
                except Exception as e:
                    print(f"[QCar2] Write error: {e}")

            display = draw_hud(display, driving_active, throttle_cmd, steering_cmd,
                               detector.error, obstacle_detected, resume_counter, fps)

            cv2.imshow("QCar2 Lane Follower", display)
            cv2.waitKey(1)

            elapsed = time.time() - t0
            wait    = (1.0/30.0) - elapsed
            if wait > 0: time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        print("\n[Shutdown] Stopping everything...")
        hardware_stop(car)
        lidar.terminate()
        camera.terminate()
        keys.should_exit = True
        try: cv2.destroyAllWindows()
        except Exception: pass
        print("[Done] ✅")


if __name__ == "__main__":
    main()
