#!/usr/bin/env python3
"""
QCar2 AI Sensor Fusion — CSI Front Camera + RealSense RGB
Stream what the car sees at:  http://<qcar-ip>:8080
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Fix EGL / display issues on headless Orin ──────────────────────────────
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

try:
    from pal.products.qcar import QCarCameras
except ImportError:
    print("ERROR: Could not import QCarCameras"); exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: pip3 install ultralytics"); exit(1)


# ── Human-friendly class info ───────────────────────────────────────────────
CLASS_INFO = {
    0:  {"name": "Stop Sign",           "color": (0,   0,   220), "tip": "STOP the vehicle!"},
    1:  {"name": "Turn Left Ahead",     "color": (220, 150, 0),   "tip": "Left turn coming up"},
    2:  {"name": "Turn Right Ahead",    "color": (220, 150, 0),   "tip": "Right turn coming up"},
    3:  {"name": "Straight Ahead",      "color": (0,   200, 50),  "tip": "Keep going straight"},
    4:  {"name": "Give Way",            "color": (0,   165, 255), "tip": "Yield to other traffic"},
    5:  {"name": "No Entry",            "color": (0,   0,   200), "tip": "Do NOT enter!"},
    6:  {"name": "Keep Left",           "color": (255, 200, 0),   "tip": "Stay on the left side"},
    7:  {"name": "Keep Right",          "color": (255, 200, 0),   "tip": "Stay on the right side"},
    8:  {"name": "Round About",         "color": (180, 0,   180), "tip": "Enter the roundabout"},
    9:  {"name": "Pedestrian Crossing", "color": (0,   220, 220), "tip": "Watch for pedestrians!"},
    10: {"name": "Traffic Signal",      "color": (0,   180, 255), "tip": "Obey the traffic light"},
    11: {"name": "U Turn",              "color": (100, 100, 255), "tip": "U-turn permitted"},
    12: {"name": "No U Turn",           "color": (0,   0,   200), "tip": "No U-turn allowed"},
}
VALID_CLASS_IDS = set(CLASS_INFO.keys())  # 0-12 only — ignores garbage like class 258

# ── Global frame buffer shared with MJPEG server ────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()


# ── MJPEG HTTP Server ────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence request logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html><head>
            <title>QCar2 AI Vision</title>
            <style>
              body { background:#111; color:#eee; font-family:sans-serif;
                     display:flex; flex-direction:column; align-items:center;
                     justify-content:center; height:100vh; margin:0; }
              h2   { margin-bottom:10px; letter-spacing:2px; color:#0f0; }
              img  { border:2px solid #0f0; border-radius:8px; max-width:95vw; }
              p    { color:#888; font-size:12px; margin-top:10px; }
            </style>
            </head><body>
            <h2>QCar2 - Live AI Vision</h2>
            <img src="/stream" /><br>
            <p>Open this page on any device connected to the same network.</p>
            </body></html>""")

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    _, jpeg = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
                    data = jpeg.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


def start_stream_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n  Open in laptop browser ->  http://<qcar-ip>:{port}")
    print(  "  Find QCar IP with:         hostname -I\n")


# ── Draw detections with human-friendly overlay ──────────────────────────────
def draw_detections(frame, result, sensor_name, fps):
    out = frame.copy()
    h, w = out.shape[:2]

    # top status bar
    cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.putText(out,
                f"QCar2 AI Vision  |  {sensor_name}  |  {fps:.1f} FPS",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)

    detected = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VALID_CLASS_IDS:
            continue  # ignore invalid class IDs like 258

        conf  = float(box.conf[0])
        info  = CLASS_INFO[cls_id]
        color = info["color"]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # label pill
        tag = f"{info['name']}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        py = max(y1 - th - 10, 0)
        cv2.rectangle(out, (x1, py), (x1 + tw + 10, y1), color, -1)
        cv2.putText(out, tag, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detected.append((conf, info["name"], info["tip"], color))

    # bottom info panel — "what the car sees"
    detected.sort(key=lambda x: x[0], reverse=True)
    panel_h = 42 + max(len(detected), 1) * 34
    py0     = h - panel_h - 8
    cv2.rectangle(out, (8, py0), (500, h - 8), (20, 20, 20), -1)
    cv2.rectangle(out, (8, py0), (500, h - 8), (60, 60, 60),  1)
    cv2.putText(out, "WHAT THE CAR SEES RIGHT NOW:",
                (18, py0 + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1)

    if detected:
        for i, (conf, name, tip, color) in enumerate(detected):
            y = py0 + 52 + i * 34
            # confidence bar
            cv2.rectangle(out, (18, y - 16), (18 + int(conf * 170), y), color, -1)
            cv2.putText(out, f"{name}: {tip}  ({conf*100:.0f}%)",
                        (20, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
    else:
        cv2.putText(out, "No road signs detected",
                    (18, py0 + 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 120, 120), 1)

    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global latest_frame

    print("Loading AI Model...")
    try:
        model = YOLO("BEST_MODEL_QCAR2.pt")
        print(f"  Model loaded on NVIDIA Orin GPU!")
        print(f"  Detects: {', '.join(CLASS_INFO[i]['name'] for i in range(13))}")
    except Exception as e:
        print(f"Failed to load model: {e}"); return

    print("\nInitializing cameras...")

    # ── CSI Front Camera ──────────────────────────────────────────────────
    csi_cam = None
    try:
        csi_cam = QCarCameras(
            frameWidth=820, frameHeight=616, frameRate=80,
            enableRight=False, enableBack=False,
            enableFront=True,  enableLeft=False
        )
        print("  [CSI Front Camera]  /dev/video2  ✅")
    except Exception as e:
        print(f"  [CSI Front Camera]  ❌  {e}")

    # ── RealSense RGB — starts at /dev/video4 on this QCar ────────────────
    rs_cam = None
    for dev_id in [4, 5, 6, 7, 8, 9]:
        cap = cv2.VideoCapture(dev_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, test = cap.read()
            if ret and test is not None:
                rs_cam = cap
                print(f"  [RealSense Camera]  /dev/video{dev_id}  ✅")
                break
            cap.release()
    if rs_cam is None:
        print("  [RealSense Camera]  ⚠️  not streaming — running CSI only")

    if csi_cam is None and rs_cam is None:
        print("ERROR: No cameras available."); return

    start_stream_server(port=8080)
    print("SENSOR FUSION RUNNING. Press CTRL+C to stop.\n")

    fps    = 0.0
    t_prev = time.time()
    n      = 0

    try:
        while True:
            frames, labels = [], []

            # CSI
            if csi_cam is not None:
                try:
                    csi_cam.readAll()
                    f = csi_cam.csi[2].imageData
                    if f is not None and f.max() > 10:
                        frames.append(f)
                        labels.append("CSI Front Camera")
                except Exception as e:
                    print(f"[CSI] {e}")

            # RealSense
            if rs_cam is not None:
                ret, f = rs_cam.read()
                if ret and f is not None:
                    frames.append(f)
                    labels.append("RealSense Camera")

            if not frames:
                time.sleep(0.01)
                continue

            # inference
            results = model.predict(source=frames, conf=0.45, verbose=False)

            # pick best frame
            best_conf, best_idx = 0.0, 0
            for i, r in enumerate(results):
                valid = [float(b.conf[0]) for b in r.boxes
                         if int(b.cls[0]) in VALID_CLASS_IDS]
                if valid and max(valid) > best_conf:
                    best_conf = max(valid)
                    best_idx  = i

            # FPS
            n += 1
            if n % 15 == 0:
                fps    = 15.0 / (time.time() - t_prev)
                t_prev = time.time()

            display = draw_detections(frames[best_idx], results[best_idx],
                                      labels[best_idx], fps)

            with frame_lock:
                latest_frame = display

            # terminal log
            if best_conf > 0:
                top = max(
                    [(float(b.conf[0]), CLASS_INFO[int(b.cls[0])]["name"])
                     for b in results[best_idx].boxes
                     if int(b.cls[0]) in VALID_CLASS_IDS],
                    key=lambda x: x[0]
                )
                print(f"[{labels[best_idx]}]  {top[1]}  {top[0]*100:.1f}%  |  {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if csi_cam:
            try: csi_cam.terminate()
            except: pass
        if rs_cam:
            try: rs_cam.release()
            except: pass
        print("Done.")


if __name__ == '__main__':
    main()
