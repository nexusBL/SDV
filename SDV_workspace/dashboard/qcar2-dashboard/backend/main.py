import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import math
import random
import time
import os
import cv2
import numpy as np
import threading
import sys

# Quanser path setup - pop DISPLAY for headless operation
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
QUANSER_LIB_PATH = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.exists(QUANSER_LIB_PATH) and QUANSER_LIB_PATH not in sys.path:
    sys.path.insert(0, QUANSER_LIB_PATH)

class HardwareManager:
    def __init__(self):
        self.has_hardware = False
        self.qcar = None
        self.cameras = None
        self.realsense = None
        self.latest_frames = [None] * 4
        self.latest_realsense = None
        self.telemetry = {
            "battery": 100.0,
            "steering_angle": 0.0,
            "speed": 0.0,
            "mode": "SIMULATION"
        }
        self.latest_lidar = {"distances": [], "angles": []}
        self.active = True
        
        try:
            print("🔍 Attempting to import Quanser libraries...")
            from pal.products.qcar import QCar, QCarCameras, QCarRealSense, QCarLidar
            print("🔍 Initializing QCar...")
            self.qcar = QCar()
            print("🔍 Initializing Cameras...")
            self.cameras = QCarCameras(
                frameWidth=640, frameHeight=480, frameRate=30,
                enableFront=True, enableLeft=True, enableRight=True, enableBack=True
            )
            print("🔍 Initializing RealSense...")
            self.realsense = QCarRealSense(mode='RGB', frameWidthRGB=640, frameHeightRGB=480)
            print("🔍 Initializing LiDAR...")
            self.lidar = QCarLidar(numMeasurements=720, rangingDistanceMode=2, interpolationMode=0)
            
            self.has_hardware = True
            self.telemetry["mode"] = "REAL"
            print("✅ QCar 2 Hardware Initialized Successfully. Waiting 1s for sensors...")
            time.sleep(1.0)
        except Exception as e:
            print(f"⚠️ Hardware not found or error: {e}. Initializing in Simulation Mode.")

    def update_loop(self):
        while self.active:
            if self.has_hardware:
                try:
                    # Sync hardware state
                    self.cameras.readAll()
                    for i in range(4):
                        if self.cameras.csi[i] is not None:
                            self.latest_frames[i] = self.cameras.csi[i].imageData
                    
                    if self.realsense:
                        self.realsense.read_RGB()
                        self.latest_realsense = self.realsense.imageBufferRGB
                    
                    if self.lidar:
                        self.lidar.read()
                        if len(self.lidar.distances) > 0:
                            dists = self.lidar.distances.tolist()
                            dists = [d if not (math.isinf(d) or math.isnan(d)) else 0.0 for d in dists]
                            self.latest_lidar = {
                                "distances": dists,
                                "angles": self.lidar.angles.tolist()
                            }
                    
                    self.telemetry["battery"] -= 0.0001 
                except Exception as e:
                    print(f"❌ Hardware loop error: {e}")
            else:
                # Simulation Mode Telemetry
                t = time.time()
                self.telemetry["battery"] = max(0.0, 100.0 - (t % 1000) / 100.0)
                self.telemetry["steering_angle"] = math.sin(t) * 25.0
                self.telemetry["speed"] = 1.2 + math.sin(t / 2) * 0.5
            
            time.sleep(0.04) # ~25Hz

    def stop(self):
        self.active = False
        time.sleep(0.1)
        try:
            if self.cameras: self.cameras.terminate()
            if self.realsense: self.realsense.terminate()
            if hasattr(self, 'lidar') and self.lidar: self.lidar.terminate()
            if self.qcar: self.qcar.terminate()
            print("🛑 Hardware components standard termination successful. LiDAR motor should be STOPPED.")
        except Exception as e:
            print(f"⚠️ Error during standard hardware termination: {e}")
            print("⚠️ Triggering brute force fallback processes kill...")
            os.system("pkill -9 -f uvicorn")
            os.system("pkill -9 -f main.py")
        print("🛑 Hardware components terminated.")

# Global HW Manager
hw = HardwareManager()
threading.Thread(target=hw.update_loop, daemon=True).start()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video/csi/{cam_id}")
async def video_feed(cam_id: int):
    def gen_frames():
        while True:
            frame = hw.latest_frames[cam_id] if cam_id < 4 else None
            
            if frame is None:
                # Generate a black frame with a label if no real frame is available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"CSI Camera {cam_id} (No Signal)", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04) # Limit FPS for streaming

    return StreamingResponse(gen_frames(),
                    media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/realsense")
async def realsense_feed():
    def gen_frames():
        while True:
            frame = hw.latest_realsense
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "RealSense RGB (No Signal)", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/lane")
async def lane_feed():
    def gen_frames():
        while True:
            # Placeholder for lane detection overlay
            # In a real scenario, this would come from a ROS2 subscription or a shared buffer with the perception node
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.line(frame, (200, 480), (300, 300), (0, 255, 0), 5)
            cv2.line(frame, (440, 480), (340, 300), (0, 255, 0), 5)
            cv2.putText(frame, "Lane Detection Preview", (150, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.06)
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(hw.telemetry))
            await asyncio.sleep(0.1) # 10Hz telemetry
    except Exception:
        pass

@app.websocket("/ws/lidar")
async def lidar_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send latest LiDAR distances
            # hw.telemetry could be updated to include lidar if we want it all in one, 
            # but separate WS is often better for high-bandwidth points
            if hw.has_hardware and hasattr(hw, 'latest_lidar'):
                await websocket.send_text(json.dumps(hw.latest_lidar))
            else:
                # Mock lidar points
                num_points = 720
                angles = [i * (2 * math.pi / num_points) for i in range(num_points)]
                t = time.time()
                distances = [6.0 + 1.5 * math.sin(a * 4) + random.uniform(-0.05, 0.05) for a in angles]
                await websocket.send_text(json.dumps({"angles": angles, "distances": distances}))
            await asyncio.sleep(0.1) # 10Hz for smoother dashboard visualization
    except Exception:
        pass

# Mount the frontend directory
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.abspath(os.path.join(current_dir, "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
    hw.stop()

