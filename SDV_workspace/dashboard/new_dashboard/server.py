from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import sys
import os
import logging
import asyncio
import json
import math
import random
import time
import threading
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np

# Quanser path setup - pop DISPLAY for headless operation
os.environ.pop("DISPLAY", None)
os.environ.pop("XAUTHORITY", None)
QUANSER_LIB_PATH = '/home/nvidia/Documents/Quanser/0_libraries/python'
if os.path.exists(QUANSER_LIB_PATH) and QUANSER_LIB_PATH not in sys.path:
    sys.path.insert(0, QUANSER_LIB_PATH)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'qcar2_dashboard')
client = None
db = None

try:
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    print(f"✅ MongoDB connected to {mongo_url}")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}. Running without database support.")

# --- Lifespan ---
@asynccontextmanager
async def lifespan(application):
    yield
    print("🛑 Shutting down server and hardware...")
    if client:
        client.close()
    if hw:
        hw.stop()
    print("✅ Shutdown complete.")

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")


# --- Hardware Manager ---
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
        self.latest_lidar = [] # Array of distances
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
            print("✅ QCar 2 Hardware Initialized Successfully.")
        except Exception as e:
            print(f"⚠️ Hardware not found or error: {e}. Running in Simulation Mode.")

    def update_loop(self):
        while self.active:
            if self.has_hardware:
                try:
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
                            self.latest_lidar = [d if not (math.isinf(d) or math.isnan(d)) else 0.0 for d in dists]
                    
                    # Basic telemetry update (mock battery drain, but could be real if QCar supports it)
                    self.telemetry["battery"] = max(0.0, self.telemetry["battery"] - 0.0001)
                except Exception as e:
                    print(f"❌ Hardware loop error: {e}")
            else:
                # Simulation Mode
                t = time.time()
                self.telemetry["battery"] = max(0.0, 100.0 - (t % 1000) / 100.0)
                self.telemetry["steering_angle"] = math.sin(t) * 25.0
                self.telemetry["speed"] = 1.2 + math.sin(t / 2) * 0.5
                self.latest_lidar = [1.5 + math.sin(i * 0.1 + t) * 0.5 + random.uniform(0, 0.2) for i in range(40)]
            
            time.sleep(0.04)

    def stop(self):
        self.active = False
        try:
            if self.cameras: self.cameras.terminate()
            if self.realsense: self.realsense.terminate()
            if hasattr(self, 'lidar') and self.lidar: self.lidar.terminate()
            if self.qcar: self.qcar.terminate()
        except Exception as e:
            print(f"⚠️ Error during termination: {e}")

hw = HardwareManager()
threading.Thread(target=hw.update_loop, daemon=True).start()


# --- Video Streaming Endpoints ---
def generate_sim_frame(label, color_base, cam_id=None):
    """Generate a simulation frame with label and animated elements."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = time.time()

    # Background gradient
    for y in range(480):
        ratio = y / 480
        frame[y, :] = [int(color_base[0] * (0.3 + 0.7 * ratio)),
                       int(color_base[1] * (0.3 + 0.7 * ratio)),
                       int(color_base[2] * (0.3 + 0.7 * ratio))]

    # Animated scan line
    scan_y = int((math.sin(t * 2) * 0.5 + 0.5) * 480)
    cv2.line(frame, (0, scan_y), (640, scan_y), (0, 255, 100), 1)

    # Grid overlay
    for x in range(0, 640, 80):
        cv2.line(frame, (x, 0), (x, 480), (40, 40, 40), 1)
    for y in range(0, 480, 60):
        cv2.line(frame, (0, y), (640, y), (40, 40, 40), 1)

    # Center crosshair
    cx, cy = 320, 240
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 200, 200), 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 200, 200), 1)

    # Label
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Timestamp
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

    # REC indicator
    if int(t * 2) % 2 == 0:
        cv2.circle(frame, (600, 460), 6, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (610, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame


CSI_LABELS = ["Front CSI", "Left CSI", "Right CSI", "Rear CSI"]
CSI_COLORS = [(20, 40, 60), (20, 50, 40), (40, 20, 50), (50, 30, 20)]


@api_router.get("/video/csi/{cam_id}")
async def video_feed(cam_id: int):
    def gen_frames():
        while True:
            frame = hw.latest_frames[cam_id] if cam_id < 4 else None
            
            if frame is None:
                # Black frame placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                label = CSI_LABELS[cam_id] if cam_id < 4 else f"CSI Cam {cam_id}"
                cv2.putText(frame, f"{label} (No Signal)", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# RealSense view labels for fallback frames
RS_VIEWS = {
    "rgb": ("RGB View",),
    "depth": ("Depth View",),
    "infrared": ("Infrared View",),
}

@api_router.get("/video/realsense")
async def realsense_feed(view: str = "rgb"):
    def gen_frames():
        while True:
            frame = hw.latest_realsense if view == "rgb" else None
            
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                label = RS_VIEWS.get(view, RS_VIEWS["rgb"])[0]
                cv2.putText(frame, f"{label} (No Signal)", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@api_router.get("/video/lane")
async def lane_feed():
    def gen_frames():
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.line(frame, (200, 480), (300, 300), (0, 255, 0), 5)
            cv2.line(frame, (440, 480), (340, 300), (0, 255, 0), 5)
            cv2.putText(frame, "Lane Detection Preview", (150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.06)

    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# --- WebSocket Endpoints ---
@api_router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(hw.telemetry))
            await asyncio.sleep(0.1)
    except Exception:
        pass


@api_router.websocket("/ws/lidar")
async def websocket_lidar(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(hw.latest_lidar))
            await asyncio.sleep(0.1)
    except Exception:
        pass


# --- Existing Models & Routes ---
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    if db is None:
        return StatusCheck(client_name=input.client_name)
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is None:
        return []
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


# Include router and middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mount the static files (the dashboard)
# This allows running the dashboard on the car without a separate Node.js server
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=current_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



