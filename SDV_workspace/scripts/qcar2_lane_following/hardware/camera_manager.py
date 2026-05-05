import os
import cv2
import numpy as np
import time

try:
    from pal.products.qcar import QCarCameras
except ImportError:
    print("[WARNING] QCar PAL libraries not found. Running CameraManager in MOCK Mode.")
    QCarCameras = None


class CameraManager:
    """Manages the QCar2 front CSI camera with headless initialization."""

    def __init__(self, config):
        self.cfg = config
        self.cameras = None
        self._mock_mode = (QCarCameras is None)
        self._frame_count = 0

    def initialize(self):
        if self._mock_mode:
            print("[CameraManager] Simulated camera initialized.")
            return

        print("[CameraManager] Initializing front CSI camera...")

        # Strip DISPLAY/XAUTHORITY for headless nvarguscamerasrc
        display_backup = os.environ.pop("DISPLAY", None)
        xauth_backup = os.environ.pop("XAUTHORITY", None)

        try:
            self.cameras = QCarCameras(
                frameWidth=self.cfg.camera.width,
                frameHeight=self.cfg.camera.height,
                frameRate=self.cfg.camera.fps,
                enableRight=False,
                enableBack=False,
                enableFront=True,
                enableLeft=False
            )
            print(f"[CameraManager] Camera initialized at "
                  f"{self.cfg.camera.width}×{self.cfg.camera.height} @ {self.cfg.camera.fps}fps")
        except Exception as e:
            print(f"[CameraManager] FATAL: Failed to init camera: {e}")
            raise
        finally:
            # Restore DISPLAY for OpenCV GUI
            if display_backup:
                os.environ["DISPLAY"] = display_backup
            else:
                os.environ["DISPLAY"] = ":1"
            if xauth_backup:
                os.environ["XAUTHORITY"] = xauth_backup

        # Warmup: skip first N frames (often dark/blank)
        print(f"[CameraManager] Warming up ({self.cfg.camera.warmup_frames} frames)...")
        for _ in range(self.cfg.camera.warmup_frames):
            self.cameras.readAll()
            time.sleep(0.03)
        print("[CameraManager] Camera ready ✅")

    def get_frame(self):
        """Read a frame from the front CSI camera. Returns BGR numpy array or None."""
        if self._mock_mode:
            return self._mock_frame()

        self.cameras.readAll()
        img = self.cameras.csiFront.imageData

        if img is None or img.max() <= 10:
            return None

        self._frame_count += 1
        return img

    def _mock_frame(self):
        """Generate a synthetic test frame with lane lines for offline testing."""
        h, w = self.cfg.camera.height, self.cfg.camera.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark road surface
        frame[:, :] = (60, 60, 60)

        # Draw lane lines (dark lines on gray surface, mimicking real track)
        cv2.line(frame, (200, h), (350, int(h * 0.45)), (30, 30, 30), 4)
        cv2.line(frame, (620, h), (470, int(h * 0.45)), (30, 30, 30), 4)

        # Yellow center dashes
        for y in range(int(h * 0.5), h, 30):
            cv2.line(frame, (410, y), (410, min(y + 15, h)), (0, 180, 255), 2)

        cv2.putText(frame, "MOCK CAMERA", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    def terminate(self):
        if self.cameras is not None:
            try:
                self.cameras.terminate()
            except Exception:
                pass
            print("[CameraManager] Terminated safely.")
