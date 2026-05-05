"""
camera_manager.py - QCar2 CSI Camera Hardware Abstraction
=========================================================
Encapsulates the QCarCameras PAL API with the critical nvarguscamerasrc
EGL authorization fix required for headless Jetson Orin operation.

The fix works by:
1. Popping DISPLAY and XAUTHORITY env vars BEFORE camera init
2. This prevents nvarguscamerasrc from trying to connect to an X11 display
3. After init, we restore the vars so cv2.imshow() still works

QCar2 Camera IDs:
  0 = Front CSI
  1 = Right CSI
  2 = Back  CSI
  3 = Left  CSI
"""

import os
import sys
import cv2
import numpy as np

# ── CRITICAL FIX: Pop EGL/X11 vars BEFORE any GStreamer/camera import ──
# This MUST happen at module load time, before QCarCameras touches GStreamer
_display_backup = os.environ.pop("DISPLAY", None)
_xauth_backup = os.environ.pop("XAUTHORITY", None)

try:
    from pal.products.qcar import QCarCameras
except ImportError:
    print("[WARNING] QCarCameras PAL library not found. Camera runs in MOCK mode.")
    QCarCameras = None

# ── Restore display vars so OpenCV windows work later ──
if _display_backup:
    os.environ["DISPLAY"] = _display_backup
else:
    os.environ["DISPLAY"] = ":1"  # Default for RDP/VNC on Jetson

if _xauth_backup:
    os.environ["XAUTHORITY"] = _xauth_backup


class CameraManager:
    """
    Manages the QCar2 front CSI camera lifecycle.

    Methods:
        initialize()  - Opens camera hardware connection
        get_frame()   - Returns BGR numpy array or None on failure
        terminate()   - Safely releases camera hardware
    """

    def __init__(self, config):
        """
        Args:
            config: AppConfig instance containing camera parameters.
        """
        self.cfg = config.camera
        self.cameras = None
        self._mock_mode = (QCarCameras is None)

    def initialize(self):
        """
        Opens the front CSI camera on QCar2 with configured resolution/fps.
        Uses the nvarguscamerasrc EGL fix (env vars already popped at import).
        """
        if self._mock_mode:
            print("[CameraManager] MOCK mode - no physical camera.")
            return

        print(f"[CameraManager] Initializing Front CSI camera "
              f"({self.cfg.capture_width}x{self.cfg.capture_height} @ {self.cfg.fps}fps)...")

        # Pop display vars again right before init for extra safety
        display_bak = os.environ.pop("DISPLAY", None)
        xauth_bak = os.environ.pop("XAUTHORITY", None)

        try:
            # QCar2 QCarCameras API: enable only the front camera
            # camera_id=0 is Front on QCar2
            self.cameras = QCarCameras(
                frameWidth=self.cfg.capture_width,
                frameHeight=self.cfg.capture_height,
                frameRate=self.cfg.fps,
                enableFront=True,
                enableRight=False,
                enableBack=False,
                enableLeft=False
            )
            print("[CameraManager] ✓ Front CSI camera initialized successfully.")
        except Exception as e:
            print(f"[CameraManager] ✗ FATAL: Camera init failed: {e}")
            raise
        finally:
            # Restore display vars so cv2.imshow works
            if display_bak:
                os.environ["DISPLAY"] = display_bak
            else:
                os.environ["DISPLAY"] = ":1"
            if xauth_bak:
                os.environ["XAUTHORITY"] = xauth_bak

    def get_frame(self):
        """
        Reads and returns the latest frame from the front CSI camera.

        Returns:
            numpy.ndarray: BGR image of shape (height, width, 3), or
            None: if frame capture failed (blank/dropped frame).
        """
        if self._mock_mode:
            # Generate a synthetic test frame for development without hardware
            frame = np.zeros((self.cfg.capture_height, self.cfg.capture_width, 3), dtype=np.uint8)
            cv2.putText(frame, "MOCK CAMERA - No QCar2 Hardware",
                        (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Draw fake yellow lane lines for testing
            cv2.line(frame, (100, 600), (350, 300), (0, 255, 255), 5)
            cv2.line(frame, (720, 600), (470, 300), (0, 255, 255), 5)
            return frame

        # Read all enabled cameras (only front is enabled)
        self.cameras.readAll()
        img = self.cameras.csi[self.cfg.camera_id].imageData

        # Validate frame: blank frames (CSI grab failure) have max pixel ≤ 10
        if img is None or img.max() <= 10:
            return None

        return img

    def terminate(self):
        """Safely releases camera hardware."""
        if self.cameras is not None:
            try:
                self.cameras.terminate()
                print("[CameraManager] ✓ Camera terminated safely.")
            except Exception as e:
                print(f"[CameraManager] Warning during termination: {e}")
