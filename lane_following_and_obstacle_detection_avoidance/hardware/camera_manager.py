"""
camera_manager.py - QCar2 CSI Camera Hardware Abstraction
=========================================================
Encapsulates the QCarCameras PAL API with the critical nvarguscamerasrc
EGL authorization fix required for headless Jetson Orin operation.

QCar2 Camera IDs (PAL):
  0 = Front CSI
  1 = Right CSI
  2 = Back  CSI
  3 = Left  CSI

When enable_side_csi is True, Front + Right + Left are enabled; lane frame uses camera_id
(typically 0 for front). Side frames: get_side_frames() -> (left, right) from indices 3 and 1.
"""

import os
import sys
import cv2
import numpy as np

# ── CRITICAL FIX: Pop EGL/X11 vars BEFORE any GStreamer/camera import ──
_display_backup = os.environ.pop("DISPLAY", None)
_xauth_backup = os.environ.pop("XAUTHORITY", None)

try:
    from pal.products.qcar import QCarCameras
except ImportError:
    print("[WARNING] QCarCameras PAL library not found. Camera runs in MOCK mode.")
    QCarCameras = None

if _display_backup:
    os.environ["DISPLAY"] = _display_backup
else:
    os.environ["DISPLAY"] = ":1"

if _xauth_backup:
    os.environ["XAUTHORITY"] = _xauth_backup

CSI_LEFT = 3
CSI_RIGHT = 1


class CameraManager:
    """
    Manages QCar2 CSI cameras: primary lane view + optional left/right for avoidance hints.
    """

    def __init__(self, config):
        self.cfg = config.camera
        self.cameras = None
        self._mock_mode = (QCarCameras is None)
        self._last_left = None
        self._last_right = None

    def initialize(self):
        if self._mock_mode:
            print("[CameraManager] MOCK mode - no physical camera.")
            return

        side = bool(getattr(self.cfg, "enable_side_csi", True))
        print(
            f"[CameraManager] Initializing CSI "
            f"({self.cfg.capture_width}x{self.cfg.capture_height} @ {self.cfg.fps}fps), "
            f"lane_idx={self.cfg.camera_id}, side_csi={side}"
        )

        display_bak = os.environ.pop("DISPLAY", None)
        xauth_bak = os.environ.pop("XAUTHORITY", None)

        try:
            self.cameras = QCarCameras(
                frameWidth=int(self.cfg.capture_width),
                frameHeight=int(self.cfg.capture_height),
                frameRate=int(self.cfg.fps),
                enableFront=True,
                enableRight=side,
                enableBack=False,
                enableLeft=side,
            )
            print("[CameraManager] ✓ CSI camera(s) initialized successfully.")
        except Exception as e:
            print(f"[CameraManager] ✗ FATAL: Camera init failed: {e}")
            raise
        finally:
            if display_bak:
                os.environ["DISPLAY"] = display_bak
            else:
                os.environ["DISPLAY"] = ":1"
            if xauth_bak:
                os.environ["XAUTHORITY"] = xauth_bak

    def get_frame(self):
        """Primary lane-following frame from csi[camera_id]. Updates side cache if enabled."""
        if self._mock_mode:
            frame = np.zeros((self.cfg.capture_height, self.cfg.capture_width, 3), dtype=np.uint8)
            cv2.putText(frame, "MOCK CAMERA - Lane",
                        (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.line(frame, (100, 600), (350, 300), (0, 255, 255), 5)
            cv2.line(frame, (720, 600), (470, 300), (0, 255, 255), 5)
            self._last_left = frame.copy()
            self._last_right = frame.copy()
            return frame

        self.cameras.readAll()
        idx = int(self.cfg.camera_id)

        def _valid_img(slot):
            if slot is None:
                return None
            im = getattr(slot, "imageData", None)
            if im is None or im.max() <= 10:
                return None
            return im

        img = _valid_img(self.cameras.csi[idx])
        if img is None:
            for cand in (0, 1, 3, 2):
                if cand == idx:
                    continue
                img = _valid_img(self.cameras.csi[cand])
                if img is not None:
                    break

        if bool(getattr(self.cfg, "enable_side_csi", True)):
            sl = self.cameras.csi[CSI_LEFT]
            sr = self.cameras.csi[CSI_RIGHT]
            self._last_left = getattr(sl, "imageData", None) if sl is not None else None
            self._last_right = getattr(sr, "imageData", None) if sr is not None else None
            if self._last_left is not None and self._last_left.max() <= 10:
                self._last_left = None
            if self._last_right is not None and self._last_right.max() <= 10:
                self._last_right = None
        else:
            self._last_left = None
            self._last_right = None

        return img

    def get_side_frames(self):
        """Last left/right BGR frames from previous get_frame() (same readAll). May be None."""
        return self._last_left, self._last_right

    def terminate(self):
        if self.cameras is not None:
            try:
                self.cameras.terminate()
                print("[CameraManager] ✓ Camera terminated safely.")
            except Exception as e:
                print(f"[CameraManager] Warning during termination: {e}")
