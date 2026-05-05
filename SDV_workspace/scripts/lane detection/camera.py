import os
from pal.utilities.vision import Camera2D

class CameraProcessor:
    """
    Handles initialization and reading from the QCar 2's CSI cameras.
    Optimized for QCar 2's internal 8MP (3264x2464) hardware limit, 
    requesting a downsampled resolution from hardware to save OpenCV CPU cycles.
    """
    def __init__(self, camera_id='2', image_width=820, image_height=410, sample_rate=60):
        # QCar 2 CSI Front camera is '2' (Left is '3', Right is '0', Rear is '1')
        self.camera_id = camera_id
        
        # Request heavily downsampled resolution from driver to preserve computational overhead
        self.image_width = image_width
        self.image_height = image_height
        
        self.sample_rate = sample_rate
        self.sample_time = 1.0 / float(self.sample_rate)

        print(f"Initializing QCar 2 Camera (ID '{self.camera_id}') at {self.image_width}x{self.image_height} @ {self.sample_rate}fps...")

        # Temporarily hide DISPLAY & XAUTHORITY to fix nvarguscamerasrc EGL bug over SSH
        old_disp = os.environ.get('DISPLAY')
        old_xauth = os.environ.get('XAUTHORITY')
        os.environ.pop('DISPLAY', None)
        os.environ.pop('XAUTHORITY', None)

        try:
            # Initialize the CSI camera
            self.my_cam = Camera2D(
                cameraId=self.camera_id,
                frameWidth=self.image_width,
                frameHeight=self.image_height,
                frameRate=self.sample_rate
            )
        finally:
            # Restore X11 environment so cv2.imshow() can open a window for the user
            if old_disp is not None: os.environ['DISPLAY'] = old_disp
            if old_xauth is not None: os.environ['XAUTHORITY'] = old_xauth

    def take_frame(self):
        """
        Grabs a fresh frame from the CSI camera buffer.
        """
        self.my_cam.read()
        return self.my_cam.imageData

    def end_camera(self):
        """
        Safely releases the camera hardware.
        """
        self.my_cam.terminate()
