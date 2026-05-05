import cv2
import numpy as np

class VisionController:
    def __init__(self, kp=0.002, kd=0.0005, target_offset_x=320):
        """
        Initializes the PID vision controller.
        kp: Proportional Gain
        kd: Derivative Gain (dampens the steering, prevents zig-zags)
        target_offset_x: Desired centerline in the image
        """
        self.kp = kp
        self.kd = kd
        self.target_x = target_offset_x
        self.last_error = 0
        
        # HSV threshold for Yellow (adjust if your mat colors differ)
        self.lower_y = np.array([10, 50, 100])
        self.upper_y = np.array([45, 255, 255])
        
        # Morphology kernels
        self.kernel = np.ones((5, 5), np.uint8)

    def get_mask(self, image):
        """Performs color filtering and noise reduction."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_y, self.upper_y)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def calculate_steering(self, image, dt):
        """
        Processes the image and returns a PD-controlled steering command.
        """
        # Crop the image to focus on the road (bottom 40%)
        h, w = image.shape[:2]
        roi = image[int(h*0.6):h, :]
        
        mask = self.get_mask(roi)
        
        # Identify the largest lane contour
        centroids = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        if len(centroids) > 0:
            c = max(centroids, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                
                # PD Control Law
                error = cx - self.target_x
                derivative = (error - self.last_error) / dt if dt > 0 else 0
                self.last_error = error
                
                # Calculate output steer (radians)
                steer = (self.kp * error) + (self.kd * derivative)
                
                # Clip output
                return -np.clip(steer, -0.5, 0.5), True
        
        # No line detected
        return 0.0, False

    def turn_maneuver(self, cam_image, side="right"):
        """
        Processes the side cameras (Left/Right) to hug the boundary curve during a turn.
        side: "right" uses Right_CSI to hug right edge. "left" uses Left_CSI.
        """
        # Similar logic but targeting a specific vertical offset 
        # based on your map geometry during the corner.
        # For simplicity, returning a fixed nudge if line is lost.
        return 0.35 if side == "left" else -0.35

if __name__ == '__main__':
    # Mock Test
    vc = VisionController()
    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put a yellow line in the ROI
    cv2.rectangle(fake_img, (350, 300), (370, 480), (0, 255, 255), -1) 
    steer, found = vc.calculate_steering(fake_img, 0.1)
    print(f"Steer command: {steer:.3f} | Found: {found}")
