import numpy as np
import warnings
import cv2
from time import time

class LaneDetect:
    """
    Lane Detection vision pipeline optimized for QCar 2.
    Features bird's-eye perspective transform, dynamic adaptive HSV thresholding,
    optimized sliding windows, and a lightweight custom RANSAC polynomial fitter.
    """
    def __init__(self) -> None:
        warnings.simplefilter('ignore', np.RankWarning)
        # Polynomial state storage
        self.polyright = [0.0, 0.0, 0.0]
        self.polyleft = [0.0, 0.0, 0.0]
        self.polyleft_last = [0.0, 0.0, 0.0]
        self.polyright_last = [0.0, 0.0, 0.0]
        
        # Points found in current frame
        self.left_points = []
        self.right_points = []
        
        self.error = 0.0
        self.matrix_perspective = []
        self.matrix_cache = {}
        
        # Base HSV parameters for Yellow Lane Lines
        self.base_min_hue = 10
        self.base_max_hue = 45
        self.base_min_sat = 50
        self.base_max_sat = 255
        self.base_min_val = 100
        self.base_max_val = 255
        
        # RANSAC params
        self.ransac_min_samples = 15
        self.ransac_residual_threshold = 2.5
        self.ransac_max_trials = 100
        
        # Temporal filtering for error
        self.error_history = []
        self.max_history = 5
        self.process_time = 0.0

    def compute_adaptive_thresholds(self, frame):
        """
        Dynamically calculates HSV bounds based on average scene brightness.
        Unlike the original version, this DOES NOT permanently mutate the baseline fields,
        preventing permanent blindness if the car drives through a long dark tunnel.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        min_v = self.base_min_val
        min_s = self.base_min_sat
        
        if mean_brightness < 80:  # Dark conditions
            min_v = max(50, min_v - 20)
            min_s = max(30, min_s - 20)
        elif mean_brightness > 180:  # Wash-out bright conditions
            min_v = min(150, min_v + 20)
            min_s = min(80, min_s + 20)
            
        lower_bounds = np.array([self.base_min_hue, min_s, min_v])
        upper_bounds = np.array([self.base_max_hue, self.base_max_sat, self.base_max_val])
        return lower_bounds, upper_bounds

    def TransformImage(self, frame):
        """Applies Bird's-Eye transform and isolates yellow lines."""
        # Halve resolution for performance (e.g. 820x410 -> 410x205)
        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        original_img = resize_frame.copy()
        
        h, w = resize_frame.shape[:2]
        
        # Dynamically scale Source and Destination to whatever exact resolution the hardware yields
        self.Source = np.float32([
            [w * 0.33, h * 0.65],
            [w * 0.67, h * 0.65],
            [w * 0.00, h * 0.95],
            [w * 1.00, h * 0.95]
        ])
        self.Destination = np.float32([
            [w * 0.33, 0],
            [w * 0.67, 0],
            [w * 0.33, h],
            [w * 0.67, h]
        ])
        
        frame_shape = resize_frame.shape
        cache_key = f"{frame_shape[0]}_{frame_shape[1]}"
        
        # Cache perspective matrix to save CPU cycles
        if cache_key not in self.matrix_cache:
            self.matrix_perspective = cv2.getPerspectiveTransform(self.Source, self.Destination)
            self.matrix_cache[cache_key] = self.matrix_perspective
        else:
            self.matrix_perspective = self.matrix_cache[cache_key]
            
        resize_frame = cv2.warpPerspective(resize_frame, self.matrix_perspective, (resize_frame.shape[1], resize_frame.shape[0]))
        
        # HSV filtering for yellow track lines
        hsvBuf = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)
        lower_bounds, upper_bounds = self.compute_adaptive_thresholds(resize_frame)
        binaryImage = cv2.inRange(hsvBuf, lower_bounds, upper_bounds)
        
        # Morph ops to remove salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
        
        return original_img, resize_frame, binaryImage
    
    def histogram(self, binaryImage):
        """Finds the starting X position of left and right lanes at the bottom of the screen."""
        init_row = binaryImage.shape[0] // 2
        roi = binaryImage[init_row:, :]
        
        histogram = np.sum(roi, axis=0)
        midpoint = binaryImage.shape[1] // 2
        
        left_ptr = np.argmax(histogram[:midpoint])
        right_ptr = np.argmax(histogram[midpoint:]) + midpoint
        
        # Stronger rejection of noise peaks
        if histogram[left_ptr] < 100: left_ptr = -1
        if histogram[right_ptr] < 100: right_ptr = -1
            
        return left_ptr, right_ptr
    
    def locate_lanes(self, img):
        """Sliding window search along the vertical axis of the bird's eye view."""
        start_time = time()
        self.left_points = []
        self.right_points = []
        
        nwindows = 10
        margin = 40
        minpix = 30
        
        leftx_current, rightx_current = self.histogram(img)
        
        # Fallback to history if peaks lost
        if leftx_current == -1 and hasattr(self, 'left_base_pos'):
            leftx_current = self.left_base_pos
        elif leftx_current != -1:
            self.left_base_pos = leftx_current
            
        if rightx_current == -1 and hasattr(self, 'right_base_pos'):
            rightx_current = self.right_base_pos
        elif rightx_current != -1:
            self.right_base_pos = rightx_current
            
        window_height = img.shape[0] // nwindows
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            
            # Left window bounds
            win_xleft_low = int(max(0, leftx_current - margin))
            win_xleft_high = int(min(img.shape[1] - 1, leftx_current + margin))
            
            # Right window bounds
            win_xright_low = int(max(0, rightx_current - margin))
            win_xright_high = int(min(img.shape[1] - 1, rightx_current + margin))
            
            # Extract points within windows (NumPy vectorized boolean mask logic)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            if len(good_left_inds) > 0:
                self.left_points.extend(zip(nonzeroy[good_left_inds], nonzerox[good_left_inds]))
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                
            if len(good_right_inds) > 0:
                self.right_points.extend(zip(nonzeroy[good_right_inds], nonzerox[good_right_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        self.process_time = time() - start_time
        
    def custom_ransac_fit(self, points, degree=2):
        """Custom RANSAC to fit polynomials to lane points resistant to noise."""
        if len(points) < self.ransac_min_samples:
            return None, False
            
        x_vals = np.array([p[0] for p in points])
        y_vals = np.array([p[1] for p in points])
        
        best_coeffs = None
        best_inlier_count = 0
        total_pts = len(points)
        
        for _ in range(self.ransac_max_trials):
            sample_indices = np.random.choice(total_pts, self.ransac_min_samples, replace=False)
            try:
                coeffs = np.polyfit(x_vals[sample_indices], y_vals[sample_indices], degree)
                y_pred = np.polyval(coeffs, x_vals)
                errors = np.abs(y_vals - y_pred)
                
                inliers = errors < self.ransac_residual_threshold
                inlier_count = np.sum(inliers)
                
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_coeffs = coeffs
                    if inlier_count > total_pts * 0.85: # Good enough, early exit
                        break
            except:
                continue
        
        # Polish step with all best inliers
        if best_coeffs is not None and best_inlier_count >= self.ransac_min_samples:
            try:
                y_pred = np.polyval(best_coeffs, x_vals)
                inliers = np.abs(y_vals - y_pred) < self.ransac_residual_threshold
                final_coeffs = np.polyfit(x_vals[inliers], y_vals[inliers], degree)
                return final_coeffs, True
            except:
                return best_coeffs, True
            
        # Fallback to least squares
        try:
            return np.polyfit(x_vals, y_vals, degree), True
        except:
            return None, False
    
    def process_lane_geometry(self, img, original_img):
        """Fits lines, computes tracking error, and draws visualization."""
        center_cam = (img.shape[1] // 2) # Front camera is perfectly centered in the QCar
        center_lines = center_cam 
        
        coeff_R, success_R = self.custom_ransac_fit(self.right_points)
        coeff_L, success_L = self.custom_ransac_fit(self.left_points)
        
        if success_R: self.polyright = coeff_R.tolist()
        if success_L: self.polyleft = coeff_L.tolist()
        
        draw_color = (0, 255, 0)
        
        if success_L and success_R:
            # Both lines
            y_eval = img.shape[0] - 1
            x_R = self.polyright[2] + self.polyright[1]*y_eval + self.polyright[0]*(y_eval**2)
            x_L = self.polyleft[2] + self.polyleft[1]*y_eval + self.polyleft[0]*(y_eval**2)
            center_lines = (x_R + x_L) / 2
            self.polyleft_last, self.polyright_last = list(self.polyleft), list(self.polyright)
        elif success_L:
            # Left only
            y_eval = img.shape[0] - 1
            x_L = self.polyleft[2] + self.polyleft[1]*y_eval + self.polyleft[0]*(y_eval**2)
            center_lines = x_L + 125  # Estimated right lane offset
            self.polyleft_last = list(self.polyleft)
        elif success_R:
            # Right only
            y_eval = img.shape[0] - 1
            x_R = self.polyright[2] + self.polyright[1]*y_eval + self.polyright[0]*(y_eval**2)
            center_lines = x_R - 125
            self.polyright_last = list(self.polyright)
        else:
            # Neither found - Extrapolate from history
            draw_color = (0, 0, 255) # Red implies degraded state
            y_eval = img.shape[0] - 1
            x_L = self.polyleft_last[2] + self.polyleft_last[1]*y_eval + self.polyleft_last[0]*(y_eval**2)
            x_R = self.polyright_last[2] + self.polyright_last[1]*y_eval + self.polyright_last[0]*(y_eval**2)
            center_lines = (x_R + x_L) / 2
        
        # Clamp bounds
        center_lines = min(max(0, center_lines), img.shape[1] - 1)
        
        # Calculate Error
        distance_center = center_cam - center_lines
        self.error = distance_center
        
        # Smooth error for HUD direction
        self.error_history.append(distance_center / (img.shape[1] / 2))
        if len(self.error_history) > self.max_history: self.error_history.pop(0)
        smoothed_error = np.mean(self.error_history)
        
        turn_dir = "CENTER" if abs(smoothed_error) < 0.1 else ("LEFT" if smoothed_error > 0 else "RIGHT")
        
        # Visual Drawing 
        for row in range(img.shape[0] - 1, -1, -12):
            if success_L or draw_color == (0, 0, 255):
                poly = self.polyleft if success_L else self.polyleft_last
                c = poly[2] + poly[1]*row + poly[0]*(row**2)
                cv2.circle(img, (int(c), int(row)), 2, draw_color, -1)
            if success_R or draw_color == (0, 0, 255):
                poly = self.polyright if success_R else self.polyright_last
                c = poly[2] + poly[1]*row + poly[0]*(row**2)
                cv2.circle(img, (int(c), int(row)), 2, draw_color, -1)
                
        # Draw Center axis
        cv2.line(img, (int(center_cam), int(img.shape[0]/4)), (int(center_cam), int(img.shape[0]*3/4)), (255, 0, 0), 2)
        cv2.line(img, (int(center_lines), 0), (int(center_cam), int(img.shape[0]-1)), (0, 165, 255), 3)
        
        cv2.putText(img, f"Dir: {turn_dir}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Error: {self.error:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Time: {self.process_time*1000:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Warp back and overlay
        invertedMatrix = np.linalg.inv(self.matrix_perspective)
        warped_frame = cv2.warpPerspective(img, invertedMatrix, (img.shape[1], img.shape[0]))
        return cv2.addWeighted(original_img, 0.7, warped_frame, 0.4, 0)
    
    def find_lines(self, frame):
        """Top-level execution pipeline."""
        original_img, resize_frame, binary_image = self.TransformImage(frame)
        self.locate_lanes(binary_image)
        return self.process_lane_geometry(resize_frame, original_img)
