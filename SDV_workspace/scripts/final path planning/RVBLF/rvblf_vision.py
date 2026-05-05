import cv2
import numpy as np

class RVBLFVisionV2:
    """
    Robust Vision-Based Lane Follower (RVBLF) V2.
    Enhanced with:
    - Perspective Transform (Bird's Eye View)
    - Adaptive Threshold + Canny Edge Detection
    - Sliding Window Polynomial Fitting
    - Metric Calibration (meters)
    """
    def __init__(self, width=820, height=410):
        self.W = width
        self.H = height
        
        # --- CALIBRATION POINTS (Standard QCar2) ---
        # Perspective Transform Source (trapezoid on floor)
        # and Destination (rect BEV)
        self.src_pts = np.float32([
            [260, 200],   # Top-left
            [560, 200],   # Top-right
            [780, 400],   # Bottom-right
            [ 40, 400],   # Bottom-left
        ])
        self.dst_pts = np.float32([
            [100, 0],     # Top-left
            [720, 0],     # Top-right
            [720, 410],   # Bottom-right
            [100, 410],   # Bottom-left
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        # --- METRIC CONVERSION ---
        # 1/10th scale track (~0.37m lane width)
        # These map pixels in BEV to physical meters
        self.lane_width_m = 0.37
        self.lane_width_px = 620.0
        self.xm_per_px = self.lane_width_m / self.lane_width_px
        
        # --- PIPELINE STATE ---
        self.left_fit = None
        self.right_fit = None
        self.confidence = 0.0
        self.lateral_offset_m = 0.0
        
        # Sliding Window Params
        self.n_windows = 10
        self.margin = 60 # Narrower margin to avoid noise
        self.min_pix = 50
        
    def process_frame(self, frame):
        """
        Full V2.1 Pipeline: BEV -> Mask -> Clean Edges -> Sliding Window -> Lookahead Offset
        """
        # 1. Perspective Transform (BEV)
        bev = cv2.warpPerspective(frame, self.M, (self.W, self.H))
        
        # 2. NOISE SUPPRESSION: Mask the background (top 30% of BEV)
        # The background often contains noisy desk/wall elements.
        bev[0:int(self.H * 0.3), :] = 0
        
        # 3. Adaptive Edge Detection with STRONGER noise rejection
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        # Larger blur to kill floor glare speckles
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        
        adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 25, 6)
        canny = cv2.Canny(blur, 30, 100)
        binary = cv2.bitwise_or(adapt, canny)
        
        # Clean up binary with morphological opening
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. Sliding Window Search
        left_fit, right_fit, left_px, right_px = self._sliding_window(binary)
        
        # 5. Polynomial Smoothing
        if left_fit is not None: self.left_fit = left_fit
        if right_fit is not None: self.right_fit = right_fit
        
        # 6. LOOKAHEAD: Evaluate offset further ahead (at H/2) instead of the bumper.
        # This allows the PD controller to anticipate turns.
        self._compute_metrics(self.left_fit, self.right_fit, y_eval=int(self.H * 0.5))
        
        # Confidence calculation (weighted by pixel count)
        self.confidence = min(1.0, (left_px + right_px) / 800.0)
        
        return self.lateral_offset_m, self.confidence > 0.15

    def _sliding_window(self, binary_bev):
        h, w = binary_bev.shape
        histogram = np.sum(binary_bev[h // 2:, :], axis=0)
        midpoint = w // 2
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Ignore weak foundations
        if histogram[leftx_base] < 100: leftx_base = None
        if histogram[rightx_base] < 100: rightx_base = None
        
        nonzero = binary_bev.nonzero()
        ny, nx = np.array(nonzero[0]), np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        lx_curr, rx_curr = leftx_base, rightx_base
        win_h = h // self.n_windows
        
        for win in range(self.n_windows):
            y_low = h - (win + 1) * win_h
            y_high = h - win * win_h
            
            if lx_curr is not None:
                win_inds = ((ny >= y_low) & (ny < y_high) & (nx >= lx_curr - self.margin) & (nx < lx_curr + self.margin)).nonzero()[0]
                left_lane_inds.append(win_inds)
                if len(win_inds) > self.min_pix: lx_curr = int(np.mean(nx[win_inds]))
                
            if rx_curr is not None:
                win_inds = ((ny >= y_low) & (ny < y_high) & (nx >= rx_curr - self.margin) & (nx < rx_curr + self.margin)).nonzero()[0]
                right_lane_inds.append(win_inds)
                if len(win_inds) > self.min_pix: rx_curr = int(np.mean(nx[win_inds]))

        l_fit, r_fit = None, None
        l_px, r_px = 0, 0
        
        if left_lane_inds:
            left_lane_inds = np.concatenate(left_lane_inds)
            if len(left_lane_inds) > 100:
                l_fit = np.polyfit(ny[left_lane_inds], nx[left_lane_inds], 2)
                l_px = len(left_lane_inds)
        
        if right_lane_inds:
            right_lane_inds = np.concatenate(right_lane_inds)
            if len(right_lane_inds) > 100:
                r_fit = np.polyfit(ny[right_lane_inds], nx[right_lane_inds], 2)
                r_px = len(right_lane_inds)
                
        return l_fit, r_fit, l_px, r_px

    def _compute_metrics(self, l_fit, r_fit, y_eval=None):
        if y_eval is None: y_eval = self.H - 1
        x_bev_center = self.W / 2.0
        
        x_left = np.polyval(l_fit, y_eval) if l_fit is not None else None
        x_right = np.polyval(r_fit, y_eval) if r_fit is not None else None
        
        if x_left is not None and x_right is not None:
            lane_center_px = (x_left + x_right) / 2.0
        elif x_left is not None:
            lane_center_px = x_left + (self.lane_width_px / 2.0)
        elif x_right is not None:
            lane_center_px = x_right - (self.lane_width_px / 2.0)
        else:
            self.lateral_offset_m = 0.0
            return
            
        offset_px = lane_center_px - x_bev_center
        self.lateral_offset_m = offset_px * self.xm_per_px
