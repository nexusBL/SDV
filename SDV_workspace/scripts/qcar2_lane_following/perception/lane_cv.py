import cv2
import numpy as np
import time

class LaneDetector:
    """
    Edge-detection based lane detector for the QCar2 physical track.

    Pipeline:
      1. ROI masking (trapezoidal, bottom portion of frame)
      2. Grayscale → Gaussian blur → Adaptive threshold + Canny edges
      3. Perspective transform to bird's-eye view (BEV)
      4. Histogram peak detection for lane base positions
      5. Sliding window search for lane pixels
      6. 2nd-degree polynomial fit
      7. EMA temporal smoothing with confidence gating
      8. Metric computation (lateral offset in meters, curvature radius)
      9. Visualization overlays
    """

    def __init__(self, config):
        self.cfg = config
        self.W = config.camera.width
        self.H = config.camera.height

        # Precompute perspective transform matrices
        self.M = cv2.getPerspectiveTransform(
            config.cv.src_points, config.cv.dst_points
        )
        self.Minv = cv2.getPerspectiveTransform(
            config.cv.dst_points, config.cv.src_points
        )

        # Precompute ROI mask
        self._roi_mask = self._build_roi_mask()

        # Pixel-to-meter conversion factors
        self.xm_per_px = config.cv.lane_width_m / config.cv.lane_width_px
        self.ym_per_px = config.cv.bev_height_m / config.cv.bev_height_px

        # EMA-smoothed polynomial coefficients
        self._left_fit = None    # Current smoothed left polynomial
        self._right_fit = None   # Current smoothed right polynomial

        # Last valid fits (for confidence hold)
        self._last_good_left = None
        self._last_good_right = None

        # Metrics
        self.lateral_offset_m = 0.0
        self.curvature_radius_m = float('inf')
        self.confidence = 0.0   # 0.0 to 1.0
        self.left_pixel_count = 0
        self.right_pixel_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # ROI Mask
    # ──────────────────────────────────────────────────────────────────────
    def _build_roi_mask(self):
        """Build a trapezoidal binary mask from fractional coordinates."""
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        poly_frac = self.cfg.cv.roi_poly_frac
        pts = np.int32([
            [int(poly_frac[i][0] * self.W), int(poly_frac[i][1] * self.H)]
            for i in range(len(poly_frac))
        ])
        cv2.fillPoly(mask, [pts], 255)
        return mask

    # ──────────────────────────────────────────────────────────────────────
    # Main Processing Entry Point
    # ──────────────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Process a single camera frame.

        Returns:
            error_m: Lateral offset from lane center in meters (float or None).
                     Positive = car is RIGHT of center, needs to steer LEFT.
                     None = no lanes detected at all.
            hud:     Annotated debug frame (BGR image).
        """
        # 1. Apply ROI mask
        roi_frame = cv2.bitwise_and(frame, frame, mask=self._roi_mask)

        # 2. Edge detection (grayscale → blur → adaptive + canny)
        binary = self._edge_detection(roi_frame)

        # 3. Perspective transform to bird's-eye view
        bev_binary = cv2.warpPerspective(
            binary, self.M, (self.W, self.H), flags=cv2.INTER_LINEAR
        )

        # 4. Sliding window search
        left_fit_raw, right_fit_raw, left_px, right_px, sw_debug = \
            self._sliding_window(bev_binary)

        self.left_pixel_count = left_px
        self.right_pixel_count = right_px

        # 5. Confidence gating + EMA smoothing
        left_fit, right_fit = self._smooth_and_gate(
            left_fit_raw, right_fit_raw, left_px, right_px
        )

        # 6. Compute metrics
        error_m = self._compute_metrics(left_fit, right_fit)

        # 7. Visualization
        hud = self._draw_overlays(
            frame, binary, bev_binary, sw_debug, left_fit, right_fit
        )

        return error_m, hud

    # ──────────────────────────────────────────────────────────────────────
    # Edge Detection
    # ──────────────────────────────────────────────────────────────────────
    def _edge_detection(self, frame):
        """
        Produce a binary edge mask from the ROI-masked frame.
        Uses adaptive threshold + Canny, combined with OR.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.cfg.cv.blur_ksize,
                                        self.cfg.cv.blur_ksize), 0)

        # Adaptive threshold: picks up dark lines on lighter road surface
        adapt = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.cfg.cv.adaptive_block_size,
            self.cfg.cv.adaptive_C
        )

        # Canny edge detection: picks up sharp edges
        canny = cv2.Canny(blur, self.cfg.cv.canny_low, self.cfg.cv.canny_high)

        # Combine both approaches
        combined = cv2.bitwise_or(adapt, canny)

        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    # ──────────────────────────────────────────────────────────────────────
    # Sliding Window Search
    # ──────────────────────────────────────────────────────────────────────
    def _sliding_window(self, binary_bev):
        """
        Histogram-based sliding window lane pixel search on the BEV binary image.

        Returns:
            left_fit:  2nd order poly coefficients or None
            right_fit: 2nd order poly coefficients or None
            left_px:   number of left lane pixels found
            right_px:  number of right lane pixels found
            out_img:   debug visualization image
        """
        out_img = np.dstack((binary_bev, binary_bev, binary_bev))
        h, w = binary_bev.shape

        # Histogram of bottom half to find lane base positions
        histogram = np.sum(binary_bev[h // 2:, :], axis=0)
        midpoint = w // 2

        # Find peaks in left and right halves
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Check if there's actually a peak (not just noise)
        if histogram[leftx_base] < 100:
            leftx_base = None
        if histogram[rightx_base] < 100:
            rightx_base = None

        # Window parameters
        n_windows = self.cfg.cv.n_windows
        window_height = h // n_windows
        margin = self.cfg.cv.margin
        min_pix = self.cfg.cv.min_pixels

        # All nonzero pixel coordinates
        nonzero = binary_bev.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for win in range(n_windows):
            win_y_low = h - (win + 1) * window_height
            win_y_high = h - win * window_height

            # Left window
            if leftx_current is not None:
                win_xl = max(0, leftx_current - margin)
                win_xh = min(w, leftx_current + margin)
                cv2.rectangle(out_img, (win_xl, win_y_low),
                              (win_xh, win_y_high), (0, 255, 0), 2)
                good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xl) & (nonzerox < win_xh)).nonzero()[0]
                left_lane_inds.append(good)
                if len(good) > min_pix:
                    leftx_current = int(np.mean(nonzerox[good]))

            # Right window
            if rightx_current is not None:
                win_xl = max(0, rightx_current - margin)
                win_xh = min(w, rightx_current + margin)
                cv2.rectangle(out_img, (win_xl, win_y_low),
                              (win_xh, win_y_high), (0, 255, 0), 2)
                good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xl) & (nonzerox < win_xh)).nonzero()[0]
                right_lane_inds.append(good)
                if len(good) > min_pix:
                    rightx_current = int(np.mean(nonzerox[good]))

        # Concatenate indices and fit polynomials
        left_fit, right_fit = None, None
        left_px_count, right_px_count = 0, 0

        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            left_px_count = len(left_lane_inds)
            if left_px_count > 0:
                lx = nonzerox[left_lane_inds]
                ly = nonzeroy[left_lane_inds]
                out_img[ly, lx] = [255, 0, 0]  # Blue for left lane pixels
                try:
                    left_fit = np.polyfit(ly, lx, self.cfg.cv.poly_degree)
                except (np.linalg.LinAlgError, TypeError):
                    left_fit = None

        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(right_lane_inds)
            right_px_count = len(right_lane_inds)
            if right_px_count > 0:
                rx = nonzerox[right_lane_inds]
                ry = nonzeroy[right_lane_inds]
                out_img[ry, rx] = [0, 0, 255]  # Red for right lane pixels
                try:
                    right_fit = np.polyfit(ry, rx, self.cfg.cv.poly_degree)
                except (np.linalg.LinAlgError, TypeError):
                    right_fit = None

        return left_fit, right_fit, left_px_count, right_px_count, out_img

    # ──────────────────────────────────────────────────────────────────────
    # Temporal Smoothing + Confidence Gating
    # ──────────────────────────────────────────────────────────────────────
    def _smooth_and_gate(self, left_fit_raw, right_fit_raw, left_px, right_px):
        """
        Apply EMA smoothing with confidence gating.
        If a lane has <min_lane_pixels, hold the last good fit.
        """
        alpha = self.cfg.cv.ema_alpha
        min_px = self.cfg.cv.min_lane_pixels

        # Left lane
        if left_fit_raw is not None and left_px >= min_px:
            if self._left_fit is not None:
                self._left_fit = alpha * left_fit_raw + (1 - alpha) * self._left_fit
            else:
                self._left_fit = left_fit_raw.copy()
            self._last_good_left = self._left_fit.copy()
        else:
            # Hold last good fit
            if self._last_good_left is not None:
                self._left_fit = self._last_good_left.copy()

        # Right lane
        if right_fit_raw is not None and right_px >= min_px:
            if self._right_fit is not None:
                self._right_fit = alpha * right_fit_raw + (1 - alpha) * self._right_fit
            else:
                self._right_fit = right_fit_raw.copy()
            self._last_good_right = self._right_fit.copy()
        else:
            if self._last_good_right is not None:
                self._right_fit = self._last_good_right.copy()

        # Compute overall confidence
        left_conf = min(1.0, left_px / max(min_px, 1))
        right_conf = min(1.0, right_px / max(min_px, 1))
        self.confidence = (left_conf + right_conf) / 2.0

        return self._left_fit, self._right_fit

    # ──────────────────────────────────────────────────────────────────────
    # Metrics Computation
    # ──────────────────────────────────────────────────────────────────────
    def _compute_metrics(self, left_fit, right_fit):
        """
        Compute lateral offset (meters) and curvature radius (meters).
        Returns error_m or None if both lanes are missing.
        """
        y_eval = self.H - 1   # Evaluate at bottom of BEV image
        x_center = self.W / 2.0  # BEV image center

        x_left = None
        x_right = None

        if left_fit is not None:
            x_left = np.polyval(left_fit, y_eval)
        if right_fit is not None:
            x_right = np.polyval(right_fit, y_eval)

        # Calculate lane center in pixels
        if x_left is not None and x_right is not None:
            lane_center_px = (x_left + x_right) / 2.0
        elif x_left is not None:
            # Only left lane visible: estimate right lane from known lane width
            lane_center_px = x_left + self.cfg.cv.lane_width_px / 2.0
        elif x_right is not None:
            lane_center_px = x_right - self.cfg.cv.lane_width_px / 2.0
        else:
            # Both lanes lost
            self.lateral_offset_m = 0.0
            self.curvature_radius_m = float('inf')
            return None

        # Convert pixel offset to meters
        offset_px = lane_center_px - x_center
        self.lateral_offset_m = offset_px * self.xm_per_px

        # Compute curvature radius in meters
        self.curvature_radius_m = self._compute_curvature(
            left_fit, right_fit, y_eval
        )

        return self.lateral_offset_m

    def _compute_curvature(self, left_fit, right_fit, y_eval):
        """Compute radius of curvature in meters at y_eval."""
        radii = []

        for fit in [left_fit, right_fit]:
            if fit is not None and len(fit) >= 3:
                # Convert polynomial to meter-space
                # x = a*y^2 + b*y + c  →  x_m = a_m * y_m^2 + b_m * y_m + c_m
                a = fit[0] * self.xm_per_px / (self.ym_per_px ** 2)
                b = fit[1] * self.xm_per_px / self.ym_per_px
                y_m = y_eval * self.ym_per_px

                # R = (1 + (2*a*y + b)^2)^(3/2) / |2*a|
                denom = abs(2 * a)
                if denom > 1e-6:
                    numer = (1 + (2 * a * y_m + b) ** 2) ** 1.5
                    radii.append(numer / denom)

        if len(radii) > 0:
            return np.mean(radii)
        return float('inf')

    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────
    def _draw_overlays(self, original, binary, bev_binary, sw_debug,
                       left_fit, right_fit):
        """
        Draw lane polygon overlay, center line, metrics text, and PiP debug panels.
        """
        color_warp = np.zeros_like(original)
        ploty = np.linspace(0, self.H - 1, self.H)

        has_both = left_fit is not None and right_fit is not None
        has_any = left_fit is not None or right_fit is not None

        # 1. Lane polygon overlay on BEV, then warp back
        if has_both:
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)

            # Clamp to frame bounds
            left_fitx = np.clip(left_fitx, 0, self.W - 1)
            right_fitx = np.clip(right_fitx, 0, self.W - 1)

            pts_left = np.array(
                [np.transpose(np.vstack([left_fitx, ploty]))]
            )
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]
            )
            pts = np.hstack((pts_left, pts_right))

            # Green filled lane polygon
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 180, 0))

            # Yellow center trajectory line
            center_fitx = (left_fitx + right_fitx) / 2
            center_pts = np.int32(np.column_stack((center_fitx, ploty)))
            cv2.polylines(color_warp, [center_pts], False, (0, 255, 255), 3)

            # Draw lane boundary curves (red/blue)
            left_pts = np.int32(np.column_stack((left_fitx, ploty)))
            right_pts = np.int32(np.column_stack((right_fitx, ploty)))
            cv2.polylines(color_warp, [left_pts], False, (255, 100, 0), 2)
            cv2.polylines(color_warp, [right_pts], False, (0, 100, 255), 2)

        elif has_any:
            fit = left_fit if left_fit is not None else right_fit
            fitx = np.clip(np.polyval(fit, ploty), 0, self.W - 1)
            pts = np.int32(np.column_stack((fitx, ploty)))
            color = (255, 100, 0) if left_fit is not None else (0, 100, 255)
            cv2.polylines(color_warp, [pts], False, color, 3)

        # 2. Warp overlay back to camera perspective
        newwarp = cv2.warpPerspective(
            color_warp, self.Minv, (self.W, self.H)
        )
        result = cv2.addWeighted(original, 1, newwarp, 0.45, 0)

        # 3. Draw BEV center reference line
        cv2.line(result,
                 (self.W // 2, self.H),
                 (self.W // 2, int(self.H * 0.7)),
                 (255, 255, 0), 2)

        # 4. HUD Info Box
        cv2.rectangle(result, (10, 10), (380, 180), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (380, 180), (100, 100, 100), 1)

        cv2.putText(result, "QCar2 Lane Detection",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Offset
        offset_color = (0, 255, 0) if abs(self.lateral_offset_m) < 0.05 else \
                       (0, 255, 255) if abs(self.lateral_offset_m) < 0.10 else (0, 0, 255)
        cv2.putText(result, f"Offset: {self.lateral_offset_m:+.3f} m",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)

        # Curvature
        curv_text = f"Curvature: {self.curvature_radius_m:.1f} m" \
            if self.curvature_radius_m < 100 else "Curvature: Straight"
        cv2.putText(result, curv_text,
                    (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Confidence
        conf_color = (0, 255, 0) if self.confidence > 0.5 else (0, 0, 255)
        cv2.putText(result, f"Confidence: {self.confidence:.0%}",
                    (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

        # Lane pixel counts
        lane_status = "BOTH" if has_both else ("LEFT" if left_fit is not None else
                      ("RIGHT" if right_fit is not None else "NONE"))
        status_color = (0, 255, 0) if has_both else (0, 0, 255)
        cv2.putText(result, f"Lanes: {lane_status}  L:{self.left_pixel_count} R:{self.right_pixel_count}",
                    (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # 5. Picture-in-Picture panels (top-right)
        pip_w, pip_h = 200, 100

        # Binary mask panel
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        resized_bin = cv2.resize(binary_color, (pip_w, pip_h))

        # Sliding window debug panel
        resized_sw = cv2.resize(sw_debug, (pip_w, pip_h))

        margin_x = self.W - pip_w - 10
        y1 = 10
        y2 = y1 + pip_h + 5

        result[y1:y1 + pip_h, margin_x:margin_x + pip_w] = resized_sw
        result[y2:y2 + pip_h, margin_x:margin_x + pip_w] = resized_bin

        cv2.rectangle(result, (margin_x, y1),
                      (margin_x + pip_w, y1 + pip_h), (255, 255, 255), 1)
        cv2.putText(result, "Sliding Window",
                    (margin_x + 5, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.rectangle(result, (margin_x, y2),
                      (margin_x + pip_w, y2 + pip_h), (255, 255, 255), 1)
        cv2.putText(result, "Edge Mask",
                    (margin_x + 5, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return result
