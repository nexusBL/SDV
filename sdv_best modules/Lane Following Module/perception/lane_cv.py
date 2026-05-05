"""
lane_cv.py - Lane Detection Pipeline for QCar2
================================================
Matches the EXACT pipeline from the original lines.py:
  1. Resize frame by 0.5x (1640x820 → 820x410)
  2. Bird's Eye View perspective transform
  3. HSV thresholding with adaptive brightness compensation
  4. Morphological cleanup (close + open)
  5. Histogram-based lane base detection
  6. Sliding window search (10 windows, margin=40, minpix=30)
  7. RANSAC polynomial fitting (10 samples, 2.0 threshold, 100 trials)
  8. Error = center_cam - center_lines (center_cam = width//2 + 22)
  9. Temporal smoothing over 5 frames
  10. Inverse-perspective overlay on original image
"""

import cv2
import numpy as np
import collections
import time


class LaneDetector:
    """
    Direct port of the original LaneDetect class from lines.py,
    restructured into a clean pipeline with all original values preserved.
    """

    def __init__(self, config):
        self.cfg = config

        # Precompute perspective matrices (for 820x410 processing resolution)
        self.M = cv2.getPerspectiveTransform(
            self.cfg.cv.src_points, self.cfg.cv.dst_points
        )
        self.Minv = cv2.getPerspectiveTransform(
            self.cfg.cv.dst_points, self.cfg.cv.src_points
        )

        # Morphological kernel (from original: np.ones((3,3), np.uint8))
        self.morph_kernel = np.ones((3, 3), np.uint8)

        # Polynomial state (from original lines.py)
        self.polyright = [0, 0, 0]
        self.polyleft = [0, 0, 0]
        self.polyleft_last = [0, 0, 0]
        self.polyright_last = [0, 0, 0]

        # Points collected by sliding window
        self.left_points = []
        self.right_points = []

        # Saved base positions for fallback (from original locate_lanes)
        self.left_base_pos = None
        self.right_base_pos = None

        # Error and smoothing (from original)
        self.error = 0
        self.error_history = []

        # Processing time
        self.process_time = 0

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Main entry. Matches original find_lines() exactly.

        Args:
            frame: BGR image from camera (1640x820 or any size)

        Returns:
            tuple: (error: float, result_frame: ndarray)
        """
        total_start = time.time()

        # Step 1: Transform (resize + perspective + threshold)
        original_img, resize_frame, binary_image = self._transform_image(frame)

        # Step 2: Locate lanes via sliding windows
        self._locate_lanes(binary_image)

        # Step 3: Fit polynomials + draw lines + compute error
        result = self._draw_lines(resize_frame, original_img)

        self.process_time = time.time() - total_start
        return self.error, result

    # ──────────────────────────────────────────────────────────────
    # STEP 1: IMAGE TRANSFORM  (from original TransformImage)
    # ──────────────────────────────────────────────────────────────

    def _transform_image(self, frame):
        """
        Exact replica of original TransformImage():
        1. Resize by 0.5x
        2. Perspective warp
        3. HSV threshold with adaptive bounds
        4. Morphological close + open
        """
        # Resize to 820x410 (original: cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        original_img = resize_frame.copy()

        # Perspective transform
        resize_frame = cv2.warpPerspective(
            resize_frame, self.M,
            (resize_frame.shape[1], resize_frame.shape[0])
        )

        # Adaptive HSV thresholding (matches original adaptive_threshold)
        lower, upper = self._adaptive_threshold(resize_frame)
        hsv_buf = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)
        binary_image = cv2.inRange(hsv_buf, lower, upper)

        # Morphological cleanup (from original: close then open)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, self.morph_kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.morph_kernel)

        return original_img, resize_frame, binary_image

    def _adaptive_threshold(self, frame):
        """
        From original adaptive_threshold().
        Adjusts HSV bounds based on mean brightness.
        Uses TEMPORARY copies — never mutates the config baseline.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Start from config baselines (copies, not references)
        lower = self.cfg.cv.hsv_lower.copy()
        upper = self.cfg.cv.hsv_upper.copy()

        if mean_brightness < self.cfg.cv.dark_brightness_threshold:
            # Dark: relax value and saturation (from original)
            lower[2] = max(50, lower[2] - 10)    # min_value
            lower[1] = max(30, lower[1] - 10)    # min_saturation
        elif mean_brightness > self.cfg.cv.bright_brightness_threshold:
            # Bright: tighten (from original)
            lower[2] = min(150, lower[2] + 10)   # min_value
            lower[1] = min(80, lower[1] + 10)    # min_saturation

        return lower, upper

    # ──────────────────────────────────────────────────────────────
    # STEP 2: SLIDING WINDOW  (from original locate_lanes + histogram)
    # ──────────────────────────────────────────────────────────────

    def _histogram(self, binary_image):
        """
        Exact replica of original histogram().
        Uses bottom half of image to find lane base x-positions.
        """
        init_row = binary_image.shape[0] // 2
        end_row = binary_image.shape[0] - 1
        roi = binary_image[init_row:end_row, :]

        histogram = np.sum(roi, axis=0)

        midpoint = binary_image.shape[1] // 2
        left_ptr = np.argmax(histogram[:midpoint])
        right_ptr = np.argmax(histogram[midpoint:]) + midpoint

        # Validate peaks (from original: histogram[x] < 50 → invalid)
        if histogram[left_ptr] < self.cfg.cv.histogram_min_peak:
            left_ptr = -1
        if histogram[right_ptr] < self.cfg.cv.histogram_min_peak:
            right_ptr = -1

        return np.array([left_ptr, right_ptr], dtype=int)

    def _locate_lanes(self, img):
        """
        Exact replica of original locate_lanes().
        Sliding window search with fallback to saved base positions.
        """
        start_time = time.time()

        self.left_points = []
        self.right_points = []

        nwindows = self.cfg.cv.n_windows      # 10
        margin = self.cfg.cv.margin            # 40
        minpix = self.cfg.cv.min_pixels        # 30

        # Get initial positions from histogram
        lane_positions = self._histogram(img)
        leftx_current = lane_positions[0]
        rightx_current = lane_positions[1]

        # Fallback to saved positions if histogram failed (from original)
        if leftx_current == -1 and self.left_base_pos is not None:
            leftx_current = self.left_base_pos
        elif leftx_current != -1:
            self.left_base_pos = leftx_current

        if rightx_current == -1 and self.right_base_pos is not None:
            rightx_current = self.right_base_pos
        elif rightx_current != -1:
            self.right_base_pos = rightx_current

        window_height = img.shape[0] // nwindows

        # Find nonzero pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Iterate windows bottom-to-top (from original)
        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height

            # Left window bounds (clamped, from original)
            win_xleft_low = int(max(0, leftx_current - margin))
            win_xleft_high = int(min(img.shape[1] - 1, leftx_current + margin))
            win_xright_low = int(max(0, rightx_current - margin))
            win_xright_high = int(min(img.shape[1] - 1, rightx_current + margin))

            # Find pixels in left window
            left_lane_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            # Find pixels in right window
            right_lane_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Store points and recenter (from original)
            if len(left_lane_inds) > 0:
                for i in left_lane_inds:
                    self.left_points.append((nonzeroy[i], nonzerox[i]))
                leftx_current = int(np.mean(nonzerox[left_lane_inds]))

            if len(right_lane_inds) > 0:
                for i in right_lane_inds:
                    self.right_points.append((nonzeroy[i], nonzerox[i]))
                rightx_current = int(np.mean(nonzerox[right_lane_inds]))

        self.process_time = time.time() - start_time

    # ──────────────────────────────────────────────────────────────
    # STEP 3: RANSAC POLYFIT  (from original custom_ransac_fit)
    # ──────────────────────────────────────────────────────────────

    def _custom_ransac_fit(self, points, degree=2):
        """
        Exact replica of original custom_ransac_fit().
        Returns (coefficients, success_bool).
        """
        if len(points) < self.cfg.cv.ransac_min_samples:
            return None, False

        x_values = np.array([p[0] for p in points])
        y_values = np.array([p[1] for p in points])

        best_coeffs = None
        best_inlier_count = 0

        for _ in range(self.cfg.cv.ransac_max_trials):
            sample_indices = np.random.choice(
                len(points), self.cfg.cv.ransac_min_samples, replace=False
            )
            sample_x = x_values[sample_indices]
            sample_y = y_values[sample_indices]

            try:
                coeffs = np.polyfit(sample_x, sample_y, degree)
                y_pred = np.polyval(coeffs, x_values)
                errors = np.abs(y_values - y_pred)
                inliers = errors < self.cfg.cv.ransac_residual_threshold
                inlier_count = np.sum(inliers)

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_coeffs = coeffs

                    # Early exit at 80% consensus (from original)
                    if inlier_count > len(points) * 0.8:
                        break
            except:
                continue

        # Refine with all inliers (from original)
        if best_coeffs is not None and best_inlier_count >= self.cfg.cv.ransac_min_samples:
            y_pred = np.polyval(best_coeffs, x_values)
            errors = np.abs(y_values - y_pred)
            inliers = errors < self.cfg.cv.ransac_residual_threshold

            if np.sum(inliers) >= self.cfg.cv.ransac_min_samples:
                try:
                    final_coeffs = np.polyfit(x_values[inliers], y_values[inliers], degree)
                    return final_coeffs, True
                except:
                    pass
            return best_coeffs, True

        # Fallback: plain polyfit (from original)
        try:
            coeffs = np.polyfit(x_values, y_values, degree)
            return coeffs, True
        except:
            return None, False

    def _regression_right(self):
        """From original regression_right()."""
        coeffs, success = self._custom_ransac_fit(self.right_points)
        if success:
            self.polyright = coeffs.tolist()
            return True
        return False

    def _regression_left(self):
        """From original regression_left()."""
        coeffs, success = self._custom_ransac_fit(self.left_points)
        if success:
            self.polyleft = coeffs.tolist()
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # STEP 4: DRAW LINES + ERROR  (from original draw_lines)
    # ──────────────────────────────────────────────────────────────

    def _calculate_turn_direction(self, error, img_width):
        """From original calculate_turn_direction()."""
        normalized_error = error / (img_width / 2)

        self.error_history.append(normalized_error)
        if len(self.error_history) > self.cfg.cv.smoothing_frames:
            self.error_history.pop(0)

        smoothed_error = np.mean(self.error_history)

        if abs(smoothed_error) < 0.1:
            return "CENTRO", smoothed_error
        elif smoothed_error > 0:
            return "IZQUIERDA", smoothed_error
        else:
            return "DERECHA", smoothed_error

    def _draw_lines(self, img, original_img):
        """
        Exact replica of original draw_lines().
        Computes polynomial visualizations, error, and overlay.
        """
        # center_cam = (img.shape[1] // 2) + 22  (from original)
        center_cam = (img.shape[1] // 2) + self.cfg.cv.center_cam_offset
        center_lines = center_cam  # Default

        find_line_right = self._regression_right()
        find_line_left = self._regression_left()

        # Clear points after regression (from original)
        self.right_points = []
        self.left_points = []

        # ── Case 1: Both lines detected ──
        if find_line_left and find_line_right:
            for row in range(img.shape[0] - 1, -1, -8):
                columnR = (self.polyright[2] +
                           self.polyright[1] * row +
                           self.polyright[0] * (row * row))
                cv2.circle(img, (int(columnR), int(row)), 2, (0, 255, 0), 2)

                columnL = (self.polyleft[2] +
                           self.polyleft[1] * row +
                           self.polyleft[0] * (row * row))
                cv2.circle(img, (int(columnL), int(row)), 2, (0, 255, 0), 2)

            center_lines = (columnR + columnL) / 2

            for k in range(3):
                self.polyleft_last[k] = self.polyleft[k]
                self.polyright_last[k] = self.polyright[k]

        # ── Case 2: Only left line ──
        elif find_line_left:
            for row in range(img.shape[0] - 1, -1, -8):
                columnL = (self.polyleft[2] +
                           self.polyleft[1] * row +
                           self.polyleft[0] * (row * row))
                cv2.circle(img, (int(columnL), int(row)), 2, (0, 255, 0), 2)

            # Estimate center: left line + offset (from original: +125)
            columnL_aux = self.polyleft[2]
            center_lines = columnL_aux + self.cfg.cv.single_lane_offset_px

            for k in range(3):
                self.polyleft_last[k] = self.polyleft[k]

        # ── Case 3: Only right line ──
        elif find_line_right:
            for row in range(img.shape[0] - 1, -1, -8):
                columnR = (self.polyright[2] +
                           self.polyright[1] * row +
                           self.polyright[0] * (row * row))
                cv2.circle(img, (int(columnR), int(row)), 2, (0, 255, 0), 2)

            # Estimate center: right line - offset (from original: -125)
            columnR_aux = self.polyright[2]
            center_lines = columnR_aux - self.cfg.cv.single_lane_offset_px

            for k in range(3):
                self.polyright_last[k] = self.polyright[k]

        # ── Case 4: No lines → use previous ──
        else:
            for row in range(img.shape[0] - 1, -1, -8):
                columnR = (self.polyright_last[2] +
                           self.polyright_last[1] * row +
                           self.polyright_last[0] * (row * row))
                cv2.circle(img, (int(columnR), int(row)), 2, (255, 0, 0), 2)

                columnL = (self.polyleft_last[2] +
                           self.polyleft_last[1] * row +
                           self.polyleft_last[0] * (row * row))
                cv2.circle(img, (int(columnL), int(row)), 2, (255, 0, 0), 2)

            center_lines = (columnR + columnL) / 2

        # Clamp center_lines (from original)
        center_lines = min(max(0, center_lines), img.shape[1] - 1)

        # Calculate error (from original: center_cam - center_lines)
        distance_center = center_cam - center_lines
        self.error = distance_center

        # Turn direction with smoothing (from original)
        turn_direction, smoothed_error = self._calculate_turn_direction(
            distance_center, img.shape[1]
        )

        # Draw reference lines (from original)
        cv2.line(img,
                 (int(center_cam), int(img.shape[0] / 4)),
                 (int(center_cam), int(img.shape[0] * 3 / 4)),
                 (0, 255, 0), 2)
        cv2.line(img,
                 (int(center_lines), 0),
                 (int(center_cam), int(img.shape[0] - 1)),
                 (0, 0, 255), 2)

        # HUD text (from original)
        cv2.putText(img, f"Dir: {turn_direction}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Error: {smoothed_error:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Time: {self.process_time*1000:.1f}ms",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Inverse perspective overlay (from original)
        inverted_matrix = np.linalg.inv(self.M)
        warped_frame = cv2.warpPerspective(
            img, inverted_matrix, (img.shape[1], img.shape[0])
        )
        result = cv2.addWeighted(original_img, 0.7, warped_frame, 0.3, 0)

        return result
