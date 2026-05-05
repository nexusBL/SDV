
"""

lane_detector_simple.py — Simple Lane Detection Module

=======================================================

Classical computer vision approach using:

1. ROI masking (trapezoid region)

2. Edge detection (Canny)

3. Line detection (Hough Transform)

4. Line filtering and averaging

5. Lane center calculation



No neural networks, runs fast on Jetson Nano.

"""



import cv2

import numpy as np

import config_lane as cfg





class SimpleLaneDetector:

    """

    Detects lane lines using classical computer vision.

    Returns lane center position for steering control.

    """

    

    def __init__(self):

        """Initialize lane detector with configuration parameters."""

        self.width = cfg.CAMERA_WIDTH

        self.height = cfg.CAMERA_HEIGHT

        

        # Calculate ROI polygon points from config (in pixel coordinates)

        self.roi_vertices = self._calculate_roi_vertices()

        

        # Image center (where we want the lane center to be)

        self.image_center_x = self.width // 2

        

        # Lane detection stats

        self.frames_processed = 0

        self.lanes_detected = 0

    

    def _calculate_roi_vertices(self):

        """

        Calculate ROI polygon vertices from config fractions.

        Returns 4-point trapezoid: [top-left, top-right, bottom-right, bottom-left]

        """

        w, h = self.width, self.height

        

        # Convert fractional coordinates to pixel coordinates

        vertices = np.array([[

            (int(w * cfg.ROI_TOP_LEFT_X), int(h * cfg.ROI_TOP_Y)),      # Top-left

            (int(w * cfg.ROI_TOP_RIGHT_X), int(h * cfg.ROI_TOP_Y)),     # Top-right

            (int(w * cfg.ROI_BOTTOM_RIGHT_X), int(h * cfg.ROI_BOTTOM_Y)), # Bottom-right

            (int(w * cfg.ROI_BOTTOM_LEFT_X), int(h * cfg.ROI_BOTTOM_Y))  # Bottom-left

        ]], dtype=np.int32)

        

        return vertices

    

    def process_frame(self, frame):

        """

        Main processing pipeline. Takes BGR frame, returns:

        - lane_center_x: X-coordinate of detected lane center (or None)

        - deviation: normalized deviation from image center (-1.0 to +1.0)

        - confidence: detection confidence (0.0 to 1.0)

        - debug_frame: annotated frame for visualization

        

        Returns: (lane_center_x, deviation, confidence, debug_frame)

        """

        self.frames_processed += 1

        

        # Make copy for debugging

        debug_frame = frame.copy()

        

        # Step 1: Convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        

        # Step 2: Apply Gaussian blur to reduce noise

        blurred = cv2.GaussianBlur(gray, (cfg.BLUR_KERNEL_SIZE, cfg.BLUR_KERNEL_SIZE), 0)

        

        # Step 3: Canny edge detection

        edges = cv2.Canny(blurred, cfg.CANNY_LOW_THRESHOLD, cfg.CANNY_HIGH_THRESHOLD)

        

        # Step 4: Apply ROI mask

        masked_edges = self._apply_roi_mask(edges)

        

        # Step 5: Detect lines using Hough Transform

        lines = cv2.HoughLinesP(

            masked_edges,

            rho=cfg.HOUGH_RHO,

            theta=np.pi / cfg.HOUGH_THETA_FACTOR,

            threshold=cfg.HOUGH_THRESHOLD,

            minLineLength=cfg.HOUGH_MIN_LINE_LENGTH,

            maxLineGap=cfg.HOUGH_MAX_LINE_GAP

        )

        

        # Step 6: Filter and separate left/right lane lines

        left_lines, right_lines = self._filter_lines(lines)

        

        # Step 7: Average lines to get single left and right lane

        left_lane = self._average_lines(left_lines)

        right_lane = self._average_lines(right_lines)

        

        # Step 8: Calculate lane center and deviation

        lane_center_x, deviation, confidence = self._calculate_lane_center(left_lane, right_lane)

        

        # Step 9: Draw visualization

        self._draw_debug_info(debug_frame, left_lane, right_lane, lane_center_x, 

                              left_lines, right_lines, confidence)

        

        # Update stats

        if lane_center_x is not None:

            self.lanes_detected += 1

        

        return lane_center_x, deviation, confidence, debug_frame

    

    def _apply_roi_mask(self, edges):

        """Apply ROI mask to edge image."""

        mask = np.zeros_like(edges)

        cv2.fillPoly(mask, self.roi_vertices, 255)

        masked = cv2.bitwise_and(edges, mask)

        return masked

    

    def _filter_lines(self, lines):

        """

        Filter detected lines into left and right lane lines.

        Left lanes have negative slope, right lanes have positive slope.

        """

        left_lines = []

        right_lines = []

        

        if lines is None:

            return left_lines, right_lines

        

        for line in lines:

            x1, y1, x2, y2 = line[0]

            

            # Skip vertical lines (avoid division by zero)

            if x2 - x1 == 0:

                continue

            

            # Calculate slope

            slope = (y2 - y1) / (x2 - x1)

            

            # Filter by slope magnitude

            if abs(slope) < cfg.MIN_LANE_SLOPE or abs(slope) > cfg.MAX_LANE_SLOPE:

                continue

            

            # Separate left (negative slope) and right (positive slope)

            if slope < 0:

                left_lines.append(line[0])

            else:

                right_lines.append(line[0])

        

        return left_lines, right_lines

    

    def _average_lines(self, lines):

        """

        Average multiple line segments into a single line.

        Returns (x1, y1, x2, y2) or None if no lines.

        """

        if len(lines) == 0:

            return None

        

        # Collect all points

        x_coords = []

        y_coords = []

        

        for x1, y1, x2, y2 in lines:

            x_coords.extend([x1, x2])

            y_coords.extend([y1, y2])

        

        # Fit a line using least squares

        if len(x_coords) < 2:

            return None

        

        # Use polyfit to get line equation: y = mx + b

        try:

            coefficients = np.polyfit(x_coords, y_coords, 1)

            slope, intercept = coefficients

        except np.linalg.LinAlgError:

            return None

        

        # Calculate line endpoints within ROI

        # Use ROI bottom and top Y coordinates

        y1 = int(self.height * cfg.ROI_BOTTOM_Y)

        y2 = int(self.height * cfg.ROI_TOP_Y)

        

        # Calculate corresponding X coordinates

        if slope == 0:

            return None

        

        x1 = int((y1 - intercept) / slope)

        x2 = int((y2 - intercept) / slope)

        

        return (x1, y1, x2, y2)

    

    def _calculate_lane_center(self, left_lane, right_lane):

        """

        Calculate the center of the lane and deviation from image center.

        

        Returns:

        - lane_center_x: X-coordinate of lane center (or None)

        - deviation: normalized deviation (-1.0 to +1.0)

        - confidence: detection confidence (0.0 to 1.0)

        """

        # Determine detection quality

        has_left = left_lane is not None

        has_right = right_lane is not None

        

        # Calculate confidence based on what we detected

        if has_left and has_right:

            confidence = 1.0  # Both lanes detected = high confidence

        elif has_left or has_right:

            confidence = 0.6  # One lane detected = medium confidence

        else:

            confidence = 0.0  # No lanes = no confidence

            return None, 0.0, confidence

        

        # Calculate lane center at bottom of ROI (near car)

        y_eval = int(self.height * cfg.ROI_BOTTOM_Y)

        

        if has_left and has_right:

            # Both lanes: center is midpoint

            x1_left, y1_left, x2_left, y2_left = left_lane

            x1_right, y1_right, x2_right, y2_right = right_lane

            

            # Get X coordinates at evaluation height

            left_x = self._interpolate_x(left_lane, y_eval)

            right_x = self._interpolate_x(right_lane, y_eval)

            

            if left_x is None or right_x is None:

                return None, 0.0, 0.0

            

            lane_center_x = (left_x + right_x) // 2

            

        elif has_left:

            # Only left lane: assume lane width and extrapolate

            left_x = self._interpolate_x(left_lane, y_eval)

            if left_x is None:

                return None, 0.0, 0.0

            

            # Assume standard lane width (adjust based on your setup)

            assumed_lane_width = int(self.width * 0.4)  # 40% of image width

            lane_center_x = left_x + assumed_lane_width // 2

            

        else:  # has_right only

            # Only right lane: assume lane width and extrapolate

            right_x = self._interpolate_x(right_lane, y_eval)

            if right_x is None:

                return None, 0.0, 0.0

            

            assumed_lane_width = int(self.width * 0.4)

            lane_center_x = right_x - assumed_lane_width // 2

        

        # Calculate deviation from image center

        # Positive = lane center is to the right (car is left of lane center)

        # Negative = lane center is to the left (car is right of lane center)

        pixel_deviation = lane_center_x - self.image_center_x

        

        # Normalize to -1.0 to +1.0

        max_deviation = self.width / 2

        deviation = np.clip(pixel_deviation / max_deviation, -1.0, 1.0)

        

        return lane_center_x, deviation, confidence

    

    def _interpolate_x(self, line, y):

        """

        Given a line (x1, y1, x2, y2) and a Y coordinate,

        return the corresponding X coordinate.

        """

        if line is None:

            return None

        

        x1, y1, x2, y2 = line

        

        # Avoid division by zero

        if y2 - y1 == 0:

            return x1

        

        # Linear interpolation

        slope = (x2 - x1) / (y2 - y1)

        x = x1 + slope * (y - y1)

        

        return int(x)

    

    def _draw_debug_info(self, frame, left_lane, right_lane, lane_center_x,

                        left_lines, right_lines, confidence):

        """Draw visualization overlays on debug frame."""

        

        # Draw ROI polygon

        cv2.polylines(frame, self.roi_vertices, isClosed=True, 

                     color=cfg.COLOR_ROI, thickness=2)

        

        # Draw all detected line segments (faint)

        if left_lines:

            for x1, y1, x2, y2 in left_lines:

                cv2.line(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)

        

        if right_lines:

            for x1, y1, x2, y2 in right_lines:

                cv2.line(frame, (x1, y1), (x2, y2), (255, 100, 100), 1)

        

        # Draw averaged lane lines (bright)

        if left_lane is not None:

            x1, y1, x2, y2 = left_lane

            cv2.line(frame, (x1, y1), (x2, y2), cfg.COLOR_LANE_LINES, 

                    cfg.LINE_THICKNESS)

        

        if right_lane is not None:

            x1, y1, x2, y2 = right_lane

            cv2.line(frame, (x1, y1), (x2, y2), cfg.COLOR_LANE_LINES, 

                    cfg.LINE_THICKNESS)

        

        # Draw lane center line (if detected)

        if lane_center_x is not None:

            y_top = int(self.height * cfg.ROI_TOP_Y)

            y_bottom = int(self.height * cfg.ROI_BOTTOM_Y)

            cv2.line(frame, (lane_center_x, y_top), (lane_center_x, y_bottom),

                    cfg.COLOR_LANE_CENTER, cfg.CENTER_LINE_THICKNESS)

        

        # Draw image center reference line

        y_top = int(self.height * cfg.ROI_TOP_Y)

        y_bottom = int(self.height * cfg.ROI_BOTTOM_Y)

        cv2.line(frame, (self.image_center_x, y_top), 

                (self.image_center_x, y_bottom),

                cfg.COLOR_IMAGE_CENTER, cfg.CENTER_LINE_THICKNESS)

        

        # Draw text info

        info_text = [

            f"Confidence: {confidence:.2f}",

            f"Frames: {self.frames_processed}",

            f"Detected: {self.lanes_detected}",

        ]

        

        if lane_center_x is not None:

            deviation_pixels = lane_center_x - self.image_center_x

            info_text.append(f"Deviation: {deviation_pixels:+d}px")

        

        y_offset = 30

        for i, text in enumerate(info_text):

            cv2.putText(frame, text, (10, y_offset + i*25),

                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

    def get_stats(self):

        """Return detection statistics."""

        if self.frames_processed > 0:

            detection_rate = (self.lanes_detected / self.frames_processed) * 100

        else:

            detection_rate = 0.0

        

        return {

            'frames_processed': self.frames_processed,

            'lanes_detected': self.lanes_detected,

            'detection_rate': detection_rate

        }

