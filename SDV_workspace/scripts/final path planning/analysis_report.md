# QCar Codebase Analysis Report

Based on the analysis of the provided repository, here is a complete understanding of the three requested components:

## 1. Lane Detection System

**Image Processing Pipeline:**
- **Capture & Conversion:** The cameras capture physical track images (front, left, right, trailing). For line detection, BGR images are evaluated and generally converted to HSV space or Grayscale. 
- **Color Filtering:** For physical lane detection, it searches for target colors using HSV bounds. White (e.g. `lower=[0,0,200]`, `upper=[180,40,255]`) and Yellow bounds are common. 
- **Noise Reduction:** Uses Gaussian Blurring (e.g., `5x5`) followed by morphology (opening and closing with `5x5` kernels) to filter out spurious pixels.
- **Region of Interest (ROI):** The image is often cropped to just the lower half or bottom fraction (e.g., `BOTTOM_FRAC=0.4`) to ignore background noise and focus purely on the road surface.

**Lane Line Extraction:**
- Several methods are observed across the codebase (`lfDC.py` vs `followLine.py` vs virtual simulation).
- **Column Array Searching (`lfDC.py`):** Converts to a binary mask using thresholding (e.g., 100-130). Evaluates predefined X columns (`cols = [100, 500, 240, 250...]`) and finds the maximum Y index (lowest point in the image) where a white pixel occurs. This determines distance to lane components.
- **Centroid/Contour Extraction (A* Virtual FSM):** Finds contours (`cv2.findContours`) in the masked ROI. Finds the largest contour area, extracting the bottom `20%` band of curve-points. The centroid (`cx`, `cy`) of this contour band defines the lane position. 

**Output Format:**
- Extracts an array of track distances (`maxY[x]`), bounding boxes `[x, y, w, h]`, or explicit centroid coordinates `(cx, cy)` indicating where the lane bounds are relatively placed.

## 2. Lane Following Control

**Control Method:**
- Mostly uses **Proportional (P) Control** for steering and a tuned **Proportional-Integral (PI / PID)** controller for speed regulation.
- Uses finite state machine configurations (e.g. `straight`, `left()`, `rightNoLine()`) for intersection handling.

**Steering Angle Computation:**
- In `lfDC.py`: Analyzes the tracked distances `distance = maxY[index]`. If the distance goes beyond `high` threshold or falls below `low` threshold, it calculates an angle error: `angle = 0.02 * direction * abs(distance - threshold)`.
- In `Virtual FSM`: Compares contour centroid `cx` to a `desired_x` target. `steering = np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP)`.  

**Inputs & Outputs:**
- **Inputs:** Vision processing centroids/distances, threshold values, direction scalar (-1 or 1).
- **Outputs:** Motor speed (`drive_speed`, e.g., 0.066 up to 3.0 virtual bounds) and steering angle (`drive_steer`, clamped between `[-0.5, 0.5]` radians).

## 3. Taxi Bay / Parking / Path Planning Logic

**Path Planning Method:**
- The system uses a Topological Graph and **A* (A-Star) Pathfinding** to calculate the shortest path.
- The map is defined as Nodes (`HUB, A, B, J...`) and weighted Edges indicating physical distances.

**Maneuver Strategy & Logic (Turning/Aligning):**
- **Turn Restrictions:** A constraint dictionary (`TURN_RESTRICTIONS`) restricts invalid physical turns (e.g. 'A' entering 'B' cannot go to 'C'). It factors the `prev_node` in A* states to avoid impossible U-turns or overlapping structures.
- **FSM Navigation:** 
  - `DRIVE`: Runs line follower.
  - `STOPPING`: Driven by visual stop-sign detection (RGB bounding box via contour area matching). The car halts for a configured duration.
  - `TURN_RIGHT` / `TURN_LEFT`: Triggered at intersections. A side CSI camera (e.g. `RIGHT_CAM`) is used to follow an inner curve. The side camera's visual output keeps an offset `TARGET_Y_OFFSET`.
- **Progress Tracking:** In the absence of a topological sensor, edges are navigated via speed-time integration (dead-reckoning). The system transitions to the next edge when the distance threshold of the current edge is satisfied.
