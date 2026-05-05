"""
QCar2 LiDAR Radar Visualization
===============================

Features:
1. Concentric Distance Circles with labels (1m intervals by default).
2. Real-Time Point Plotting using optimized pyqtgraph ScatterPlotItem.
3. Coordinate Display for Each Point (subsampled to ensure performance when toggled on).
4. Clean Radar-Style UI with a dark background, cyan high-contrast points, and crosshairs.
5. QCar2 Compatibility - designed with a mock layer that is easily swappable with actual QCar2 streams.
6. Highlights closest obstacle and allows threshold distance filtering.

Dependencies:
    pip install pyqtgraph PyQt5 numpy
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QHBoxLayout, QLabel, QCheckBox, QDoubleSpinBox, QSlider)
from PyQt5.QtCore import QTimer, Qt

# Auto-detect QCar2 hardware libraries
try:
    from pal.products.qcar import QCarLidar
    HAS_HARDWARE = True
except ImportError:
    print("Warning: Quanser 'pal.products.qcar' library not found. Running in simulation/mock mode.")
    HAS_HARDWARE = False


class RadarPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCar2 LiDAR Radar System")
        self.resize(1000, 850)
        
        # Main widget & layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # ------------ Top Control Panel ------------
        self.control_layout = QHBoxLayout()
        self.layout.addLayout(self.control_layout)
        
        # 1. Toggle Coordinate Texts
        self.toggle_text_btn = QCheckBox("Show Coordinates for Points")
        self.toggle_text_btn.setStyleSheet("color: white; font-weight: bold;")
        self.toggle_text_btn.setChecked(False)
        self.control_layout.addWidget(self.toggle_text_btn)
        
        self.control_layout.addSpacing(20)
        
        # 2. Distance Threshold Filter
        self.threshold_lbl = QLabel("Max Filter Distance (m):")
        self.threshold_lbl.setStyleSheet("color: white;")
        self.control_layout.addWidget(self.threshold_lbl)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 30.0)
        self.threshold_spin.setValue(10.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        self.control_layout.addWidget(self.threshold_spin)
        
        self.control_layout.addStretch()
        
        # Status Label
        self.status_lbl = QLabel("Mode: 🔴 Simulation" if not HAS_HARDWARE else "Mode: 🟢 Live Sensor")
        self.status_lbl.setStyleSheet("color: #aaa; font-style: italic;")
        self.control_layout.addWidget(self.status_lbl)
        
        # ------------ Graphics Layout ------------
        # Feature 4: Clean Radar-Style UI
        pg.setConfigOption('background', (10, 15, 20)) # Dark UI
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOptions(antialias=True)
        
        self.glw = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.glw)
        
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(True) # Keep 1:1 aspect ratio for accurate radar
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setMouseEnabled(x=True, y=True)
        
        # ------------ Styling Colors ------------
        self.radar_color = (0, 255, 255, 220)        # Bright Cyan points
        self.highlight_color = (255, 50, 50, 255)    # Red for closest obstacle
        self.grid_color = (40, 60, 80, 150)          # Subtle blue-grey grid
        self.text_color = (130, 150, 170, 200)       # Muted text for coordinates
        
        # Draw Radar Background (Feature 1: Concentric Distance Circles)
        self.max_radius = 10.0
        self.ring_step = 1.0 # Interval for rings (1m)
        self.plot.setXRange(-self.max_radius, self.max_radius)
        self.plot.setYRange(-self.max_radius, self.max_radius)
        self.draw_radar_background()
        
        # Feature 2: Scatter plot for LiDAR points (Real-Time Point Plotting)
        self.scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(*self.radar_color))
        self.plot.addItem(self.scatter)
        
        # Highlighted closest point scatter plot
        self.closest_pt_scatter = pg.ScatterPlotItem(size=12, symbol='star', pen=pg.mkPen('w', width=1), brush=pg.mkBrush(*self.highlight_color))
        self.plot.addItem(self.closest_pt_scatter)
        
        # Container for coordinate labels
        self.text_items = []
        self.closest_text = None
        
        # Setup continuous update timer (approx 30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(33) 
        
        # Dark styling for the main window frame
        self.setStyleSheet("QMainWindow { background-color: #1a1a1a; }")

        # Hardware Initialization
        if HAS_HARDWARE:
            self.init_hardware()
            
    def init_hardware(self):
        """
        Initialize the QCar2 LiDAR hardware here using Quanser APIs.
        """
        global HAS_HARDWARE
        if HAS_HARDWARE:
            try:
                # Optimized QCarLidar Settings
                numMeasurements = 720
                lidarMeasurementMode = 2
                lidarInterpolationMode = 0
                
                # Initialize Quanser QCarLidar
                self.myLidar = QCarLidar(
                    numMeasurements=numMeasurements,
                    rangingDistanceMode=lidarMeasurementMode,
                    interpolationMode=lidarInterpolationMode
                )
                print("Successfully initialized QCar LiDAR.")
            except Exception as e:
                print(f"Failed to initialize LiDAR: {e}")
                HAS_HARDWARE = False
        
    def draw_radar_background(self):
        """ Draws the concentric distance circles and crosshair axes """
        # Crosshairs
        v_line = pg.PlotCurveItem([0, 0], [-self.max_radius*1.5, self.max_radius*1.5], pen=pg.mkPen(self.grid_color, width=1))
        h_line = pg.PlotCurveItem([-self.max_radius*1.5, self.max_radius*1.5], [0, 0], pen=pg.mkPen(self.grid_color, width=1))
        self.plot.addItem(v_line)
        self.plot.addItem(h_line)
        
        # Concentric distance circles
        theta = np.linspace(0, 2*np.pi, 200)
        for r in np.arange(self.ring_step, self.max_radius + self.ring_step, self.ring_step):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            circle = pg.PlotCurveItem(x, y, pen=pg.mkPen(self.grid_color, width=1, style=Qt.DashLine))
            self.plot.addItem(circle)
            
            # Feature 6: Label distance values on concentric circles
            label = pg.TextItem(f"{r:.1f}m", color=self.text_color, anchor=(0, 1))
            label.setPos(r, 0)
            font = label.textItem.font()
            font.setPointSize(8)
            label.setFont(font)
            self.plot.addItem(label)
            
    def get_lidar_data(self):
        """
        Retrieves angles (radians) and distances (meters) from the LiDAR.
        """
        if HAS_HARDWARE:
            try:
                self.myLidar.read()
                # Quanser lidar might return empty arrays if no new scan is ready
                if len(self.myLidar.distances) > 0:
                    return np.array(self.myLidar.angles), np.array(self.myLidar.distances)
            except Exception as e:
                print(f"LiDAR read error: {e}")
                pass
            
        # --- MOCK DATA GENERATION FOR TESTING (Fallback) ---
        num_points = 720
        angles = np.linspace(0, 2*np.pi, num_points)
        
        distances = 6.0 + 1.5 * np.sin(angles * 4) + np.random.normal(0, 0.05, num_points)
        
        t = (time.time() % 10) / 10.0
        obs_angle = t * 2 * np.pi
        angle_diff = (angles - obs_angle + np.pi) % (2*np.pi) - np.pi
        obstacle_mask = np.abs(angle_diff) < 0.15
        distances[obstacle_mask] = 2.0 + np.random.normal(0, 0.05, np.sum(obstacle_mask))
        
        return angles, distances
        
    def update_data(self):
        """ Core update loop triggered by the QTimer """
        angles, distances = self.get_lidar_data()
        
        # Distance Filter (Feature 6 enhancement)
        max_dist = self.threshold_spin.value()
        
        # Ensure distances are valid and within threshold
        valid_mask = (distances > 0.05) & (distances <= max_dist)
        angles = angles[valid_mask]
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            self.scatter.clear()
            self.closest_pt_scatter.clear()
            self.clear_texts()
            if self.closest_text:
                self.plot.removeItem(self.closest_text)
                self.closest_text = None
            return
            
        # Convert polar to Cartesian for display (x, y)
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        
        # Update main Point Cloud Scatter
        self.scatter.setData(x, y)
        
        # Highlight closest obstacle
        closest_idx = np.argmin(distances)
        cx, cy = x[closest_idx], y[closest_idx]
        cd = distances[closest_idx]
        ca = angles[closest_idx]
        
        self.closest_pt_scatter.setData([cx], [cy])
        
        # Manage text items cleanly
        if self.closest_text is not None:
            self.plot.removeItem(self.closest_text)
            
        # Always display coordinate for the closest point
        self.closest_text = pg.TextItem(
            f"⚠️ Closest: {cd:.2f}m\n(x: {cx:.2f}, y: {cy:.2f})", 
            color=self.highlight_color, 
            anchor=(-0.1, 1.1)
        )
        self.closest_text.setPos(cx, cy)
        font = self.closest_text.textItem.font()
        font.setPointSize(10)
        font.setBold(True)
        self.closest_text.setFont(font)
        self.plot.addItem(self.closest_text)
        
        # Feature 3: Coordinate Display for Points
        self.clear_texts()
        
        # Only render labels if toggled (for massive performance tracking)
        if self.toggle_text_btn.isChecked():
            # Decimation logic: displaying thousands of text nodes drops FPS.
            # We subsample to show around 40 labels spread across the scan.
            step = max(1, len(x) // 40)
            for i in range(0, len(x), step):
                if i == closest_idx: continue
                txt = pg.TextItem(f"({x[i]:.1f}, {y[i]:.1f})", color=self.text_color, anchor=(0.5, 1.5))
                txt.setPos(x[i], y[i])
                font = txt.textItem.font()
                font.setPointSize(7)
                txt.setFont(font)
                self.plot.addItem(txt)
                self.text_items.append(txt)
                
    def clear_texts(self):
        """ Removes old text items to prevent memory / visual overlap issues """
        for txt in self.text_items:
            self.plot.removeItem(txt)
        self.text_items.clear()


if __name__ == '__main__':
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Gives an extra polished look
    
    window = RadarPlot()
    window.show()
    sys.exit(app.exec_())
