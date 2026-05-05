import numpy as np
import pyrealsense2 as rs

class LaneFilter:
    def __init__(self, camera_height=0.15, camera_angle=20):
        # camera_height: Rough mounting height in meters
        # camera_angle: Angle down from horizontal in degrees
        self.camera_height = camera_height
        self.camera_angle_rad = np.radians(camera_angle)

    def filter_road_points(self, points_vtx, z_min=0.2, z_max=1.0, y_threshold=None):
        """
        Filter points to isolate the road.
        points_vtx: Nx3 numpy array (X, Y, Z) from get_vertices()
        z_min/max: Distance range ahead of the car (meters)
        y_threshold: Approximate Y (downwards) in camera space that corresponds to the ground
        """
        # 1. Basic distance filter
        mask = (points_vtx[:, 2] > z_min) & (points_vtx[:, 2] < z_max)
        
        # 2. Height filter (Y in RealSense camera coordinate system)
        # Assuming camera is tilted down, "down" points have positive Y.
        # The ground is usually at a specific Y range related to Z.
        # Simple heuristic: filter by Y > threshold if camera is mounted high
        if y_threshold is not None:
            mask &= (points_vtx[:, 1] > y_threshold)

        return points_vtx[mask]

    def extract_lane_candidates(self, points_vtx, color_data):
        """
        Filter by color (whitish/yellowish) for lane markings.
        color_data: RGB values for each point
        """
        # 1. Normalize color to [0, 1]
        colors = color_data.astype(float) / 255.0
        
        # 2. Heuristic for "white" (high R, G, B)
        is_white = (colors[:, 0] > 0.8) & (colors[:, 1] > 0.8) & (colors[:, 2] > 0.8)
        
        # 3. Heuristic for "yellow" (high R, G, lower B)
        is_yellow = (colors[:, 0] > 0.8) & (colors[:, 1] > 0.8) & (colors[:, 2] < 0.6)
        
        mask = is_white | is_yellow
        return points_vtx[mask], color_data[mask]

def get_vertices_array(points):
    """Utility to convert RealSense points object to numpy array."""
    vtx = np.asanyarray(points.get_vertices())
    return vtx.view(np.float32).reshape(-1, 3)

def get_color_array(color_frame):
    """Utility to get flat RGB array from color frame."""
    color_image = np.asanyarray(color_frame.get_data())
    return color_image.reshape(-1, 3)

