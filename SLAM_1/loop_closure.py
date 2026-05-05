"""
Loop Closure Detection using ICP-lite Scan Matching.

Stores a history of LiDAR scans with their poses. When the robot revisits
a previously-seen area (distance threshold), runs iterative-closest-point
matching between scans to compute a pose correction.
"""
import numpy as np
import math
import utils


class ScanRecord:
    """A snapshot of a LiDAR scan with its associated pose."""
    def __init__(self, pose, angles, dists, max_dist, map_units):
        self.pose = pose.copy()
        self.points = self._to_cartesian(pose, angles, dists, max_dist, map_units)

    def _to_cartesian(self, pose, angles, dists, max_dist, map_units):
        """Convert polar LiDAR readings to local Cartesian points (robot frame)."""
        pts = []
        for i in range(len(angles)):
            if dists[i] >= max_dist * map_units or dists[i] < 0.05:
                continue
            theta = pose[2] - angles[i]
            x = dists[i] * np.cos(theta)
            y = dists[i] * np.sin(theta)
            pts.append([x, y])
        return np.array(pts) if pts else np.empty((0, 2))


def _nearest_neighbor(src, dst):
    """For each point in src, find the index of the nearest point in dst."""
    indices = np.zeros(src.shape[0], dtype=int)
    distances = np.zeros(src.shape[0])
    for i in range(src.shape[0]):
        diff = dst - src[i]
        d = np.sum(diff ** 2, axis=1)
        idx = np.argmin(d)
        indices[i] = idx
        distances[i] = d[idx]
    return indices, distances


def _icp_2d(source, target, max_iterations=20, tolerance=0.001):
    """
    Simple 2D ICP (Iterative Closest Point).
    Returns (R, t, mean_error) where R is 2x2 rotation, t is 2x1 translation.
    """
    src = source.copy()
    T_total = np.eye(3)

    for iteration in range(max_iterations):
        indices, distances = _nearest_neighbor(src, target)
        matched_target = target[indices]

        # Compute centroids
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(matched_target, axis=0)

        # Center the points
        src_centered = src - centroid_src
        tgt_centered = matched_target - centroid_tgt

        # SVD for rotation
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - R @ centroid_src

        # Apply transformation  
        src = (R @ src.T).T + t

        # Build homogeneous transform
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
        T_total = T @ T_total

        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    # Extract final rotation angle and translation
    R_final = T_total[:2, :2]
    t_final = T_total[:2, 2]
    theta = np.arctan2(R_final[1, 0], R_final[0, 0])

    return t_final[0], t_final[1], theta, np.mean(distances)


class LoopClosureDetector:
    """
    Detects loop closures by comparing current LiDAR scans against
    historical scans taken at similar poses.
    """
    def __init__(self, distance_threshold=5.0, min_scan_gap=20,
                 icp_error_threshold=2.0, max_dist=2, map_units=20):
        """
        Args:
            distance_threshold: max Euclidean distance to consider revisit (in map units)
            min_scan_gap: minimum number of scans between current and candidate to avoid
                          matching against very recent scans
            icp_error_threshold: maximum ICP error to accept a closure
            max_dist: LiDAR max distance (meters)
            map_units: conversion factor
        """
        self.distance_threshold = distance_threshold
        self.min_scan_gap = min_scan_gap
        self.icp_error_threshold = icp_error_threshold
        self.max_dist = max_dist
        self.map_units = map_units
        self.scan_history = []
        self.scan_count = 0
        self.last_closure_scan = -self.min_scan_gap  # allow immediate first closure

    def add_scan(self, pose, angles, dists):
        """Record a new scan with its pose."""
        record = ScanRecord(pose, angles, dists, self.max_dist, self.map_units)
        self.scan_history.append(record)
        self.scan_count += 1

    def detect(self, current_pose, angles, dists):
        """
        Check if current scan matches any historical scan (loop closure).

        Returns:
            (dx, dy, dtheta) correction if closure detected, None otherwise.
        """
        if self.scan_count - self.last_closure_scan < self.min_scan_gap:
            return None

        current_record = ScanRecord(current_pose, angles, dists, self.max_dist, self.map_units)

        if current_record.points.shape[0] < 10:
            return None

        best_correction = None
        best_error = self.icp_error_threshold

        # Search historical scans for candidates
        for i, historical in enumerate(self.scan_history):
            # Skip recent scans
            if self.scan_count - i < self.min_scan_gap:
                continue

            # Check if poses are close enough to be a revisit
            pose_dist = math.sqrt(
                (current_pose[0] - historical.pose[0]) ** 2 +
                (current_pose[1] - historical.pose[1]) ** 2
            )
            if pose_dist > self.distance_threshold:
                continue

            if historical.points.shape[0] < 10:
                continue

            # Run ICP between current and historical scan
            try:
                dx, dy, dtheta, error = _icp_2d(
                    current_record.points, historical.points,
                    max_iterations=30, tolerance=0.5
                )
            except Exception:
                continue

            if error < best_error:
                best_error = error
                best_correction = (dx, dy, dtheta)

        if best_correction is not None:
            self.last_closure_scan = self.scan_count
            print(f"[LoopClosure] Detected! correction=({best_correction[0]:.2f}, "
                  f"{best_correction[1]:.2f}, {np.degrees(best_correction[2]):.1f}°) "
                  f"error={best_error:.3f}")

        return best_correction

    def apply_correction(self, particle_filter, correction):
        """
        Apply loop closure correction to the best particle's pose.
        
        Args:
            particle_filter: ParticleFilter instance
            correction: (dx, dy, dtheta) from detect()
        """
        dx, dy, dtheta = correction
        best = particle_filter.get_best_particle()
        best.pos[0] += dx
        best.pos[1] += dy
        best.pos[2] = utils.radsLimit(best.pos[2] + dtheta)
        print(f"[LoopClosure] Applied correction to best particle: {best.pos}")
