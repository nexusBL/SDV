#!/usr/bin/env python3
"""
reactive_avoidance_controller.py
================================
Hardware-agnostic reactive obstacle avoidance controller for QCar2.

This is adapted from the user's "Production Grade" potential-field controller,
but without PAL/QCar imports so it can be used inside tessting.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ReactiveAvoidanceParams:
    # --- Distance Zones (Meters) ---
    d_safe: float = 1.5
    d_critical: float = 0.6
    d_stop: float = 0.4
    d_resume: float = 0.5  # hysteresis

    # --- Speed Parameters ---
    throttle_cruise: float = 0.12
    throttle_min: float = 0.05
    ttc_threshold: float = 1.5

    # --- Steering Parameters ---
    max_steer: float = 0.5
    steer_p_gain: float = 1.5
    steer_deadzone: float = 0.02
    steer_smooth_alpha: float = 0.3
    throttle_smooth_alpha: float = 0.2

    # --- LiDAR Ranges / Geometry ---
    front_arc_deg: float = 45.0  # match your production code default
    min_lidar_dist: float = 0.05
    max_lidar_dist: float = 5.0
    min_valid_points: int = 5
    lidar_angle_offset_deg: float = 180.0  # 0 should be front after offset
    lidar_reverse: bool = True            # flip if left/right swapped
    
    # --- Gap Analysis Parameters ---
    car_width_m: float = 0.35             # Approximate width of QCar2 with margin
    gap_min_depth_m: float = 0.8          # Only consider gaps at least this deep
    gap_max_angle_deg: float = 60.0       # Max angle to look for gaps



class ReactiveController:
    def __init__(self, cfg: ReactiveAvoidanceParams):
        self.cfg = cfg
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.direction_persistence = 0.0  # -1 left, +1 right
        self.is_emergency_stop = False

    def process_lidar(self, angles_rad, distances_m):
        if distances_m is None or len(distances_m) == 0:
            return None, None

        raw_angles = np.asarray(angles_rad, dtype=float).ravel()
        raw_distances = np.asarray(distances_m, dtype=float).ravel()

        if raw_angles.size == 0 or raw_distances.size == 0:
            return None, None

        # 1) Flip angles if reversed
        if self.cfg.lidar_reverse:
            raw_angles = (2.0 * math.pi - raw_angles) % (2.0 * math.pi)

        # 2) Apply offset so that 0 rad is "front", normalize to [-pi, pi]
        adj_angles = raw_angles + math.radians(float(self.cfg.lidar_angle_offset_deg))
        adj_angles = (adj_angles + math.pi) % (2.0 * math.pi) - math.pi

        mask = (
            (np.abs(np.degrees(adj_angles)) <= float(self.cfg.front_arc_deg))
            & (raw_distances > float(self.cfg.min_lidar_dist))
            & (raw_distances < float(self.cfg.max_lidar_dist))
        )
        return adj_angles[mask], raw_distances[mask]

    def compute_control(self, angles, distances):
        if angles is None or len(angles) < int(self.cfg.min_valid_points):
            target_throttle = float(self.cfg.throttle_min)
            target_steer = 0.0
            return target_throttle, target_steer, "SPARSE_DATA"

        distances = np.asarray(distances, dtype=float)
        angles = np.asarray(angles, dtype=float)
        min_dist = float(np.min(distances))

        # --- NEW: Gap Analysis ---
        best_gap_angle, gap_passable = self.find_best_gap(angles, distances)

        # Emergency stop hysteresis
        if min_dist < float(self.cfg.d_stop):
            self.is_emergency_stop = True
        elif min_dist > float(self.cfg.d_resume):
            self.is_emergency_stop = False

        if self.is_emergency_stop:
            self.prev_throttle = 0.0
            self.prev_steer = 0.0
            return 0.0, 0.0, "EMERGENCY_STOP"

        # Repulsion model
        weights = (np.maximum(0.0, float(self.cfg.d_safe) - distances) ** 2) * np.cos(angles)
        v_repulse_x = float(np.sum(weights * -np.cos(angles)))
        v_repulse_y = float(np.sum(weights * -np.sin(angles)))

        # Lateral imbalance correction
        left_mask = angles > 0.02
        right_mask = angles < -0.02
        sum_left = float(np.sum(weights[left_mask])) if np.any(left_mask) else 0.0
        sum_right = float(np.sum(weights[right_mask])) if np.any(right_mask) else 0.0
        v_repulse_y += 0.3 * (sum_right - sum_left)

        # Forward bias + persistence
        v_total_x = 1.0 + v_repulse_x
        v_total_y = v_repulse_y

        # If a passable gap is found, slightly attract towards its center
        if gap_passable and best_gap_angle is not None:
            gap_weight = 0.4
            v_total_y += gap_weight * math.sin(best_gap_angle)
            v_total_x += gap_weight * math.cos(best_gap_angle)

        if abs(v_total_y) < 0.1 and abs(self.direction_persistence) > 0.1:
            v_total_y += 0.2 * self.direction_persistence

        # Steering mapping
        target_delta = math.atan2(v_total_y, v_total_x)
        speed_factor = 1.0 - (self.prev_throttle / float(self.cfg.throttle_cruise)) * 0.3
        target_steer = float(
            np.clip(
                float(self.cfg.steer_p_gain) * target_delta * speed_factor,
                -float(self.cfg.max_steer),
                float(self.cfg.max_steer),
            )
        )
        if abs(target_steer) > 0.1:
            self.direction_persistence = float(np.sign(target_steer))

        # Adaptive speed zones + TTC
        if min_dist > float(self.cfg.d_safe):
            target_throttle = float(self.cfg.throttle_cruise)
        elif min_dist > float(self.cfg.d_critical):
            ratio = (min_dist - float(self.cfg.d_critical)) / (float(self.cfg.d_safe) - float(self.cfg.d_critical))
            target_throttle = float(self.cfg.throttle_min) + ratio * (
                float(self.cfg.throttle_cruise) - float(self.cfg.throttle_min)
            )
        else:
            target_throttle = float(self.cfg.throttle_min)

        est_v = max(self.prev_throttle * 2.0, 0.05)
        ttc = min_dist / est_v
        if ttc < float(self.cfg.ttc_threshold):
            target_throttle = min(target_throttle, float(self.cfg.throttle_min) * 1.2)

        # Stability: deadzone + LPF
        if abs(target_steer) < float(self.cfg.steer_deadzone):
            target_steer = 0.0

        self.prev_steer = float(self.cfg.steer_smooth_alpha) * target_steer + (1.0 - float(self.cfg.steer_smooth_alpha)) * self.prev_steer
        self.prev_throttle = float(self.cfg.throttle_smooth_alpha) * target_throttle + (1.0 - float(self.cfg.throttle_smooth_alpha)) * self.prev_throttle

        status = f"DIST:{min_dist:.2f}m | GAP:{'OK' if gap_passable else 'NO'} | STEER:{self.prev_steer:.2f} | THR:{self.prev_throttle:.2f}"
        return self.prev_throttle, self.prev_steer, status

    def find_best_gap(self, angles, distances):
        """
        Analyzes the LiDAR scan to find the widest/best passable gap.
        Returns (gap_center_angle, is_passable).
        """
        if len(angles) < 2:
            return None, False

        # Sort by angle
        idx = np.argsort(angles)
        sorted_angles = angles[idx]
        sorted_dists = distances[idx]

        gaps = []
        # Check gaps between consecutive obstacle points
        for i in range(len(sorted_angles) - 1):
            ang1, ang2 = sorted_angles[i], sorted_angles[i+1]
            d1, d2 = sorted_dists[i], sorted_dists[i+1]
            
            # Gap width calculation: W = D * delta_theta
            # Use the closer obstacle as the conservative distance
            d_gap = min(d1, d2)
            if d_gap > self.cfg.max_lidar_dist: continue
            
            delta_theta = ang2 - ang1
            width = d_gap * delta_theta
            
            if width > self.cfg.car_width_m:
                center_ang = (ang1 + ang2) / 2.0
                # Score gap by width and how centered it is
                score = width * math.cos(center_ang) 
                gaps.append((center_ang, width, score))

        # Also check gaps at the edges of the FOV
        # Left edge
        if len(sorted_angles) > 0:
            left_edge = math.radians(self.cfg.gap_max_angle_deg)
            if left_edge > sorted_angles[-1]:
                delta_theta = left_edge - sorted_angles[-1]
                width = sorted_dists[-1] * delta_theta
                if width > self.cfg.car_width_m:
                    center_ang = (sorted_angles[-1] + left_edge) / 2.0
                    gaps.append((center_ang, width, width * math.cos(center_ang)))
            
            # Right edge
            right_edge = -math.radians(self.cfg.gap_max_angle_deg)
            if right_edge < sorted_angles[0]:
                delta_theta = sorted_angles[0] - right_edge
                width = sorted_dists[0] * delta_theta
                if width > self.cfg.car_width_m:
                    center_ang = (sorted_angles[0] + right_edge) / 2.0
                    gaps.append((center_ang, width, width * math.cos(center_ang)))

        if not gaps:
            return None, False

        # Select gap with highest score
        gaps.sort(key=lambda x: x[2], reverse=True)
        return gaps[0][0], True

