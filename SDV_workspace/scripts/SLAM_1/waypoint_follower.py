"""
Waypoint Following with Pure-Pursuit Controller and State Machine.

Provides autonomous path-tracking with LiDAR-based obstacle detection
for emergency stops, and a state machine managing the full
IDLE → MAPPING → PLANNING → FOLLOWING workflow.
"""
import numpy as np
import math
import utils


class PurePursuitController:
    """
    Pure-pursuit path-tracking controller for the QCar.
    Computes steering angle to follow a path given the current pose.
    """
    def __init__(self, wheelbase=0.26, lookahead_dist=3.0,
                 max_steering=0.35, base_speed=0.05):
        """
        Args:
            wheelbase: axle-to-axle distance in meters (QCar = 0.26m)
            lookahead_dist: lookahead distance in map units
            max_steering: maximum steering angle in radians
            base_speed: base throttle command
        """
        self.wheelbase = wheelbase
        self.lookahead_dist = lookahead_dist
        self.max_steering = max_steering
        self.base_speed = base_speed

    def compute(self, current_pose, path, map_units=20):
        """
        Compute motor command (throttle, steering) to follow the path.

        Args:
            current_pose: [x, y, theta] in map units
            path: list of (x, y) waypoints in grid coordinates
            map_units: conversion factor (grid coords are already in grid space)

        Returns:
            (throttle, steering): motor command tuple
            remaining_path: remaining waypoints after the lookahead
            done: True if goal reached
        """
        if not path or len(path) == 0:
            return (0.0, 0.0), [], True

        cx, cy, ctheta = current_pose[0], current_pose[1], current_pose[2]

        # Find the lookahead point on the path
        lookahead_point = None
        lookahead_idx = 0

        for i, (px, py) in enumerate(path):
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist >= self.lookahead_dist:
                lookahead_point = (px, py)
                lookahead_idx = i
                break

        # If no point is far enough, use the last point
        if lookahead_point is None:
            lookahead_point = path[-1]
            lookahead_idx = len(path) - 1

        # Check if we've reached the goal (last waypoint)
        goal = path[-1]
        goal_dist = math.sqrt((goal[0] - cx) ** 2 + (goal[1] - cy) ** 2)
        if goal_dist < self.lookahead_dist * 0.5:
            return (0.0, 0.0), [], True

        # Pure pursuit geometry
        tx, ty = lookahead_point
        dx = tx - cx
        dy = ty - cy

        # Transform to vehicle frame
        local_x = dx * np.cos(ctheta) + dy * np.sin(ctheta)
        local_y = -dx * np.sin(ctheta) + dy * np.cos(ctheta)

        # Curvature
        L = math.sqrt(local_x ** 2 + local_y ** 2)
        if L < 0.001:
            return (0.0, 0.0), path, False

        curvature = 2.0 * local_y / (L * L)

        # Steering angle (Ackermann)
        steering = np.arctan(curvature * self.wheelbase * map_units)
        steering = np.clip(steering, -self.max_steering, self.max_steering)

        # Speed: slow down on curves
        abs_curv = abs(curvature)
        if abs_curv > 0.1:
            throttle = self.base_speed * 0.5
        else:
            throttle = self.base_speed

        remaining_path = path[lookahead_idx:]
        return (throttle, steering), remaining_path, False


class ObstacleChecker:
    """Checks LiDAR readings for obstacles in the path of travel."""
    def __init__(self, stop_distance=0.3, front_angle_range=30, map_units=20):
        """
        Args:
            stop_distance: distance (in map units) at which to trigger e-stop
            front_angle_range: half-angle (degrees) of the forward cone to check
            map_units: conversion factor
        """
        self.stop_distance = stop_distance * map_units
        self.front_angle_range = np.radians(front_angle_range)

    def check(self, angles, dists, heading):
        """
        Check if there's an obstacle in the forward cone.

        Returns:
            True if obstacle detected (should stop), False otherwise.
        """
        for i in range(len(angles)):
            relative_angle = angles[i]
            # Check if angle is within forward cone
            if abs(relative_angle) < self.front_angle_range:
                if 0 < dists[i] < self.stop_distance:
                    return True
        return False


# State machine states
IDLE = "IDLE"
MAPPING = "MAPPING"
PLANNING = "PLANNING"
FOLLOWING = "FOLLOWING"
ESTOP = "ESTOP"
DONE = "DONE"


class WaypointManager:
    """
    State machine for autonomous waypoint following.

    States:
        IDLE     → waiting for start command
        MAPPING  → teleop-driven SLAM mapping
        PLANNING → computing A* path to goal
        FOLLOWING → pure-pursuit path tracking
        ESTOP    → emergency stop (obstacle detected)
        DONE     → goal reached
    """
    def __init__(self, planner, controller, obstacle_checker, map_units=20):
        """
        Args:
            planner: AStarPlanner instance
            controller: PurePursuitController instance
            obstacle_checker: ObstacleChecker instance
            map_units: conversion factor
        """
        self.planner = planner
        self.controller = controller
        self.obstacle_checker = obstacle_checker
        self.map_units = map_units

        self.state = IDLE
        self.goal = None
        self.path = None
        self.replan_counter = 0
        self.replan_interval = 100  # re-plan every N iterations
        self.estop_counter = 0
        self.estop_wait = 50  # wait N iterations before re-checking

    def set_goal(self, goal_xy):
        """Set the navigation goal (x, y) in grid coordinates."""
        self.goal = goal_xy
        self.state = PLANNING
        print(f"[WaypointManager] Goal set to {goal_xy}, switching to PLANNING")

    def start_mapping(self):
        """Switch to MAPPING mode (teleop SLAM)."""
        self.state = MAPPING
        print("[WaypointManager] Switched to MAPPING mode")

    def get_motor_command(self, current_pose, angles, dists, grid_map):
        """
        Main state machine update. Returns (throttle, steering) motor command.

        Args:
            current_pose: [x, y, theta] in map units
            angles: LiDAR angles array
            dists: LiDAR distances array (in map units)
            grid_map: current GridMap instance

        Returns:
            (throttle, steering) tuple
        """
        if self.state == IDLE:
            return (0.0, 0.0)

        elif self.state == MAPPING:
            # In mapping mode, return zero — the gamepad drives
            return (0.0, 0.0)

        elif self.state == PLANNING:
            return self._handle_planning(current_pose, grid_map)

        elif self.state == FOLLOWING:
            return self._handle_following(current_pose, angles, dists, grid_map)

        elif self.state == ESTOP:
            return self._handle_estop(angles, dists)

        elif self.state == DONE:
            print("[WaypointManager] Goal reached!")
            return (0.0, 0.0)

        return (0.0, 0.0)

    def _handle_planning(self, current_pose, grid_map):
        """Plan a path from current position to goal."""
        if self.goal is None:
            print("[WaypointManager] No goal set!")
            self.state = IDLE
            return (0.0, 0.0)

        gsize = grid_map.gsize
        start_grid = (int(round(current_pose[0] / gsize)),
                      int(round(current_pose[1] / gsize)))
        goal_grid = self.goal

        path = self.planner.plan(start_grid, goal_grid, grid_map)

        if path is None:
            print("[WaypointManager] Planning failed! Retrying in next cycle...")
            return (0.0, 0.0)

        self.path = path
        self.replan_counter = 0
        self.state = FOLLOWING
        print(f"[WaypointManager] Path planned with {len(path)} waypoints, switching to FOLLOWING")
        return (0.0, 0.0)

    def _handle_following(self, current_pose, angles, dists, grid_map):
        """Follow the planned path with obstacle detection."""
        # Check for obstacles
        if self.obstacle_checker.check(angles, dists, current_pose[2]):
            print("[WaypointManager] Obstacle detected! E-STOP")
            self.state = ESTOP
            self.estop_counter = 0
            return (0.0, 0.0)

        # Periodic re-planning
        self.replan_counter += 1
        if self.replan_counter >= self.replan_interval:
            self.state = PLANNING
            return (0.0, 0.0)

        # Pure pursuit
        if self.path is None or len(self.path) == 0:
            self.state = DONE
            return (0.0, 0.0)

        (throttle, steering), remaining, done = self.controller.compute(
            current_pose, self.path, self.map_units
        )

        if done:
            self.state = DONE
            return (0.0, 0.0)

        self.path = remaining
        return (throttle, steering)

    def _handle_estop(self, angles, dists):
        """Wait during emergency stop, then re-plan."""
        self.estop_counter += 1
        if self.estop_counter >= self.estop_wait:
            # Check if obstacle is cleared
            if not self.obstacle_checker.check(angles, dists, 0):
                print("[WaypointManager] Obstacle cleared, re-planning...")
                self.state = PLANNING
            else:
                self.estop_counter = 0  # reset wait
        return (0.0, 0.0)

    def get_state(self):
        """Return current state string."""
        return self.state
