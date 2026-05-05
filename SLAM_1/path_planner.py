"""
A* Path Planner on the occupancy grid.

Provides path planning with obstacle inflation and path smoothing
for use with the GridMap SLAM system.
"""
import numpy as np
import heapq
import math


class AStarPlanner:
    """
    A* path planner that operates on the GridMap's occupancy grid.
    Uses 8-connected grid neighbors with obstacle inflation.
    """
    def __init__(self, obstacle_threshold=0.65, inflation_radius=3):
        """
        Args:
            obstacle_threshold: probability above which a cell is considered occupied
            inflation_radius: number of grid cells to inflate obstacles by (safety margin)
        """
        self.obstacle_threshold = obstacle_threshold
        self.inflation_radius = inflation_radius

    def _inflate_obstacles(self, grid):
        """
        Inflate obstacles in the occupancy grid by the configured radius.
        Returns a binary grid: True = blocked (obstacle or inflated), False = free.
        """
        blocked = grid > self.obstacle_threshold
        inflated = np.copy(blocked)

        if self.inflation_radius <= 0:
            return inflated

        # Simple dilation using a circular kernel
        rows, cols = grid.shape
        r = self.inflation_radius
        for iy in range(rows):
            for ix in range(cols):
                if blocked[iy, ix]:
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            if dx * dx + dy * dy <= r * r:
                                ny, nx = iy + dy, ix + dx
                                if 0 <= ny < rows and 0 <= nx < cols:
                                    inflated[ny, nx] = True
        return inflated

    def plan(self, start_grid, goal_grid, grid_map):
        """
        Plan a path from start to goal on the grid map using A*.

        Args:
            start_grid: (x, y) in grid coordinates
            goal_grid: (x, y) in grid coordinates
            grid_map: GridMap instance

        Returns:
            List of (x, y) grid coordinates forming the path, or None if no path found.
        """
        # Convert map to array
        grid_array, origin = grid_map.to_array(padding=10)

        # Convert grid coordinates to array indices
        sx = int(round(start_grid[0])) - origin[0]
        sy = int(round(start_grid[1])) - origin[1]
        gx = int(round(goal_grid[0])) - origin[0]
        gy = int(round(goal_grid[1])) - origin[1]

        rows, cols = grid_array.shape

        # Bounds check
        if not (0 <= sx < cols and 0 <= sy < rows):
            print(f"[AStarPlanner] Start ({sx},{sy}) out of grid bounds ({cols},{rows})")
            return None
        if not (0 <= gx < cols and 0 <= gy < rows):
            print(f"[AStarPlanner] Goal ({gx},{gy}) out of grid bounds ({cols},{rows})")
            return None

        # Inflate obstacles
        blocked = self._inflate_obstacles(grid_array)

        if blocked[sy, sx]:
            print("[AStarPlanner] Start position is inside an obstacle!")
            return None
        if blocked[gy, gx]:
            print("[AStarPlanner] Goal position is inside an obstacle!")
            return None

        # A* search
        # 8-connected neighbors: (dx, dy, cost)
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        def heuristic(x1, y1, x2, y2):
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0.0}
        closed = set()

        while open_set:
            f, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))

            # Goal reached
            if cx == gx and cy == gy:
                # Reconstruct path in grid coordinates
                path = []
                node = (gx, gy)
                while node in came_from:
                    path.append((node[0] + origin[0], node[1] + origin[1]))
                    node = came_from[node]
                path.append((sx + origin[0], sy + origin[1]))
                path.reverse()
                return self._smooth_path(path)

            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                if blocked[ny, nx]:
                    continue
                if (nx, ny) in closed:
                    continue

                new_g = g_score[(cx, cy)] + cost
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f_score = new_g + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (f_score, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

        print("[AStarPlanner] No path found!")
        return None

    def _smooth_path(self, path):
        """
        Simplify path by removing collinear/near-collinear points.
        Keeps start, end, and points where direction changes significantly.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            # Check if point i is nearly collinear with previous and next
            prev = smoothed[-1]
            curr = path[i]
            nxt = path[i + 1]

            # Vectors
            v1x = curr[0] - prev[0]
            v1y = curr[1] - prev[1]
            v2x = nxt[0] - curr[0]
            v2y = nxt[1] - curr[1]

            # Cross product magnitude (deviation from straight line)
            cross = abs(v1x * v2y - v1y * v2x)
            if cross > 0.5:  # direction change threshold
                smoothed.append(curr)

        smoothed.append(path[-1])
        return smoothed

    def plan_from_poses(self, start_pos, goal_pos, grid_map, gsize=1.0):
        """
        Convenience method: plan from world-coordinate poses.

        Args:
            start_pos: [x, y, theta] in map units
            goal_pos: [x, y] in map units (or [x, y, theta])
            grid_map: GridMap instance
            gsize: grid size

        Returns:
            List of (x, y) in grid coordinates, or None.
        """
        start_grid = (int(round(start_pos[0] / gsize)), int(round(start_pos[1] / gsize)))
        goal_grid = (int(round(goal_pos[0] / gsize)), int(round(goal_pos[1] / gsize)))
        return self.plan(start_grid, goal_grid, grid_map)
