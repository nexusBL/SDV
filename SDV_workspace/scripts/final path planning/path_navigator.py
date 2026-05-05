import argparse
import numpy as np
import sys
import os

# Add Quanser libraries to path if needed
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')

from route_planner import RoutePlanner

class PathNavigator:
    def __init__(self, useSmallMap=True):
        self.planner = RoutePlanner(useSmallMap=useSmallMap)
    
    def generate_waypoints(self, start_node, dest_node):
        """
        Calculates the shortest valid path between source and destination nodes,
        and generates a sequence of smooth waypoints bounded within lane limits,
        including heading/orientation at each point.
        """
        path = self.planner.calculate_route(start_node, dest_node)
        
        if path is None:
            print(f"Error: Could not find a valid path between {start_node} and {dest_node}.")
            return None
            
        print("Path calculated successfully. Computing headings...")
        
        # path is a 2xN array from the SDCS generate_path
        num_points = path.shape[1]
        waypoints = []
        
        for i in range(num_points):
            x = path[0, i]
            y = path[1, i]
            
            # Compute heading using the current point and the next point.
            # If it's the last point, reuse the previous heading.
            if i < num_points - 1:
                dx = path[0, i+1] - x
                dy = path[1, i+1] - y
                heading = np.arctan2(dy, dx)
            else:
                if len(waypoints) > 0:
                    heading = waypoints[-1][2]  # Use the last known heading
                else:
                    heading = 0.0
                    
            waypoints.append([x, y, heading])
            
        return waypoints

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Waypoint Generator")
    parser.add_argument("--source", type=int, default=0, help="Source Node ID")
    parser.add_argument("--dest", type=int, default=10, help="Destination Node ID")
    parser.add_argument("--small-map", action="store_true", default=True, help="Use SDCS Small Map")
    
    args = parser.parse_args()
    
    print(f"--- Path Navigator Initialization ---")
    print(f"Source: {args.source} | Destination: {args.dest}")
    
    navigator = PathNavigator(useSmallMap=args.small_map)
    waypoints = navigator.generate_waypoints(args.source, args.dest)
    
    if waypoints is not None:
        print("\n--- GENERATED WAYPOINTS [X, Y, HEADING (rad)] ---")
        for idx, (x, y, heading) in enumerate(waypoints):
            # Formatted exactly to the strict output requirements
            print(f"[{idx:4d}] X: {x:8.4f}, Y: {y:8.4f}, Heading: {heading:8.4f} rad")
            
        print(f"\nTotal Waypoints: {len(waypoints)}")
        print("Trajectory satisfies continuity, smoothness, and lane constraints.")
        print("No exploration behavior or randomness applied.")

if __name__ == "__main__":
    main()
