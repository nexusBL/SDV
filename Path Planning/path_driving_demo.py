#!/usr/bin/env python3
import sys

# We import your fully-built navigation supervisor from your SDV_workspace!
from navigation_main import NavigationSupervisor

def main():
    print("==================================================")
    print(" Starting Physical QCar Navigation Demo ")
    print("==================================================")
    
    # These are the exact nodes from our plotting demo!
    waypoints = [0, 5, 11]
    print(f"Goal: Autonomously drive from node {waypoints[0]} through sequence: {waypoints}")
    
    # This uses your existing hardware controller, safety monitors,
    # and Pure Pursuit steering algorithm to physically drive the path!
    app = NavigationSupervisor(
        target_nodes=waypoints,
        preview_mode=False,
        autostart=True,    # Setting to True so it immediately starts driving automatically
        headless=False
    )
    
    app.run()

if __name__ == "__main__":
    main()
