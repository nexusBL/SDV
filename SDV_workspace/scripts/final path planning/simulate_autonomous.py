import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add Quanser libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/0_libraries/python')
sys.path.insert(0, '/home/nvidia/Desktop/SDV_workspace/scripts/Path Planning')

from route_planner import RoutePlanner
from kinematics_engine import KinematicsEngine
from control.pure_pursuit import PurePursuitController
from main_autonomous import MockAppConfig  # Import the configuration we used

def simulate():
    print("--- Simulating QCar Pure Pursuit Navigator ---")
    planner = RoutePlanner()
    
    start_node = 1
    dest_node = 5
    
    path = planner.calculate_route(start_node, dest_node)
    if path is None:
        print("Fatal: Could not find path.")
        return

    # Initialize Controllers
    start_x = path[0, 0]
    start_y = path[1, 0]
    next_x = path[0, 1]
    next_y = path[1, 1]
    start_theta = math.atan2(next_y - start_y, next_x - start_x)
    
    kinematics = KinematicsEngine(x0=start_x, y0=start_y, th0=start_theta)
    kinematics.prev_counts = 0 # Initialize encoder explicitly
    virtual_encoder = 0.0      # Track simulated distance
    
    pursuit = PurePursuitController(MockAppConfig())
    # Tighten parameters for pure simulation where dynamics are exact
    pursuit.cfg.max_steering_rate = 1.0 
    pursuit.cfg.steering_alpha = 1.0    # No smoothing delay
    
    pursuit.set_path(path)
    
    end_x = path[0, -1]
    end_y = path[1, -1]
    
    dt = 0.033 # Simulate 30Hz loop
    
    driven_x = []
    driven_y = []
    
    print("Simulating virtual drive...")
    
    step = 0
    max_steps = 3000
    while True:
        current_pose = (kinematics.x, kinematics.y, kinematics.theta)
        driven_x.append(kinematics.x)
        driven_y.append(kinematics.y)
        
        dist_to_goal = math.hypot(end_x - kinematics.x, end_y - kinematics.y)
        if dist_to_goal < 0.2:
            print("🏁 FINAL DESTINATION REACHED!")
            break
            
        steer, speed, _ = pursuit.compute(current_pose)
        
        # Simulate hardware applying the speed over dt
        delta_counts = (speed * dt) / kinematics.m_per_cnt
        virtual_encoder += delta_counts
        
        # Update kinematic state
        kinematics.update_state(virtual_encoder, steer, dt)
        
        step += 1
        if step > max_steps:
             print("Simulation timed out.")
             break
    
    print("Drive complete. Plotting trajectory...")
    
    # ---------------- Plotting ---------------- #
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Render map
    imgFile = './SDCS_SmallMapLayout.png'
    yOrigin = 299.5
    imgPath = os.path.normpath(os.path.join(os.path.dirname(__file__), imgFile))
    try:
        img = mpimg.imread(imgPath)
        img = np.flipud(img)
        scale = 0.00476556
        x_translation, y_translation = -483.5 * scale, (yOrigin-img.shape[0]) * scale
        img_extent = (
            x_translation, img.shape[1] * scale + x_translation,
            y_translation, img.shape[0] * scale + y_translation
        )
        ax.imshow(img, extent=img_extent, origin='lower', zorder=0)
    except FileNotFoundError:
        print(f"Warning: Image file {imgPath} not found.")

    # Plot optimal mathematical path
    ax.plot(path[0, :], path[1, :], 'blue', linestyle='--', linewidth=2, label="Optimal Route")
    
    # Plot simulated driven trajectory
    ax.plot(driven_x, driven_y, 'red', linestyle='-', linewidth=2, alpha=0.8, label="Simulated Car Trajectory")
    
    # Start and End markers
    ax.scatter([start_x], [start_y], color='green', marker='o', s=100, label="Start", zorder=5)
    ax.scatter([end_x], [end_y], color='purple', marker='x', s=100, label="Destination", zorder=5)

    ax.legend(loc='upper right')
    ax.set_title('Pure Pursuit Controller Trajectory Verification')
    
    # Save the plot
    output_img = 'trajectory_simulation.png'
    plt.savefig(output_img, dpi=150)
    print(f"Plot saved to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    simulate()
