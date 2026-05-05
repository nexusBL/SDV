import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    from hal.products.mats import SDCSRoadMap
except ImportError:
    print("[GlobalPlanner] WARNING: hal.products.mats not found. Script may fail without the Quanser HAL library.")
    SDCSRoadMap = None


class GlobalPlanner:
    """
    An industry-grade utility for Global Path Planning on the Quanser SDCS physical track.
    
    This class wraps the SDCS topological node graph to generate optimal driving
    trajectories (X, Y physical coordinates) for the QCar to follow. It also includes
    robust visualization overlay tools.
    """

    def __init__(self, left_hand_traffic=True, use_small_map=False, map_image_dir="."):
        """
        Initializes the topological roadmap.
        
        Args:
            left_hand_traffic (bool): Use left-hand or right-hand driving rules.
            use_small_map (bool): Target the smaller studio map instead of the large one.
            map_image_dir (str): Directory containing the SDCS_MapLayout.png images.
        """
        self.use_small_map = use_small_map
        self.left_hand_traffic = left_hand_traffic
        self.map_image_dir = map_image_dir
        
        self.roadmap = None
        
        if SDCSRoadMap is not None:
            print("[GlobalPlanner] Initializing QCar topological nodes...")
            self.roadmap = SDCSRoadMap(
                leftHandTraffic=self.left_hand_traffic,
                useSmallMap=self.use_small_map
            )
        else:
            raise RuntimeError("Quanser HAL library is required to initialize the roadmap.")

    def plan_path(self, node_sequence):
        """
        Generates the shortest legal driving path passing through the given sequence of nodes.
        
        Args:
            node_sequence (list): A list or tuple of node indices. e.g., [0, 5, 8, 12].
                                  These represent physical intersections/waypoints on the track.
        
        Returns:
            np.ndarray: A 2xN numpy array representing the optimal XY coordinates of the route.
        """
        if not self.roadmap:
            return None
        
        print(f"[GlobalPlanner] Generating shortest path for route: {node_sequence}")
        path = self.roadmap.generate_path(nodeSequence=node_sequence)
        
        # Print out the location of each node as requested
        print("\n[GlobalPlanner] --- Roadmap Nodes ---")
        for i, node in enumerate(self.roadmap.nodes):
            print(
                'Node ' + str(i) + ': Pose = ['
                + f"{node.pose[0, 0]:.3f}" + ', '
                + f"{node.pose[1, 0]:.3f}" + ', '
                + f"{node.pose[2, 0]:.3f}" + ']'
            )
        print("-----------------------------------\n")
        
        if path is None:
            print("[GlobalPlanner] ❌ ERROR: Failed to generate a valid path!")
        else:
            print(f"[GlobalPlanner] ✓ Path generated successfully ({path.shape[1]} physical coordinates).")
            
        return path

    def display_path(self, path=None):
        """
        Visually renders the generated path over the physical SDCS layout image
        to verify the trajectory before sending to the car.
        """
        if not self.roadmap:
            return

        # 1. Fetch matplotlib objects from Quanser's HAL display method
        plt_obj, ax = self.roadmap.display()
        plt_obj.title("QCar Autonomous Trajectory Plan", fontsize=14, fontweight='bold')
        
        # 2. Determine Map Layout Configuration
        if self.use_small_map:
            img_file = 'SDCS_SmallMapLayout.png'
            y_origin = 299.5
        else:
            img_file = 'SDCS_MapLayout.png'
            y_origin = 1008.5
            
        # Locate the background image file
        img_path = os.path.normpath(os.path.join(self.map_image_dir, img_file))
        
        if not os.path.exists(img_path):
             print(f"[GlobalPlanner] ⚠️ Warning: Background image '{img_file}' not found at {img_path}.")
             print("[GlobalPlanner] Note: Place the blueprint image in that directory for the full visual overlay.")
        else:
             print(f"[GlobalPlanner] Overlaying trajectory on blueprint: {img_path}")
             # Load and align the physical mat image perfectly with coordinate space
             img = mpimg.imread(img_path)
             img = np.flipud(img)
             scale = 0.00476556
             
             # Shift image based on scale and Quanser physical origin
             x_translation = -483.5 * scale
             y_translation = (y_origin - img.shape[0]) * scale
             img_extent = (
                 x_translation, img.shape[1] * scale + x_translation,
                 y_translation, img.shape[0] * scale + y_translation
             )
             
             # Inject the floor layout underneath the nodes (zorder=0)
             ax.imshow(img, extent=img_extent, origin='lower', zorder=0)

        # 3. Draw the newly generated path on top (zorder=5)
        if path is not None:
             ax.plot(path[0, :], path[1, :], color='#00FF00', linestyle='-', linewidth=4, 
                     label='Optimal Target Trajectory', zorder=5)
             
             # Add start and end point markers
             ax.plot(path[0, 0], path[1, 0], marker='o', color='blue', markersize=12, label='Start', zorder=6)
             ax.plot(path[0, -1], path[1, -1], marker='*', color='red', markersize=14, label='Destination', zorder=6)
             
             ax.legend(loc='lower right', fancybox=True, shadow=True)

        plt_obj.show()


if __name__ == "__main__":
    # =========================================================================
    # QUICK START / DEMO
    # To run this script perfectly, make sure 'SDCS_MapLayout.png' is
    # placed in the same folder as this script, or update the directory mapping!
    # =========================================================================
    
    # Example: Initialize the planner pointing to the directory containing images
    planner = GlobalPlanner(
        left_hand_traffic=True, 
        use_small_map=False,
        map_image_dir=os.path.dirname(__file__) 
    )
    
    # Define a sequence of waypoint nodes to travel through.
    # Note: Check the raw matplotlib output to see node numbers!
    target_route = [0, 4, 12]  
    
    # 1. Generate optimal physical coordinates
    trajectory = planner.plan_path(target_route)
    
    # 2. Visually verify the trajectory before driving
    planner.display_path(trajectory)
    
    # (Future step: Send 'trajectory' to a Pure Pursuit or Stanley Controller)
