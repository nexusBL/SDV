import numpy as np
import os
import matplotlib.image as mpimg
from hal.products.mats import SDCSRoadMap

class RoutePlanner:
    def __init__(self, leftHandTraffic=True, useSmallMap=True):
        """
        Initializes the SDCSRoadMap API.
        """
        self.useSmallMap = useSmallMap
        self.roadmap = SDCSRoadMap(
            leftHandTraffic=leftHandTraffic,
            useSmallMap=useSmallMap
        )
        self.path = None
        self.node_sequence = []
        self.current_idx = 0

    def calculate_route(self, start_node, dest_node):
        """
        Calculates the shortest path from start_node to dest_node.
        Returns the (2, N) path array.
        """
        # SDCSRoadMap.generate_path internally uses Dijkstra/A-star 
        # on the nodeSequence list. If only start/dest are given, it finds the shortest.
        self.path = self.roadmap.generate_path(nodeSequence=[start_node, dest_node])
        
        # Determine the sequence of nodes for the FSM
        # (SDCSRoadMap might not explicitly return the node indices list, 
        # but we can infer them or use the node poses for proximity)
        self.node_sequence = [start_node, dest_node] 
        self.current_idx = 0
        
        if self.path is None:
            print(f"FAILED to find path from {start_node} to {dest_node}")
        return self.path

    def get_node_pose(self, node_id):
        """Returns [x, y, theta] for a given node."""
        if 0 <= node_id < len(self.roadmap.nodes):
            pose = self.roadmap.nodes[node_id].pose
            return np.array([pose[0,0], pose[1,0], pose[2,0]])
        return None

    def get_all_nodes(self):
        """Helper to list all available nodes for the user."""
        return [(i, self.get_node_pose(i)) for i in range(len(self.roadmap.nodes))]

    def display_path(self):
        """Displays the path on the roadmap using matplotlib."""
        if self.path is None:
            print("No path to display.")
            return

        plt, ax = self.roadmap.display()

        if self.useSmallMap:
            imgFile = './SDCS_SmallMapLayout.png'
            yOrigin = 299.5
        else:
            imgFile = './SDCS_MapLayout.png'
            yOrigin = 1008.5

        imgPath = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                imgFile
            )
        )

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

        # Plot the path with a blue line
        ax.plot(self.path[0, :], self.path[1, :], 'blue', linestyle='-', linewidth=2)
        plt.show()

if __name__ == '__main__':
    # Test
    planner = RoutePlanner(useSmallMap=True)
    # Using dest_node 10, as 12 is out of range for the small map (which has 11 nodes)
    path = planner.calculate_route(0, 10)
    if path is not None:
        print("Path calculated successfully with", path.shape[1], "waypoints.")
        print("Destination pose:", planner.get_node_pose(10))
        planner.display_path()
