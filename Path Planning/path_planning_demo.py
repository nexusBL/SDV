import numpy as np
from hal.products.mats import SDCSRoadMap
import matplotlib.image as mpimg
import os

def main():
    # 1. Configuration parameters for the small map
    useSmallMap = True
    leftHandTraffic = True
    
    # 2. DEFINING THE ROUTE
    # Here we define a sequence of nodes for the car to drive through.
    # For example: Start at Node 0, drive to Node 5, and end at Node 11.
    nodeSequence = [0, 5, 11]

    print(f"Calculating route for sequence: {nodeSequence}...")

    # 3. Create a SDCSRoadMap instance with desired configuration.
    roadmap = SDCSRoadMap(
        leftHandTraffic=leftHandTraffic,
        useSmallMap=useSmallMap
    )

    # 4. Generate the shortest path passing through the given sequence of nodes.
    # The A* algorithm will calculate the exact trajectory automatically.
    path = roadmap.generate_path(nodeSequence=nodeSequence)

    # 5. Display the roadmap with nodes, edges, and labels using matplotlib.
    plt, ax = roadmap.display()

    # Add the map image background
    if useSmallMap:
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
    img = mpimg.imread(imgPath)
    img = np.flipud(img)
    scale = 0.00476556
    x_translation, y_translation = -483.5 * scale, (yOrigin-img.shape[0]) * scale
    img_extent = (
        x_translation, img.shape[1] * scale + x_translation,
        y_translation, img.shape[0] * scale + y_translation
    )
    ax.imshow(img, extent=img_extent, origin='lower', zorder=0)

    # 6. Plot the calculated path
    if path is None:
        print('Error: Failed to find a valid route between your selected nodes.')
    else:
        # We increase the linewidth to 3 so the red route stands out clearly!
        ax.plot(path[0, :], path[1, :], color='blue', linestyle='-', linewidth=4)
        print("Success! Route calculated and displayed on the map.")

    plt.title("QCar Path Planning Demo")
    plt.show()

if __name__ == '__main__':
    main()
