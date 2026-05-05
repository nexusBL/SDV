import time
import numpy as np
from pal.products.qcar import QCarLidar

def main():
    print("Testing Lidar...")
    myLidar = QCarLidar(numMeasurements=1000, rangingDistanceMode=2, interpolationMode=0)
    
    for _ in range(20):
        myLidar.read()
        angles = np.array(myLidar.angles)
        dists = np.array(myLidar.distances)
        
        valid = (dists > 0.05) & (dists < 5.0)
        v_ang = angles[valid]
        v_dist = dists[valid]
        
        if len(v_dist) > 0:
            min_idx = np.argmin(v_dist)
            print(f"Min dist: {v_dist[min_idx]:.2f}m at raw angle: {v_ang[min_idx]:.2f} rad ({np.rad2deg(v_ang[min_idx]):.0f} deg)")
        time.sleep(0.5)
        
    myLidar.terminate()

if __name__ == "__main__":
    main()
