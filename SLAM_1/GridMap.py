# Copied from https://github.com/toolbuddy/2D-Grid-SLAM/blob/master/GridMap.py
import numpy as np
import utils

# Map making class
# Uses Log odds form for probabilities
class GridMap:
    # gmap saves into a dictionary of grid cels {x,y} and probabilities with max/min set by map_param[3/4]
    def __init__(self, map_param, gsize=1.0):
        self.map_param = map_param
        self.gmap = {}
        self.gsize = gsize
        self.center = np.array([0,0])
        self.boundary = [9999, -9999, 9999, -9999] 

    # Returns the probability of an object in that grid cell
    # Returns .5 when not initialized
    def GetGridProb(self, pos):
        if pos in self.gmap:
            return np.exp(self.gmap[pos]) / (1.0 + np.exp(self.gmap[pos]))
        else:
            return 0.5

    # Converts a coordinate to grid cell and calls GetGridProb
    def GetCoordProb(self, pos):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        return self.GetGridProb((x,y))

    # Given a range of x and y values, map that range to an array 
    def GetMapProb(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.GetGridProb((i,j))
                if i ==0 and j == 0:
                    self.center = np.array([idx, idy])
                idy += 1
            idx += 1
        return map_prob

    # Given a start point and an end point, draw a line using Bresenhams line algorithm and update each grid cell
    # The grid cell where the endpoint is is where an object is expected to be, and the other cells on the line are expected to be empty
    # map_param: [lo_occ (+), lo_free (-), lo_max, lo_min]
    def GridMapLine(self, x0, x1, y0, y1):
        
        # Scale the position
        x0, x1 = int(round(x0/self.gsize)), int(round(x1/self.gsize))
        y0, y1 = int(round(y0/self.gsize)), int(round(y1/self.gsize))

        rec = utils.Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            # Intermediate cells are FREE (negative log-odds)
            # Last 2 cells are OCCUPIED (positive log-odds)
            if i < len(rec)-2:
                change = self.map_param[1]  # lo_free (negative)
            else:
                change = self.map_param[0]  # lo_occ (positive)

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += change
            else:
                self.gmap[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]                  
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]

            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]

    # Returns a list of (x, y) grid coordinates where probability > threshold (occupied cells)
    def get_obstacle_coords(self, threshold=0.7):
        obstacles = []
        for pos, log_odds in self.gmap.items():
            prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
            if prob > threshold:
                obstacles.append(pos)
        return obstacles

    # Returns the occupancy grid as a 2D numpy array and the origin offset
    def to_array(self, padding=5):
        x0 = self.boundary[0] - padding
        x1 = self.boundary[1] + padding
        y0 = self.boundary[2] - padding
        y1 = self.boundary[3] + padding
        width = x1 - x0
        height = y1 - y0
        grid = np.ones((height, width)) * 0.5  # unknown = 0.5
        for (gx, gy), log_odds in self.gmap.items():
            ix = gx - x0
            iy = gy - y0
            if 0 <= ix < width and 0 <= iy < height:
                grid[iy, ix] = np.exp(log_odds) / (1.0 + np.exp(log_odds))
        return grid, (x0, y0)

if __name__ == '__main__':
    #lo_occ, lo_free, lo_max, lo_min
    map_param = [0.9, -0.7, 5.0, -5.0]
    m = GridMap(map_param)
    pos = (0.0,0.0)
    m.gmap[pos] = 0.1
    print(m.GetGridProb(pos))
    print(m.GetGridProb((0,0)))
