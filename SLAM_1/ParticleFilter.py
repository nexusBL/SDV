# Adapted from https://github.com/toolbuddy/2D-Grid-SLAM/blob/master/ParticleFilter.py
import numpy as np
from GridMap import *
import random
import math
import utils
import copy

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# A particle is a virtual representation of the robot, with a position, sensor qualities, and map.
class Particle:
    def __init__(self, pos, sensorSize, maxDist, mapUnits, gmap):
        self.pos = pos
        self.sensorSize = sensorSize
        self.maxDist = maxDist
        self.mapUnits = mapUnits
        self.gmap = gmap
        # Cached distance transform for fast nearest-obstacle lookup
        self._dist_cache = None
        self._dist_origin = None
        self._dist_shape = None
        self._cache_dirty = True

    # The sampling method moves the robot and uses a gaussian distribution to model error
    def Sampling(self, heading, turnAngle, encoder_dist, sig=[0.2,0.2,0.05]):
        self.pos = utils.posUpdate(self.pos, turnAngle, self.mapUnits, encoder_dist)

        self.pos[0] += random.gauss(0,sig[0])
        self.pos[1] += random.gauss(0,sig[1])
        self.pos[2] += random.gauss(0,sig[2])

    def _rebuild_distance_cache(self):
        """Build a distance-transform from the occupancy grid for fast nearest-obstacle queries."""
        if not HAS_SCIPY:
            self._dist_cache = None
            return

        padding = 10
        b = self.gmap.boundary
        x0 = b[0] - padding
        x1 = b[1] + padding
        y0 = b[2] - padding
        y1 = b[3] + padding
        width = x1 - x0
        height = y1 - y0

        if width <= 0 or height <= 0:
            self._dist_cache = None
            return

        # Build binary grid: 1 = free/unknown, 0 = occupied
        binary = np.ones((height, width), dtype=np.float64)
        occ_threshold = 0.6
        for (gx, gy), log_odds in self.gmap.gmap.items():
            prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
            ix = gx - x0
            iy = gy - y0
            if 0 <= ix < width and 0 <= iy < height and prob > occ_threshold:
                binary[iy, ix] = 0

        self._dist_cache = distance_transform_edt(binary).astype(np.float32)
        self._dist_origin = (x0, y0)
        self._dist_shape = (width, height)
        self._cache_dirty = False

    def NearestDistanceFast(self, x, y):
        """Look up nearest obstacle distance from cached distance-transform."""
        if self._dist_cache is None:
            return 9999.0
        gsize = self.gmap.gsize
        ix = int(round(x / gsize)) - self._dist_origin[0]
        iy = int(round(y / gsize)) - self._dist_origin[1]
        if 0 <= ix < self._dist_shape[0] and 0 <= iy < self._dist_shape[1]:
            return float(self._dist_cache[iy, ix]) * gsize
        return 9999.0

    # Brute-force fallback (used when scipy is not available)
    def NearestDistance(self, x, y, wsize, th):
        min_dist = 9999
        gsize = self.gmap.gsize
        xx = int(round(x/gsize))
        yy = int(round(y/gsize))
        for i in range(xx-wsize, xx+wsize):
            for j in range(yy-wsize, yy+wsize):
                if self.gmap.GetGridProb((i,j)) < th:
                    dist = (i-xx)*(i-xx) + (j-yy)*(j-yy)
                    if dist < min_dist:
                        min_dist = dist
        return math.sqrt(float(min_dist)*gsize)

    # This method determines the likelihood of a particle.
    # The method uses the particles map and position and figures out what lidar data it should get and how closely the real lidar data corresponds.
    def LikelihoodField(self, angles, dists):
        p_hit = 0.9
        p_rand = 0.1
        sig_hit = 3.0
        q = 1

        # Rebuild distance cache if stale
        if HAS_SCIPY and self._cache_dirty:
            self._rebuild_distance_cache()

        plist = utils.EndPoint(self.pos, angles, dists)
        for i in range(len(plist)):
            if dists[i] >= self.maxDist * self.mapUnits or dists[i] < .01:
                continue

            if HAS_SCIPY and self._dist_cache is not None:
                dist = self.NearestDistanceFast(plist[i][0], plist[i][1])
            else:
                dist = self.NearestDistance(plist[i][0], plist[i][1], 4, 0.2)

            q += math.log(p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/(self.maxDist * self.mapUnits))
        return q

    # Changes the given map based on robots position and lidar scan results
    def Mapping(self, num_measurements, angles, dists):
        for i in range(num_measurements):
            if dists[i] >= (self.maxDist) * self.mapUnits:
                continue
            if dists[i] < .05:
                continue
            theta = self.pos[2] - angles[i]
            self.gmap.GridMapLine(
            int(self.pos[0]), 
            int(self.pos[0]+dists[i]*np.cos(theta)),
            int(self.pos[1]),
            int(self.pos[1]+dists[i]*np.sin(theta))
            )
        # Mark cache as dirty after mapping update
        self._cache_dirty = True

class ParticleFilter:
    """
    A particle filter is a way to estimate a robots position.
    takes robot attributes like the starting position, how many sensor readings, max Distance for sensor readings, map units for conversion, and the starting map
    size = how many particles
    A particle filter instantiates the above particle class to model the robots movement with error
    """
    def __init__(self, pos, sensorSize, maxDist, mapUnits, gmap, size):
        self.size = size
        self.particle_list = []
        self.weights = np.ones((size), dtype=float) / size
        p = Particle(pos.copy(), sensorSize, maxDist, mapUnits, copy.deepcopy(gmap))
        for i in range(size):
            self.particle_list.append(copy.deepcopy(p))

    # Resampling will check the particles list and find any particles with a low probability and replace them
    def Resampling(self, num_measurements, angles, dists):
        map_rec = np.zeros((self.size))
        re_id = np.random.choice(self.size, self.size, p=list(self.weights))
        new_particle_list = []
        for i in range(self.size):
            if map_rec[re_id[i]] == 0:
                self.particle_list[re_id[i]].Mapping(num_measurements, angles, dists)
                map_rec[re_id[i]] = 1
            new_particle_list.append(copy.deepcopy(self.particle_list[re_id[i]]))
        self.particle_list = new_particle_list
        self.weights = np.ones((self.size), dtype=float) / float(self.size)

    # Retrieves new movement and lidar measurements and applies to the particles. Also changes the weights of those particles
    def Feed(self, heading, turnAngle, encoder_dist, angles, dists):
        field = np.ones((self.size), dtype=float)
        for i in range(self.size):
            self.particle_list[i].Sampling(heading, turnAngle, encoder_dist)
            field[i] = self.particle_list[i].LikelihoodField(angles, dists)
        if (np.sum(field)!= 0):
            self.weights = field / np.sum(field)
        else:
            self.weights = np.ones((self.size), dtype=float) / self.size

    # Returns the best particle (highest weight)
    def get_best_particle(self):
        mid = np.argmax(self.weights)
        return self.particle_list[mid]

    # Returns the best estimated position
    def get_best_pose(self):
        return self.get_best_particle().pos.copy()
