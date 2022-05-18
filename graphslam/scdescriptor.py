"""
Scan Context descriptor, simplified
"""

import numpy as np
# import subprocess
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import matplotlib.pyplot as plt

class SCDescriptor():
    def __init__(self, maxR=15):
        """
        Initialize de descriptor class for the pointcloud
        """
        # self.points = points
        self.maxR = maxR

    def compute_descriptor(self, points):
        """
        The z coordinate is removed
        """
        points = np.asarray(points)
        x, y = points[:, 0], points[:, 1]
        rs = np.sqrt(x ** 2 + y ** 2)
        thetas = np.arctan2(y, x) + np.pi
        # plt.figure()
        # plt.plot(range(len(thetas)), thetas)
        # plt.plot(range(len(rs)), rs)
        # plt.show()

        # compute descriptor as 2d histogram
        redges = np.linspace(0, self.maxR, 9)
        thetaedges = np.linspace(0, 2*np.pi, 60)
        H, thetaedges, redges = np.histogram2d(thetas, rs, bins=(thetaedges, redges))

        # fig = plt.figure(figsize=(7, 3))
        # ax = fig.add_subplot(131, title='imshow: square bins')
        # plt.imshow(H.T, interpolation='nearest', origin='lower',
        #            extent=[thetaedges[0], thetaedges[-1], redges[0], redges[-1]])
        H = H/len(points)
        return H












