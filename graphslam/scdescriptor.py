"""
Scan Context descriptor, simplified
"""
import numpy as np
import matplotlib.pyplot as plt


class SCDescriptor():
    def __init__(self, max_radius=35):
        """
        Initialize de descriptor class for the pointcloud
        """
        self.max_radius = max_radius
        self.nr = 40
        self.nc = 80
        self.descriptor = np.zeros((self.nr, self.nc))

    def compute_descriptor(self, points):
        """
        Each voxel in r, theta coordinates stores the max z value is stored.
        Transform all point cloud to the center of gravity of the c
        Option: store a mean z value.
        Algorithm:
        1) Find the mean mu of all points in the pointcloud.
        2) Transform all points so that mu=[0,0,0].
        3) Transform z so that min(z)=0.
        4) Compute theta = atan2(x, y). Compute radius = sqrt(x^2+y^2)
           radius is normalized
        5) Form an image with rows, columns where I(r, c) = max(z)
        """
        # reset descriptor
        self.descriptor = np.zeros((self.nr, self.nc))
        points = np.asarray(points)
        # mu = np.mean(points, axis=0)
        # move to the center of gravity
        # points = points - mu
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        # z positive
        min_z = np.min(z)
        z = z - min_z + 1.0

        # compute columns based on theta
        thetas = np.arctan2(y, x) + np.pi
        c = np.round(self.nc * thetas / (2 * np.pi))
        c = np.clip(c, 0, self.nc - 1)
        # compute rows based on radius
        Rs = np.sqrt(x ** 2 + y ** 2)

        # normalize
        # Rs = self.max_radius*Rs/np.mean(Rs)
        # clip value of distance
        # Rs = np.clip(Rs, 0, self.max_radius)
        # plt.figure()
        # plt.plot(Rs)
        # plt.show()
        r = np.round(self.nr * Rs / self.max_radius)
        r = np.clip(r, 0, self.nr - 1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(thetas, Rs)
        # plt.show()
        #
        # plt.figure()
        # plt.scatter(c, r)
        # plt.show()

        # for each z store in the descriptor the max value.
        for i in range(len(points)):
            row = int(r[i])
            col = int(c[i])
            if self.descriptor[row, col] < z[i]:
                self.descriptor[row, col] = z[i]
        return self.descriptor

    def maximize_correlation(self, other):
        debug = True
        sc1 = self.descriptor
        sc2 = other.descriptor
        corrs = []
        # caution, computing on [0, 2pi)
        delta_r = 0
        for i in range(self.nc-delta_r):
            # Shift one column sc2 and compute correlation
            sc2 = np.roll(sc2, 1, axis=1)  # column shift
            corr = compute_correlation(sc1, sc2)
            corrs.append(corr)
        corrs = 1.0-np.array(corrs)
        # normalize to probability. Probability fo yaw being a good correlation
        probs = corrs/np.sum(corrs)

        if debug:
            plt.figure()
            plt.plot(2 * np.pi*np.arange(self.nc-delta_r)/self.nc, probs)
            # plt.plot(np.arange(self.nc - delta_r), probs)
            plt.show()

        col_diff = np.argmax(probs)
        yaw_diff = 2 * np.pi * col_diff / self.nc
        maxprob = np.max(probs)
        print('Found gamma is: ', yaw_diff)
        print('Max prob is:', maxprob)
        return yaw_diff, maxprob


def compute_correlation(sc1, sc2):
    """
    Compute correlation between two descriptors.
    """
    import matplotlib.pyplot as plt
    nr = sc1.shape[1]
    nc = sc1.shape[0]
    dists = []
    for i in range(nr):
        a = np.dot(sc1[:, i], sc2[:, i])
        b = np.linalg.norm(sc1[:, i])
        c = np.linalg.norm(sc2[:, i])
        # plt.figure()
        # plt.plot(range(nc), sc1[:, i])
        # plt.plot(range(nc), sc2[:, i])
        # plt.show()
        if b*c == 0:
            continue
        d = 1 - a/(b*c)
        dists.append(d)
    return np.mean(np.array(dists))












