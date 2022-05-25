"""
Scan Context descriptor, simplified
"""
import numpy as np


class SCDescriptor():
    def __init__(self, maxR=35):
        """
        Initialize de descriptor class for the pointcloud
        """
        self.maxR = maxR
        # self.nr = 20
        # self.nc = 60
        self.nr = 30
        self.nc = 100
        self.descriptor = np.zeros((self.nr, self.nc))
        # self.max_length = 80

    def compute_descriptor(self, points):
        """
        Each voxel in r, theta coordinates stores the max z value is stored.
        Transform all point cloud to the center of gravity of the c
        Option: store a mean z value.
        """
        points = np.asarray(points)
        # [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        mu = np.mean(points, axis=0)
        # move to the center of gravity
        points = points - mu
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        # compute radius and thetas
        rs = np.sqrt(x ** 2 + y ** 2)
        thetas = np.arctan2(y, x) + np.pi
        min_z = np.min(z)
        z = z - min_z

        for i in range(len(points)):
            # [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            c = int(np.round(self.nc*thetas[i]/(2*np.pi)))
            # clip value of distance
            R = np.clip(rs[i], 0, self.maxR)
            r = int(np.round(self.nr*R/self.maxR))
            r = np.clip(r, 0, self.nr-1)
            c = np.clip(c, 0, self.nc-1)
            if self.descriptor[r, c] < z[i]:
                self.descriptor[r, c] = z[i]

        return self.descriptor

    def compute_descriptor2(self, points):
        points = np.asarray(points)
        self.descriptor = ptcloud2sc(points, [20, 60], self.max_length)
        return self.descriptor

    def maximize_correlation(self, other):
        sc1 = self.descriptor
        sc2 = other.descriptor
        corrs = []
        for i in range(self.nc):
            # Shift one column sc2
            sc2 = np.roll(sc2, 1, axis=1)  # column shift
            corr = compute_correlation(sc1, sc2)
            corrs.append(corr)
        corrs = np.array(corrs)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(range(self.nc), corrs)
        # plt.show()

        col_diff = np.argmin(corrs)  # because python starts with 0
        yaw_diff = 2 * np.pi * col_diff / self.nc
        # dist = np.min(corrs)
        # obtain 2 lowest values
        min1, min2 = np.partition(corrs, 1)[0:2]
        dist = (1-min1)/(1-min2)
        return dist, yaw_diff

# def xy2theta(x, y):
#     if (x >= 0 and y >= 0):
#         theta = 180 / np.pi * np.arctan(y / x);
#     if (x < 0 and y >= 0):
#         theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)));
#     if (x < 0 and y < 0):
#         theta = 180 + ((180 / np.pi) * np.arctan(y / x));
#     if (x >= 0 and y < 0):
#         theta = 360 - ((180 / np.pi) * np.arctan((-y) / x));
#
#     return theta

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


def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    x = point[0]
    y = point[1]
    # z = point[2]

    # if (x == 0.0):
    #     x = 0.001
    # if (y == 0.0):
    #     y = 0.001

    # theta = xy2theta(x, y)
    theta = np.arctan2(y, x) + np.pi
    faraway = np.sqrt(x * x + y * y)

    idx_ring = np.divmod(faraway, gap_ring)[0]
    idx_sector = np.divmod(theta, gap_sector)[0]

    if (idx_ring >= num_ring):
        idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

    return int(idx_ring), int(idx_sector)


def ptcloud2sc(ptcloud, sc_shape, max_length):
    num_ring = sc_shape[0]
    num_sector = sc_shape[1]

    gap_ring = max_length / num_ring
    gap_sector = 2*np.pi / num_sector

    enough_large = 500
    sc_storage = np.zeros([enough_large, num_ring, num_sector])
    sc_counter = np.zeros([num_ring, num_sector])

    num_points = ptcloud.shape[0]
    for pt_idx in range(num_points):
        point = ptcloud[pt_idx, :]
        point_height = point[2] + 2.0  # for setting ground is roughly zero
        idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

        if sc_counter[idx_ring, idx_sector] >= enough_large:
            continue
        sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
        sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

    sc = np.amax(sc_storage, axis=0)

    return sc


def sc2rk(sc):
    return np.mean(sc, axis=1)


def maximize_correlation(sc1, sc2):
    num_sectors = sc1.shape[1]

    # repeate to move 1 columns
    _one_step = 1  # const
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = np.roll(sc1, _one_step, axis=1)  # column shift

        # compare
        sum_of_cossim = 0
        num_col_engaged = 0
        for j in range(num_sectors):
            col_j_1 = sc1[:, j]
            col_j_2 = sc2[:, j]
            if (~np.any(col_j_1) or ~np.any(col_j_2)):
                # to avoid being divided by zero when calculating cosine similarity
                # - but this part is quite slow in python, you can omit it.
                continue

            cossim = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
            sum_of_cossim = sum_of_cossim + cossim

            num_col_engaged = num_col_engaged + 1

        # save
        sim_for_each_cols[i] = sum_of_cossim / num_col_engaged

    yaw_diff = np.argmax(sim_for_each_cols) + 1  # because python starts with 0
    yaw_diff = 2*np.pi*yaw_diff/num_sectors-np.pi
    sim = np.max(sim_for_each_cols)
    dist = 1 - sim

    return dist, yaw_diff










