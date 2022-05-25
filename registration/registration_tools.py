import numpy as np
from tools.homogeneousmatrix import HomogeneousMatrix


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    T = HomogeneousMatrix(T)
    return T

def compute_occupancy_grid(points):
    """
    Computes a depth image from the pointcloud
    """
    nr = 1500
    nc = 1500

    max_m = 30
    # max_z = 20 # max height
    km = nr/(2*max_m)
    image = np.zeros((nr, nc))

    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    kz = 255.0/(max_z-min_z)


    for i in range(len(points)):
        [x, y, z] = points[i, :]
        if np.abs(x) > max_m or np.abs(y) > max_m:
            continue
        r = (x + max_m)*km
        c = (y + max_m)*km

        r = int(np.round(r))
        c = int(np.round(c))

        # try, occupancy grid or use z
        if r >= 0 and r < nr and c >= 0 and c < nc:
            # image[r, c] += 1
            zp = z - min_z
            # save max height for each cell
            if image[r, c] < zp:
                image[r, c] = zp
    # scale image
    # total = np.sum(np.sum(image))
    # image = (255.0/total)*image
    image = kz * image
    return image