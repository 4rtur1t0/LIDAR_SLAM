"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from eurocreader.eurocreader import EurocReader
from graphslam.keyframemanager import KeyFrameManager
import numpy as np
from tools.homogeneousmatrix import HomogeneousMatrix
# from tools.euler import Euler
import matplotlib.pyplot as plt
from tools.quaternion import Quaternion
from tools.euler import Euler


def compute_homogeneous_transforms(gt_pos, gt_orient):
    transforms = []
    for i in range(len(gt_pos)):
        # CAUTION: THE ORDER IN THE QUATERNION class IS [qw, qx qy qz]
        # the order in ROS is [qx qy qz qw]
        q = [gt_orient[i][3], gt_orient[i][0], gt_orient[i][1], gt_orient[i][2]]
        Q = Quaternion(q)
        Ti = HomogeneousMatrix(gt_pos[i], Q)
        transforms.append(Ti)
    return transforms


def compute_homogeneous_transforms_relative(transforms):
    transforms_relative = []
    # compute relative transformations
    for i in range(len(transforms) - 1):
        Ti = transforms[i]
        Tj = transforms[i + 1]
        Tij = Ti.inv() * Tj
        transforms_relative.append(Tij)
    return transforms_relative


def eval_errors(ground_truth_transforms, measured_transforms):
    # compute xyz alpha beta gamma
    gt_tijs = []
    meas_tijs = []
    for i in range(len(ground_truth_transforms)):
        gt_tijs.append(ground_truth_transforms[i].t2v(n=3))  # !!! convert to x y z alpha beta gamma
        meas_tijs.append(measured_transforms[i].t2v(n=3))

    gt_tijs = np.array(gt_tijs)
    meas_tijs = np.array(meas_tijs)
    errors = gt_tijs-meas_tijs

    plt.figure()
    plt.plot(range(len(errors)), errors[:, 5], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.title('Errors Gamma')
    plt.show(block=True)

    print("Covariance in angle: ")
    return np.cov(errors[:, 5])



def main():
    # Prepare data
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas_long_range'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.5, deltath=0.5,
                                                                         nmax_scans=None)
    # First loop closing [493, 0]
    # moree loop closing [500, 10]
    # moree loop closing [505, 15]
    a = 500
    b = 10

    # measured_transforms = []
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_all_keyframes()
    keyframe_manager.load_pointclouds()

    keyframe_manager.keyframes[a].pre_process()
    keyframe_manager.keyframes[b].pre_process()

    # caution, need to use something with a prior
    itj, prob = keyframe_manager.compute_transformation_global_registration(a, b)
    atb, rmse = keyframe_manager.compute_transformation_local_registration(a, b, method='B', initial_transform=itj.array)


if __name__ == "__main__":
    main()
