"""
Runs a scanmatching algorithm on the point clouds.

Two variants of the scanmatching are applied.
method A: using the whole points clouds.
method B: segments the ground plane and estimates tz, alpha and beta. Next. tx, ty, gamma are computed using the whole
            point clouds.

Parameters:
    Parameters of the experiment:
        deltaxy, deltath: define the relative movement between scans to be processed.
    Parameters of the scanmatcher:
    class Keyframe():
        self.voxel_size = 0.1 --> the size of the voxels. Pointclouds are sampled_down using this size.
        self.voxel_size_normals = 3*self.voxel_size --> the radius to compute normals on each point.
        self.icp_threshold = 3 --> the distance to associate points in the ICP algorithm.

    TODO:
        Parameters should be stored in a yaml file.
"""
from eurocreader.eurocreader import EurocReader
from graphslam.keyframemanager import KeyFrameManager
from tools.homogeneousmatrix import HomogeneousMatrix
from tools.quaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
from config import PARAMETERS


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
    plt.plot(range(len(errors)), errors[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.title('Errors XYZ')
    plt.show(block=True)

    plt.figure()
    plt.plot(range(len(errors)), errors[:, 3], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 4], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(errors)), errors[:, 5], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.title('Errors Alfa Beta Gamma')
    plt.show(block=True)

    print("Covariance matrix: ")
    print(np.cov(errors.T))


def view_pos_data(data):
    plt.figure()
    plt.plot(range(len(data)), data[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(data)), data[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(data)), data[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.show(block=True)

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.show(block=True)


def view_orient_data(data):
    eul = []
    for dat in data:
        q = [dat[3], dat[0], dat[1], dat[2]]
        Q = Quaternion(q)
        th = Q.Euler()
        eul.append(th.abg)
    eul = np.array(eul)

    plt.figure()
    plt.plot(range(len(eul)), eul[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(eul)), eul[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    plt.plot(range(len(eul)), eul[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    # plt.legend()
    plt.show(block=True)


def save_transforms_to_file(transforms):
    import pickle
    pickle.dump(transforms, open('measured_transforms.pkl', "wb"))


def main():
    directory = PARAMETERS.directory
    # Prepare data
    euroc_read = EurocReader(directory=directory)
    # nmax_scans to limit the number of scans in the experiment
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=PARAMETERS.exp_deltaxy, deltath=PARAMETERS.exp_deltath, nmax_scans=PARAMETERS.exp_long)
    start = 0
    end = 100
    scan_times = scan_times[start:end]
    gt_pos = gt_pos[start:end]
    gt_orient = gt_orient[start:end]
    # view_pos_data(gt_pos)

    measured_transforms = []
    # create KeyFrameManager
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_all_keyframes()
    keyframe_manager.load_pointclouds()
    keyframe_manager.keyframes[0].pre_process()
    for i in range(1, len(scan_times)):
        print('Adding keyframe and computing transform: ', i, 'out of ', len(scan_times))
        keyframe_manager.keyframes[i].pre_process()
        # compute relative motion between scan i and scan i-1 0 1, 1 2...
        atb, rmse = keyframe_manager.compute_transformation_local_registration(i-1, i, method='B')
        measured_transforms.append(atb)

    # compute ground truth transformations: ground truth absolute and ground truth relative
    gt_transforms = compute_homogeneous_transforms(gt_pos, gt_orient)
    gt_transforms_relative = compute_homogeneous_transforms_relative(gt_transforms)
    # compare ICP measurements with ground_truth
    eval_errors(gt_transforms_relative, measured_transforms)

    save_transforms_to_file(measured_transforms)

    # view map with computed transforms
    keyframe_manager.set_relative_transforms(relative_transforms=measured_transforms)
    keyframe_manager.view_map(keyframe_sampling=5, point_cloud_sampling=10)


if __name__ == "__main__":
    main()
