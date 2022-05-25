"""
Build a point cloud map in global coordinates using
A series of
"""
from eurocreader.eurocreader import EurocReader
from graphslam.keyframemanager import KeyFrameManager
from tools.homogeneousmatrix import HomogeneousMatrix
from tools.quaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas_long_range'
    # Prepare data
    euroc_read = EurocReader(directory=directory)
    # nmax_scans to limit the number of scans in the experiment
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.2, deltath=0.2,
                                                                         nmax_scans=None)
    # create KeyFrameManager
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    for i in range(0, len(scan_times)):
        print('Adding keyframe: ', i, 'out of ', len(scan_times))
        keyframe_manager.add_keyframe(i)
        keyframe_manager.keyframes[i].load_pointcloud()
    # compute ground truth transformations: ground truth absolute and ground truth relative
    gt_transforms = compute_homogeneous_transforms(gt_pos, gt_orient)

    # view map with ground truth transforms
    keyframe_manager.set_global_transforms(global_transforms=gt_transforms)
    keyframe_manager.view_map(keyframe_sampling=5, point_cloud_sampling=5)
    keyframe_manager.save_to_file(filename='map.pcd', keyframe_sampling=5, point_cloud_sampling=5)

    # view map with ground truth transforms
    # keyframe_manager.set_global_transforms(global_transforms=gt_transforms)
    # keyframe_manager.view_map(keyframe_sampling=5, point_cloud_sampling=5)
    # equivalent: use relative transforms to compute the global map
    # keyframe_manager.set_relative_transforms(relative_transforms=gt_transforms_relative)
    # keyframe_manager.view_map(keyframe_sampling=30, point_cloud_sampling=20)

if __name__ == "__main__":
    main()
