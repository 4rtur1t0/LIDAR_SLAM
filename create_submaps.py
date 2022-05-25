"""
Runs a scanmatching algorithm on the point clouds.

Creates submaps sequentially

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




def main():
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas_long_range'
    # Prepare data
    euroc_read = EurocReader(directory=directory)
    # nmax_scans to limit the number of scans in the experiment
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.1, deltath=0.05, nmax_scans=None)
    # start = 0
    # end = 100
    # scan_times = scan_times[start:end]
    # gt_pos = gt_pos[start:end]
    # gt_orient = gt_orient[start:end]
    # view_pos_data(gt_pos)
    submap_manager = SubMapManager(directory=directory, scan_times=scan_times)
    submap_manager.create_submaps(n=5)


    # compute ground truth transformations: ground truth absolute and ground truth relative
    # gt_transforms = compute_homogeneous_transforms(gt_pos, gt_orient)
    # gt_transforms_relative = compute_homogeneous_transforms_relative(gt_transforms)
    # compare ICP measurements with ground_truth
    # eval_errors(gt_transforms_relative, measured_transforms)
    #
    # save_transforms_to_file(measured_transforms)

    # view map with computed transforms
    keyframe_manager.set_relative_transforms(relative_transforms=measured_transforms)
    keyframe_manager.view_map(keyframe_sampling=20, point_cloud_sampling=150)





if __name__ == "__main__":
    main()
