"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from eurocreader.eurocreader import EurocReader
from graphslam.dataassociation import DataAssociation
from graphslam.graphslam import GraphSLAM
import gtsam
from graphslam.keyframemanager import KeyFrameManager
import numpy as np

# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
from tools.homogeneousmatrix import HomogeneousMatrix

prior_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 0.2
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
icp_xyz_sigma = 0.01
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 0.05

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma]))
ICP_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([icp_rpy_sigma*np.pi/180,
                                                       icp_rpy_sigma*np.pi/180,
                                                       icp_rpy_sigma*np.pi/180,
                                                       icp_xyz_sigma,
                                                       icp_xyz_sigma,
                                                       icp_xyz_sigma]))


def perform_data_associations_ground_truth(gt_pos, current_index, delta_index=80, euclidean_distance_threshold=5.0):
    candidates = []
    distances = []
    i = current_index-1
    for j in range(i-delta_index):
        d = np.linalg.norm(gt_pos[i]-gt_pos[j])
        # dth = np.linalg.norm(gt_orient[i]-gt_orient[j])
        distances.append(d)
        if d < euclidean_distance_threshold:
            candidates.append([i, j])
    return candidates


def main():
    # Prepare data
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas2'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.2, deltath=0.2,
                                                                         nmax_scans=None)
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    for i in range(0, len(scan_times)):
        print('Iteration (keyframe): ', i)
        keyframe_manager.add_keyframe(i)
        # keyframe_manager.keyframes[i].load_pointcloud()
        associations = perform_data_associations_ground_truth(gt_pos, i)
        for assoc in associations:
            i = assoc[0]
            j = assoc[1]
            keyframe_manager.keyframes[i].load_pointcloud()
            keyframe_manager.keyframes[j].load_pointcloud()
            # caution, need to use something with a prior
            itj, Oij = keyframe_manager.compute_transformation_global(i, j)
            # atb, Oij = keyframe_manager.compute_transformation_local(i, j, initial_transform=itj.array)


if __name__ == "__main__":
    main()
