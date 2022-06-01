"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from eurocreader.eurocreader import EurocReader
from graphslam.dataassociation import DataAssociation
from graphslam.graphslam import GraphSLAM
# import gtsam
from graphslam.keyframemanager import KeyFrameManager
import numpy as np

# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
from tools.homogeneousmatrix import HomogeneousMatrix


def perform_data_associations_ground_truth(gt_pos, current_index, delta_index=160, euclidean_distance_threshold=6.5):
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
    directory = '/home/arvc/Escritorio/develop/Registration/dos_vueltas_features/'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.1, deltath=0.05,
                                                                         nmax_scans=None)
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_all_keyframes()
    keyframe_manager.load_pointclouds()
    for k in range(0, len(scan_times)):
        print('Iteration (keyframe): ', k)
        # keyframe_manager.add_keyframe(i)
        # keyframe_manager.keyframes[i].load_pointcloud()
        # keyframe_manager.keyframes[i].pre_process()
        associations = perform_data_associations_ground_truth(gt_pos, k)
        for assoc in associations:
            i = assoc[0]
            j = assoc[1]
            # keyframe_manager.keyframes[i].load_pointcloud()
            # keyframe_manager.keyframes[j].load_pointcloud()
            keyframe_manager.keyframes[i].pre_process()
            keyframe_manager.keyframes[j].pre_process()

            # caution, need to use something with a prior
            itj, prob = keyframe_manager.compute_transformation_global_registration(i, j)
            atb, rmse = keyframe_manager.compute_transformation_local_registration(i, j, method='B', initial_transform=itj.array)




if __name__ == "__main__":
    main()
