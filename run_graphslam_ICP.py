"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
# from eurocreader.bagreader import RosbagReader
from eurocreader.eurocreader import EurocReader
from graphslam.dataassociation import DataAssociation
from graphslam.graphslam import GraphSLAM
import gtsam
from graphslam.keyframemanager import KeyFrameManager
import numpy as np

# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
prior_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 2
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
icp_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 2

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

def main():
    # Prepare data
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.2, deltath=0.02,
                                                                         nmax_scans=None)
    # create KeyFrameManager
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_keyframe(0)

    # create the graphslam graph
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, delta_index=80, xi2_th=20.0)
    measured_transforms = []
    # create keyframemanager and add initial observation
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_keyframe(0)
    for i in range(1, len(scan_times)):
        # CAUTION: odometry is not used. ICP computed without any prior
        keyframe_manager.add_keyframe(i)
        # compute relative motion between scan i and scan i-1 0 1, 1 2...
        atb = keyframe_manager.compute_transformation_local(i-1, i)
        # consecutive edges. Adds a new node AND EDGE with restriction aTb
        graphslam.add_consecutive_observation(atb)
        # non-consecutive edges
        associations = dassoc.perform_data_association()
        for assoc in associations:
            # graphslam.view_solution()
            i = assoc[0]
            j = assoc[1]
            atb = keyframe_manager.compute_transformation_global(i, j)
            graphslam.add_non_consecutive_observation(i, j, atb)
            measured_transforms.append(atb)
            # keyframe_manager.view_map()
        if len(associations):
            # graphslam.view_solution()
            # optimizing whenever non_consecutive observations are performed (loop closing)
            graphslam.optimize()
            # graphslam.view_solution()
            keyframe_manager.save_solution(graphslam.get_solution())
            # keyframe_manager.view_map(xgt=odo_gt)

        # graphslam.view_solution()
        # or optimizing at every new observation
        graphslam.optimize()
        # view map with computed transforms
        keyframe_manager.set_relative_transforms(relative_transforms=measured_transforms)
        keyframe_manager.view_map(keyframe_sampling=30, point_cloud_sampling=20)

    graphslam.view_solution()
    keyframe_manager.save_solution(graphslam.get_solution())
    # keyframe_manager.view_map(xgt=odo_gt, sampling=10)
    # or optimizing when all information is available
    graphslam.optimize()


if __name__ == "__main__":
    main()
