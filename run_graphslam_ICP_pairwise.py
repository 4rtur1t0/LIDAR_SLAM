"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from config import PARAMETERS
from eurocreader.eurocreader import EurocReader
from graphslam.dataassociation import DataAssociation
from graphslam.graphslam import GraphSLAM
import gtsam
from graphslam.keyframemanager import KeyFrameManager
import numpy as np

prior_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 0.2
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
icp_xyz_sigma = 0.001
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 0.005

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
ICP_NOISE_DA = np.diag([prior_xyz_sigma, prior_xyz_sigma, prior_rpy_sigma])

def read_measured_transforms():
    import pickle
    filename = PARAMETERS.directory + '/measured_transforms_0.1_0.05.pkl'
    measured_transforms = pickle.load(open(filename, 'rb'))
    return measured_transforms


def compute_transformations(keyframe_manager, candidates):
    transformations = []
    for assoc in candidates:
        i = assoc[0]
        j = assoc[1]
        keyframe_manager.keyframes[i].pre_process()
        keyframe_manager.keyframes[j].pre_process()
        # a global registration with a rough alignment
        itj, prob = keyframe_manager.compute_transformation_global_registration(i, j)
        # followed by a more subtle icp
        atb, rmse = keyframe_manager.compute_transformation_local_registration(i, j, method='B',
                                                                               initial_transform=itj.array)
        transformations.append(atb)
    return transformations


def main():
    debug = False
    # Prepare data
    directory = PARAMETERS.directory
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.1, deltath=0.05,
                                                                         nmax_scans=None)
    # measured_transforms = []
    measured_transforms = read_measured_transforms()
    # create the graphslam graph
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, delta_index=200, xi2_th=3.0, icp_noise=ICP_NOISE_DA)
    dassoc.dijkstra_algorithm.add_node()
    # create keyframemanager and add initial observation
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_all_keyframes()
    keyframe_manager.load_pointclouds()
    keyframe_manager.keyframes[0].pre_process()
    for k in range(1, len(scan_times)):
        print('Iteration (keyframe): ', k)
        # CAUTION: odometry is not used. ICP computed without any prior
        # compute relative motion between scan i and scan i-1 0 1, 1 2...
        # keyframe_manager.keyframes[k].pre_process()
        # atb = keyframe_manager.compute_transformation_local(k-1, k)
        atb = measured_transforms[k-1]
        # consecutive edges. Adds a new node AND EDGE with restriction aTb
        graphslam.add_consecutive_observation(atb)
        dassoc.dijkstra_algorithm.add_node()
        dassoc.dijkstra_algorithm.connect_nodes(k-1, k)

        # non-consecutive edges. Filter data associations
        candidates = dassoc.find_initial_candidates()
        if len(candidates) >= 4:
            # compute all candidate transformations using the candidates
            transformations = compute_transformations(keyframe_manager, candidates)
            filtered_candidates, filtered_transformations = dassoc.filter_data_associations(candidates, transformations,
                                                                                        min_number_of_candidates=4)
            for c in range(len(filtered_candidates)):
                i = filtered_candidates[c][0]
                j = filtered_candidates[c][1]
                atb = filtered_transformations[c]
                graphslam.add_loop_closing_observation(i, j, atb)
            # graphslam.optimize()

        graphslam.optimize()
        graphslam.view_solution2D_fast(skip=10)
        # graphslam.view_solution()
        if debug:
            estimated_transforms = graphslam.get_solution_transforms()
            keyframe_manager.set_global_transforms(global_transforms=estimated_transforms)
            keyframe_manager.view_map(keyframe_sampling=15, point_cloud_sampling=50)


    # keyframe_manager.save_solution(graphslam.get_solution())
    keyframe_manager.view_map(keyframe_sampling=15, point_cloud_sampling=50)
    # or optimizing when all information is available
    # graphslam.optimize()
    graphslam.view_solution()


if __name__ == "__main__":
    main()
