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


# # now represent ground truth and solution
# x = []
# for kf in self.keyframes:
#     x.append(kf.x)
# x = np.array(x)
#
# plt.figure()
# # plot ground truth
# if xgt is not None:
#     xgt = np.array(xgt)
#     plt.plot(xgt[:, 0], xgt[:, 1], color='black', linestyle='dashed', marker='+',
#              markerfacecolor='black', markersize=10)
# # plot solution
# plt.plot(x[:, 0], x[:, 1], color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# # plt.scatter(points_global[:, 0], points_global[:, 1], color='blue')
# plt.show(block=True)

def read_measured_transforms():
    import pickle
    measured_transforms = pickle.load(open('measured_transforms.pkl', 'rb'))
    return measured_transforms


def main():
    # Prepare data
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas2'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=0.2, deltath=0.2,
                                                                         nmax_scans=None)
    measured_transforms = []
    # measured_transforms = read_measured_transforms()
    # create the graphslam graph
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, delta_index=180, xi2_th=20.0, d_th=8.0)
    # create keyframemanager and add initial observation
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    keyframe_manager.add_keyframe(0)
    keyframe_manager.keyframes[0].load_pointcloud()
    for i in range(1, len(scan_times)):
        print('Iteration (keyframe): ', i)
        # CAUTION: odometry is not used. ICP computed without any prior
        # compute relative motion between scan i and scan i-1 0 1, 1 2...
        keyframe_manager.add_keyframe(i)
        keyframe_manager.keyframes[i].load_pointcloud()
        atb = keyframe_manager.compute_transformation_local(i-1, i)
        # atb = measured_transforms[i-1]
        # consecutive edges. Adds a new node AND EDGE with restriction aTb
        graphslam.add_consecutive_observation(atb)
        # non-consecutive edges
        associations = dassoc.perform_data_association()
        for assoc in associations:
            i = assoc[0]
            j = assoc[1]
            # keyframe_manager.keyframes[i].load_pointcloud()
            # keyframe_manager.keyframes[j].load_pointcloud()
            # caution, need to use something with a prior
            itj = keyframe_manager.compute_transformation_global(i, j)
            atb = keyframe_manager.compute_transformation_local(i, j, initial_transform=itj)

            graphslam.add_non_consecutive_observation(i, j, atb)
            # keyframe_manager.view_map()
        # if len(associations):
            # optimizing whenever non_consecutive observations are performed (loop closing)
            # graphslam.optimize()
            # graphslam.view_solution()
            # keyframe_manager.save_solution(graphslam.get_solution())
            # keyframe_manager.view_map(keyframe_sampling=1, point_cloud_sampling=5)

        # or optimizing at every new observation, only new updates are optimized
        graphslam.optimize()
        graphslam.view_solution()
        estimated_transforms = graphslam.get_solution_transforms()
        # view map with computed transforms
        keyframe_manager.set_global_transforms(global_transforms=estimated_transforms)
        # keyframe_manager.view_map(keyframe_sampling=1, point_cloud_sampling=5)


    # keyframe_manager.save_solution(graphslam.get_solution())
    # keyframe_manager.view_map(xgt=odo_gt, sampling=10)
    # or optimizing when all information is available
    graphslam.optimize()
    graphslam.view_solution()


if __name__ == "__main__":
    main()
