"""
View correspondences
https://github.com/hankyang94/teaser_fpfh_threedmatch_python
"""

import numpy as np
# import subprocess
from graphslam.scdescriptor import SCDescriptor
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import open3d as o3d
import copy

class KeyFrameManager():
    def __init__(self, directory, scan_times):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = scan_times
        self.keyframes = []

    def add_keyframe(self, index):
        kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index], index=index)
        self.keyframes.append(kf)

    def save_solution(self, transforms):
        for i in range(len(transforms)):
            self.keyframes[i].transform = transforms[i]

    def set_relative_transforms(self, relative_transforms):
        """
        Given a set of relative transforms. Assign to each keyframe a global transform by
        postmultiplication.
        Caution, computing global transforms from relative transforms starting from T0=I
        """
        T = HomogeneousMatrix(np.eye(4))
        global_transforms = [T]
        for i in range(len(relative_transforms)):
            T = T*relative_transforms[i]
            global_transforms.append(T)

        for i in range(len(self.keyframes)):
            self.keyframes[i].set_global_transform(global_transforms[i])

    def set_global_transforms(self, global_transforms):
        """
        Assign the global transformation for each of the keyframes.
        """
        for i in range(len(self.keyframes)):
            self.keyframes[i].set_global_transform(global_transforms[i])

    def compute_transformation_local(self, i, j, initial_transform=np.eye(4)):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j when j-i = 1.
        An initial estimate is used to compute using icp
        """
        # compute initial transform from odometry
        # # TODO: Compute inintial transformation from IMU
        # if use_initial_transform:
        #     # initial estimation
        #     # xi = self.keyframes[i].x
        #     # xj = self.keyframes[j].x
        #     Ti = self.keyframes[i].transform # HomogeneousMatrix([xi[0], xi[1], 0], Euler([0, 0, xi[2]]))
        #     Tj = self.keyframes[j].transform # HomogeneousMatrix([xj[0], xj[1], 0], Euler([0, 0, xj[2]]))
        #     Tij = Ti.inv() * Tj
        #     # muatb = Tij.t2v()
        #     transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=Tij.array)
        #     atb = HomogeneousMatrix(transform.transformation)
        #     return atb
        # else:
        transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=initial_transform)
        atb = HomogeneousMatrix(transform.transformation)
        return atb

    def compute_transformation_global(self, i, j):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j.
        An initial estimate is used.
        FPFh to align and refine with icp
        """
        atb, inf_atb = self.keyframes[i].global_registration(self.keyframes[j])
        atb = HomogeneousMatrix(atb)
        return atb, inf_atb

    def view_map(self, keyframe_sampling=10, point_cloud_sampling=1000):
        print("COMPUTING MAP FROM KEYFRAMES")
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(0, len(self.keyframes), keyframe_sampling):
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            kf = self.keyframes[i]
            # transform to global and
            pointcloud_temp = kf.transform_to_global(point_cloud_sampling=point_cloud_sampling)
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
        # draw the whole map
        o3d.visualization.draw_geometries([pointcloud_global])

    def save_to_file(self, filename, keyframe_sampling=10, point_cloud_sampling=100):
        print("COMPUTING MAP FROM KEYFRAMES")
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(0, len(self.keyframes), keyframe_sampling):
            print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
            kf = self.keyframes[i]
            # transform to global and
            pointcloud_temp = kf.transform_to_global(point_cloud_sampling=point_cloud_sampling)
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
        print("SAVING MAP AS: ", filename)
        o3d.io.write_point_cloud(filename, pointcloud_global)


class KeyFrame():
    def __init__(self, directory, scan_time, index):
        # the estimated transform of this keyframe with respect to global coordinates
        self.transform = None
        # voxel sizes
        self.voxel_size = 0.1
        self.voxel_size_normals = 3*self.voxel_size
        self.voxel_size_fpfh = 3*self.voxel_size
        self.icp_threshold = 5
        self.fpfh_threshold = 2
        self.index = index
        self.timestamp = scan_time
        self.filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'
        self.scdescriptor = SCDescriptor(maxR=15.0)

    def load_pointcloud(self):
        self.pointcloud = o3d.io.read_point_cloud(self.filename)
        # downsample pointcloud and save to pointcloud in keyframe
        # self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        # calcular las normales a cada punto
        # self.pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
        #                                                                       max_nn=30))
        # extraer los Fast Point Feature Histograms
        # self.pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud,
        #                                                                        o3d.geometry.KDTreeSearchParamHybrid(
        #                                                                            radius=self.voxel_size_fpfh,
        #                                                                            max_nn=100))
        # self.draw_cloud()
        # self.descriptorvector = self.scdescriptor.compute_descriptor(self.pointcloud.points)

    def pre_process(self):
        self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        # calcular las normales a cada punto
        self.pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                               max_nn=30))
        # extraer los Fast Point Feature Histograms
        self.pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud,
                                                                               o3d.geometry.KDTreeSearchParamHybrid(
                                                                                   radius=self.voxel_size_fpfh,
                                                                                   max_nn=100))
        # self.draw_cloud()
        self.descriptorvector = self.scdescriptor.compute_descriptor(self.pointcloud.points)

    def compute_normals(self):
        self.pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=30))

    def compute_descriptor(self):
        self.descriptorvector = self.scdescriptor.compute_descriptor(self.pointcloud.points)

    def segment_plane(self):
        """
        use icp to compute transformation using an initial estimate.
        caution, initial_transform is a np array.
        """
        plane_model, inliers = self.pointcloud.segment_plane(distance_threshold=0.05,
                                                             ransac_n=3,
                                                             num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        plane_cloud = self.pointcloud.select_by_index(inliers)
        # plane_cloud.paint_uniform_color([1.0, 0, 0])
        non_plane_cloud = self.pointcloud.select_by_index(inliers, invert=True)
        # o3d.visualization.draw_geometries([plane_cloud, non_plane_cloud],
        #                                   zoom=0.8,
        #                                   front=[-0.4999, -0.1659, -0.8499],
        #                                   lookat=[2.1813, 2.0619, 2.0999],
        #                                   up=[0.1204, -0.9852, 0.1215])
        return plane_cloud, non_plane_cloud

    def local_registration(self, other, initial_transform):
        """
        use icp to compute transformation using an initial estimate.
        caution, initial_transform is a np array.
        """
        # other.draw_registration_result(self, initial_transform)
        print("Apply point-to-plane ICP")
        print("Using threshold: ", self.icp_threshold)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        max_correspondence_fine = 0.1
        other.draw_registration_result(self, reg_p2p.transformation)
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            other.pointcloud, self.pointcloud, max_correspondence_fine, reg_p2p.transformation)
        return reg_p2p, information_icp

    def global_registration(self, other):
        """
        perform global registration followed by icp
        """
        voxel_size = 0.5
        radius_normal = voxel_size * 2
        radius_feature = voxel_size * 2
        # remove plane remove points corresponding to plane
        _, otherpcd = other.segment_plane()
        _, pcd = self.segment_plane()
        otherpcd = otherpcd.voxel_down_sample(voxel_size)
        pcd = pcd.voxel_down_sample(voxel_size)
        otherpcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
        # otherpcd.paint_uniform_color([1.0, 0, 0])
        # pcd.paint_uniform_color([0, 1.0, 0])
        # o3d.visualization.draw_geometries([otherpcd, pcd],
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
        #                                   up=[-0.0694, -0.9768, 0.2024])
        # extraer los Fast Point Feature Histograms
        otherpcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(otherpcd,
                                                                        o3d.geometry.KDTreeSearchParamHybrid(
                                                                        radius=radius_feature,
                                                                        max_nn=100))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature,
                                                                   max_nn=100))

        # self.view_correspondences(pcd, pcd_fpfh, otherpcd, otherpcd_fpfh)

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        max_correspondence_distance = 0.5
        global_transform = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            otherpcd, pcd, otherpcd_fpfh, pcd_fpfh,
            mutual_filter=True,
            max_correspondence_distance=max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            ransac_n=5,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.99))
        other.draw_registration_result(self, global_transform.transformation)
        # Now refine the previous registration
        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud, self.pointcloud, self.icp_threshold, global_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        other.draw_registration_result(self, reg_p2p.transformation)
        # now compute the information matrix based ont fine corresopndences
        max_correspondence_fine = 0.1
        # other.draw_registration_result(self, reg_p2p.transformation)
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            other.pointcloud, self.pointcloud, max_correspondence_fine, reg_p2p.transformation)
        return global_transform.transformation, information_icp

    def view_correspondences(self, pcd, pcd_fpfh, otherpcd, otherpcd_fpfh):
        correspondences = []
        feature_dists = []
        fpfh_tree = o3d.geometry.KDTreeFlann(pcd_fpfh)
        for i in range(len(otherpcd.points)):
            [_, idx, _] = fpfh_tree.search_knn_vector_xd(otherpcd_fpfh.data[:, i], 1)
            feature_dist = np.linalg.norm(otherpcd_fpfh.data[:, i]-pcd_fpfh.data[:, idx])
            feature_dists.append(feature_dist)
            dis = np.linalg.norm(otherpcd.points[i] - pcd.points[idx[0]])
            c = (0.2 - np.fmin(dis, 0.2)) / 0.2
            pcd.colors[i] = [c, c, c]
            correspondences.append([i, idx[0]])
        print(correspondences)
        print(np.mean(feature_dists))

    def draw_registration_result(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud)
        target_temp = copy.deepcopy(other.pointcloud)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                          zoom=0.4459,
                                          front=[0.9288, -0.2951, -0.2242],
                                          lookat=[1.6784, 2.0612, 1.4451],
                                          up=[-0.3402, -0.9189, -0.1996])

    def draw_cloud(self):
        o3d.visualization.draw_geometries([self.pointcloud],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

    def set_global_transform(self, transform):
        self.transform = transform
        return

    def transform_to_global(self, point_cloud_sampling=10):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        T = HomogeneousMatrix(self.transform)
        pointcloud = self.pointcloud.uniform_down_sample(every_k_points=point_cloud_sampling)
        return pointcloud.transform(T.array)













