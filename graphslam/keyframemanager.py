import numpy as np
# import subprocess
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
        kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index])
        self.keyframes.append(kf)

    def save_solution(self, x):
        for i in range(len(x)):
            self.keyframes[i].x = x[i]

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

    def compute_transformation_local(self, i, j, use_initial_transform=False):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j when j-i = 1.
        An initial estimate is used to compute using icp
        """
        # compute initial transform from odometry
        # TODO: Compute inintial transformation from IMU
        if use_initial_transform:
            # initial estimation
            xi = self.keyframes[i].x
            xj = self.keyframes[j].x
            Ti = HomogeneousMatrix([xi[0], xi[1], 0], Euler([0, 0, xi[2]]))
            Tj = HomogeneousMatrix([xj[0], xj[1], 0], Euler([0, 0, xj[2]]))
            Tij = Ti.inv() * Tj
            # muatb = Tij.t2v()
            transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=Tij.array)
            atb = HomogeneousMatrix(transform.transformation) #.t2v()
            return atb
        else:
            transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=np.eye(4))
            atb = HomogeneousMatrix(transform.transformation) #.t2v()
            return atb

    def compute_transformation_global(self, i, j):
        """
        Compute relative transformation using ICP from keyframe i to keyframe j.
        An initial estimate is used.
        FPFh to align and refine with icp
        """
        atb = self.keyframes[i].global_registration(self.keyframes[j])
        atb = HomogeneousMatrix(atb).t2v()
        return atb

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


class KeyFrame():
    def __init__(self, directory, scan_time):
        self.transform = None
        # voxel sizes
        self.voxel_size = 0.1
        self.voxel_size_normals = 3*self.voxel_size
        self.voxel_size_fpfh = 3*self.voxel_size
        self.icp_threshold = 3
        self.fpfh_threshold = 2
        filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'
        self.pointcloud = o3d.io.read_point_cloud(filename)
        # downsample pointcloud and save to pointcloud in keyframe
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

    def local_registration(self, other, initial_transform):
        """
        use icp to compute transformation using an initial estimate.
        caution, initial_transform is a np array.
        """
        print("Apply point-to-plane ICP")
        # print("Using threshold: ", self.icp_threshold)
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #         other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform,
        #         o3d.pipelines.registration.TransformationEstimationPointToPoint())
        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        # other.draw_registration_result(self, reg_p2p.transformation)
        return reg_p2p

    def global_registration(self, other):
        """
        perform global registration followed by icp
        """
        initial_transform = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            other.pointcloud, self.pointcloud, other.pointcloud_fpfh, self.pointcloud_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=self.fpfh_threshold))
        # other.draw_registration_result(self, initial_transform.transformation)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # other.draw_registration_result(self, reg_p2p.transformation)
        print(reg_p2p)
        print("Refined transformation is:")
        print(reg_p2p.transformation)
        return reg_p2p.transformation

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













