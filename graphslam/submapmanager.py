"""
The KeyFrameManager Stores a list of keyframes. Each keyframe is identified by a scan time and an index.

Methods provide a way to build a global map and save it to file.

"""
import numpy as np
from graphslam.keyframe import KeyFrame
from graphslam.keyframemanager import KeyFrameManager
from tools.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d


class SubMapManager():
    def __init__(self, directory, scan_times):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = scan_times
        # self.keyframes = []
        # create KeyFrameManager
        self.keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)

    def create_submaps(self):
        measured_transforms = []

        self.keyframe_manager.add_keyframe(0)
        self.keyframe_manager.keyframes[0].load_pointcloud()
        self.keyframe_manager.keyframes[0].pre_process()
        T = HomogeneousMatrix(np.eye(4))
        j = 0
        for i in range(1, len(self.scan_times)):
            print('Adding keyframe and computing transform: ', i, 'out of ', len(self.scan_times))
            self.keyframe_manager.add_keyframe(i)
            self.keyframe_manager.keyframes[i].load_pointcloud()
            self.keyframe_manager.keyframes[i].pre_process()
            # compute relative motion between scan i and scan i-1 0 1, 1 2...
            atb = self.keyframe_manager.compute_transformation_local(i - 1, i, method='B')
            # atb = keyframe_manager.compute_transformation_global(i - 1, i, method='B')
            measured_transforms.append(atb)
            j += 1
            if j > self.n

    def add_submap(self, index):
        kf = KeyFrame(directory=self.directory, scan_time=self.scan_times[index], index=index)
        self.keyframes.append(kf)

    # def save_submap(self, transforms):
    #     for i in range(len(transforms)):
    #         self.keyframes[i].transform = transforms[i]
    #
    # def set_relative_transforms(self, relative_transforms):
    #     """
    #     Given a set of relative transforms. Assign to each keyframe a global transform by
    #     postmultiplication.
    #     Caution, computing global transforms from relative transforms starting from T0=I
    #     """
    #     T = HomogeneousMatrix(np.eye(4))
    #     global_transforms = [T]
    #     for i in range(len(relative_transforms)):
    #         T = T*relative_transforms[i]
    #         global_transforms.append(T)
    #
    #     for i in range(len(self.keyframes)):
    #         self.keyframes[i].set_global_transform(global_transforms[i])
    #
    # def set_global_transforms(self, global_transforms):
    #     """
    #     Assign the global transformation for each of the keyframes.
    #     """
    #     for i in range(len(self.keyframes)):
    #         self.keyframes[i].set_global_transform(global_transforms[i])
    #
    # def compute_transformation_local(self, i, j, method='A', initial_transform=np.eye(4)):
    #     """
    #     Compute relative transformation using ICP from keyframe i to keyframe j when j-i = 1.
    #     An initial estimate is used to compute using icp
    #     """
    #     # compute initial transform from odometry
    #     # # TODO: Compute inintial transformation from IMU
    #     # if use_initial_transform:
    #     #     # initial estimation
    #     #     # xi = self.keyframes[i].x
    #     #     # xj = self.keyframes[j].x
    #     #     Ti = self.keyframes[i].transform # HomogeneousMatrix([xi[0], xi[1], 0], Euler([0, 0, xi[2]]))
    #     #     Tj = self.keyframes[j].transform # HomogeneousMatrix([xj[0], xj[1], 0], Euler([0, 0, xj[2]]))
    #     #     Tij = Ti.inv() * Tj
    #     #     # muatb = Tij.t2v()
    #     #     transform = self.keyframes[i].local_registration(self.keyframes[j], initial_transform=Tij.array)
    #     #     atb = HomogeneousMatrix(transform.transformation)
    #     #     return atb
    #     # else:
    #     if method == 'A':
    #         atb = self.keyframes[i].local_registrationA(self.keyframes[j], initial_transform=initial_transform)
    #     else:
    #         atb = self.keyframes[i].local_registrationB(self.keyframes[j], initial_transform=initial_transform)
    #     return atb
    #
    # def compute_transformation_global(self, i, j, method='A'):
    #     """
    #     Compute relative transformation using ICP from keyframe i to keyframe j.
    #     An initial estimate is used.
    #     FPFh to align and refine with icp
    #     """
    #     if method == 'A':
    #         atb = self.keyframes[i].global_registrationA(self.keyframes[j])
    #     elif method == 'B':
    #         atb = self.keyframes[i].global_registrationB(self.keyframes[j])
    #     elif method == 'C':
    #         atb = self.keyframes[i].global_registrationC(self.keyframes[j])
    #     else:
    #         atb = self.keyframes[i].global_registrationD(self.keyframes[j])
    #     return atb
    #
    # def view_map(self, keyframe_sampling=10, point_cloud_sampling=1000):
    #     print("COMPUTING MAP FROM KEYFRAMES")
    #     # transform all keyframes to global coordinates.
    #     pointcloud_global = o3d.geometry.PointCloud()
    #     for i in range(0, len(self.keyframes), keyframe_sampling):
    #         print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
    #         kf = self.keyframes[i]
    #         # transform to global and
    #         pointcloud_temp = kf.transform_to_global(point_cloud_sampling=point_cloud_sampling)
    #         # yuxtaponer los pointclouds
    #         pointcloud_global = pointcloud_global + pointcloud_temp
    #     # draw the whole map
    #     o3d.visualization.draw_geometries([pointcloud_global])
    #
    # def save_to_file(self, filename, keyframe_sampling=10, point_cloud_sampling=100):
    #     print("COMPUTING MAP FROM KEYFRAMES")
    #     # transform all keyframes to global coordinates.
    #     pointcloud_global = o3d.geometry.PointCloud()
    #     for i in range(0, len(self.keyframes), keyframe_sampling):
    #         print("Keyframe: ", i, "out of: ", len(self.keyframes), end='\r')
    #         kf = self.keyframes[i]
    #         # transform to global and
    #         pointcloud_temp = kf.transform_to_global(point_cloud_sampling=point_cloud_sampling)
    #         # yuxtaponer los pointclouds
    #         pointcloud_global = pointcloud_global + pointcloud_temp
    #     print("SAVING MAP AS: ", filename)
    #     o3d.io.write_point_cloud(filename, pointcloud_global)
