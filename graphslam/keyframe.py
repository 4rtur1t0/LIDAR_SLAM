"""
    A Keyframe stores the pointcloud corresponding to a timestamp.
    The class includes methods to register consecutive pointclouds (local registration) and, as well, other pointclouds
    that may be found far away.
https://github.com/hankyang94/teaser_fpfh_threedmatch_python
"""
import numpy as np
from graphslam.scdescriptor import SCDescriptor
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import open3d as o3d
import copy
from config import PARAMETERS


class KeyFrame():
    def __init__(self, directory, scan_time, index):
        # the estimated transform of this keyframe with respect to global coordinates
        self.transform = None
        # max radius to filter points
        self.max_radius = PARAMETERS.max_distance
        # voxel sizes
        self.voxel_downsample_size = PARAMETERS.voxel_size # None
        self.voxel_size_normals = PARAMETERS.radius_normals
        self.voxel_size_normals_ground_plane = PARAMETERS.radius_gd
        # self.voxel_size_fpfh = 3*self.voxel_s
        self.icp_threshold = PARAMETERS.distance_threshold
        self.fpfh_threshold = 2
        # crop point cloud to this bounding box
        # self.dims_bbox = [40, 40, 40]
        self.index = index
        self.timestamp = scan_time
        self.filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'
        # all points
        self.pointcloud = None
        # pcd with a segmented ground plate
        self.pointcloud_filtered = None
        self.pointcloud_ground_plane = None
        self.pointcloud_non_ground_plane = None
        self.max_radius_descriptor = 20
        self.scdescriptor = SCDescriptor(max_radius=self.max_radius_descriptor)
        # save the pointcloud for Scan context description

    def load_pointcloud(self):
        self.pointcloud = o3d.io.read_point_cloud(self.filename)

    def pre_process(self):
        # bbox = self.dims_bbox
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-bbox[0], -bbox[1], -bbox[2]), max_bound=(bbox[0],
        #                                                                                                 bbox[1],
        #                                                                                                 bbox[2]))
        # self.pointcloud = self.pointcloud.crop(bbox)
        # filter by a max radius to avoid erros in normal computation
        self.pointcloud_filtered = self.filter_by_radius(self.pointcloud.points, self.max_radius)

        # self.pointcloud_descriptor = self.filter_by_radius(self.max_radius_descriptor)
        # downsample pointcloud and save to pointcloud in keyframe
        if self.voxel_downsample_size is not None:
            self.pointcloud_filtered = self.pointcloud_filtered.voxel_down_sample(voxel_size=self.voxel_downsample_size)
        # segment ground plane
        pcd_ground_plane, pcd_non_ground_plane = self.segment_plane()
        self.pointcloud_ground_plane = pcd_ground_plane
        self.pointcloud_non_ground_plane = pcd_non_ground_plane
        # calcular las normales a cada punto
        self.pointcloud_filtered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=PARAMETERS.max_nn))
        self.pointcloud_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
                                                                              max_nn=PARAMETERS.max_nn_gd))
        self.pointcloud_non_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=PARAMETERS.max_nn))

    def filter_by_radius(self, points, max_radius):
        points = np.asarray(points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        r2 = x**2 + y**2
        idx = np.where(r2 < max_radius**2)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))

    def filter_by_height(self, height=-0.5):
        points = np.asarray(self.pointcloud.points)
        idx = points[:, 2] < height
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))

    def segment_plane(self, height=-0.5, thresholdA=0.01, thresholdB=0.1):
        """
        filter roughly the points that may belong to the plane.
        then estimate the plane with these points.
        find the distance of the points to the plane and classify
        """
        # find a plane by removing some of the points at a given height
        # this best estimates a ground plane.
        points = np.asarray(self.pointcloud_filtered.points)
        idx = points[:, 2] < height
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(points[idx])
        plane_model, inliers = pcd_plane.segment_plane(distance_threshold=thresholdA, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        points = np.asarray(self.pointcloud_filtered.points)
        inliers_final = []
        for i in range(len(points)):
            dist = np.abs(a*points[i, 0] + b*points[i, 1] + c*points[i, 2] + d)/np.sqrt(a*a+b*b+c*c)
            if dist < thresholdB:
                inliers_final.append(i)

        # now select the final pointclouds
        plane_cloud = self.pointcloud_filtered.select_by_index(inliers_final)
        non_plane_cloud = self.pointcloud_filtered.select_by_index(inliers_final, invert=True)
        return plane_cloud, non_plane_cloud

    def local_registrationA(self, other, initial_transform):
        """
        Use icp to compute transformation using an initial estimate.
        Method A uses all points and a PointToPlane estimation.
        caution, initial_transform is a np array.
        """
        debug = False
        print("Apply point-to-plane ICP")
        print("Using threshold: ", self.icp_threshold)
        # sigma = 0.1
        # loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        # print("Using robust loss:", loss)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            other.pointcloud_filtered, self.pointcloud_filtered, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        # print("Transformation is:")
        # print(reg_p2p.transformation)
        if debug:
            other.draw_registration_result(self, reg_p2p.transformation)
        return HomogeneousMatrix(reg_p2p.transformation), reg_p2p.inlier_rmse

    def local_registrationB(self, other, initial_transform):
        """
        Use icp to compute transformation using an initial estimate.
        Method B segments a ground plane and estimates two different transforms:
            - A: using ground planes tz, alfa and beta are estimated. Point to
            - B: using non ground planes (rest of the points) tx, ty and gamma are estimated
        caution, initial_transform is a np array.
        """
        debug = False
        # if debug:
        #     other.draw_registration_result(self, initial_transform)

        # compute a first transform for tz, alfa, gamma, using ground planes
        reg_p2pa = o3d.pipelines.registration.registration_icp(
            other.pointcloud_ground_plane, self.pointcloud_ground_plane, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2pa)
        if debug:
            other.draw_registration_result(self, reg_p2pa.transformation)
        # compute second transformation using the whole pointclouds. CAUTION: failures in ground plane segmentation
        # do affect this transform if computed with some parts of ground
        reg_p2pb = o3d.pipelines.registration.registration_icp(
            other.pointcloud_filtered, self.pointcloud_filtered, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2pb)

        # if debug:
        #     other.draw_registration_result(self, reg_p2pb.transformation)

        t1 = HomogeneousMatrix(reg_p2pa.transformation).t2v(n=3)
        t2 = HomogeneousMatrix(reg_p2pb.transformation).t2v(n=3)
        # build solution using both solutions
        tx = t2[0]
        ty = t2[1]
        tz = t1[2]
        alpha = t1[3]
        beta = t1[4]
        gamma = t2[5]
        T = HomogeneousMatrix(np.array([tx, ty, tz]), Euler([alpha, beta, gamma]))
        if debug:
            other.draw_registration_result(self, T.array)
        return T, reg_p2pb.inlier_rmse


    def global_registrationD(self, other):
        """
        Method based on Scan Context plus correlation.
        Two scan context descriptors are found.
        """
        print('Computing global registration using Scan Context')
        debug = False
        # if debug:
        #     other.draw_registration_result(self, np.eye(4))

        # sample down points
        # using points that do not belong to ground
        voxel_down_sample = 0.2

        # pcd1 = self.pointcloud_descriptor.voxel_down_sample(voxel_size=voxel_down_sample)
        # pcd2 = other.pointcloud_descriptor.voxel_down_sample(voxel_size=voxel_down_sample)

        pcd1 = self.filter_by_radius(self.pointcloud_non_ground_plane.points, max_radius=self.scdescriptor.max_radius)
        pcd2 = self.filter_by_radius(other.pointcloud_non_ground_plane.points, max_radius=self.scdescriptor.max_radius)

        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_down_sample)
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_down_sample)

        # if debug:
        #     o3d.visualization.draw_geometries([pcd1, pcd2])

        sc1, r1, c1, z1 = self.scdescriptor.compute_descriptor(pcd1.points)
        sc2, r2, c2, z2 = other.scdescriptor.compute_descriptor(pcd2.points)

        # if debug:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter(r1, c1, z1)
        #     ax.scatter(r2, c2, z2)
        #     ax.set_xlabel('r')
        #     ax.set_ylabel('c')
        #     ax.set_zlabel('Z Label')
        #     plt.show()

        # gamma1, prob = self.scdescriptor.maximize_correlation(other.scdescriptor)
        gamma2, prob = self.scdescriptor.maximize_correlation2(other.scdescriptor)
        # gamma = 1.5
        # assuming a rough SE2 transformation here
        T = HomogeneousMatrix(np.array([0, 0, 0]), Euler([0, 0, gamma2]))

        if debug:
            other.draw_registration_result(self, T.array)
        return T, prob

    def draw_registration_result(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud)
        target_temp = copy.deepcopy(other.pointcloud)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def draw_pointcloud(self):
        o3d.visualization.draw_geometries([self.pointcloud])

    def numpy2pointcloud(self, xyz):
        self.pointcloud.points = o3d.utility.Vector3dVector(xyz)

    def set_global_transform(self, transform):
        self.transform = transform
        return

    def transform_to_global(self, point_cloud_sampling=10):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        if self.transform is None:
            return None
        T = HomogeneousMatrix(self.transform)
        pointcloud = self.pointcloud.uniform_down_sample(every_k_points=point_cloud_sampling)
        return pointcloud.transform(T.array)

    def transform_by_T(self, T):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        return self.pointcloud.transform(T.array)










