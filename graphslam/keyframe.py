"""
    A Keyframe stores the pointcloud corresponding to a timestamp.
    The class includes methods to register consecutive pointclouds (local registration) and, as well, other pointclouds
    that may be found far away.
https://github.com/hankyang94/teaser_fpfh_threedmatch_python
"""
import numpy as np
# import subprocess
from graphslam.scdescriptor import SCDescriptor, maximize_correlation
from registration.registration_tools import best_fit_transform, compute_occupancy_grid
from tools.conversions import cartesian_to_spherical, spherical_to_cartesian
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import cv2


class KeyFrame():
    def __init__(self, directory, scan_time, index):
        # the estimated transform of this keyframe with respect to global coordinates
        self.transform = None
        # voxel sizes
        self.voxel_downsample_size = None # 0.0001
        self.voxel_size_normals = 0.3
        self.voxel_size_normals_ground_plane = 0.6
        # self.voxel_size_fpfh = 3*self.voxel_s
        self.icp_threshold = 3
        self.fpfh_threshold = 2
        # crop point cloud to this bounding box
        self.dims_bbox = [40, 40, 40]
        self.index = index
        self.timestamp = scan_time
        self.filename = directory + '/robot0/lidar/data/' + str(scan_time) + '.pcd'
        self.pointcloud = None
        # pcd with a segmented ground plate
        self.pointcloud_ground_plane = None
        self.pointcloud_non_ground_plane = None

        self.scdescriptor = SCDescriptor(maxR=15.0)

    def load_pointcloud(self):
        self.pointcloud = o3d.io.read_point_cloud(self.filename)

    def pre_process(self):
        bbox = self.dims_bbox
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-bbox[0], -bbox[1], -bbox[2]), max_bound=(bbox[0],
                                                                                                        bbox[1],
                                                                                                        bbox[2]))
        self.pointcloud = self.pointcloud.crop(bbox)
        # self.draw_pointcloud()
        # downsample pointcloud and save to pointcloud in keyframe
        if self.voxel_downsample_size is not None:
            self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=self.voxel_downsample_size)
        # segment ground plane
        pcd_ground_plane, pcd_non_ground_plane = self.segment_plane()
        self.pointcloud_ground_plane = pcd_ground_plane
        self.pointcloud_non_ground_plane = pcd_non_ground_plane
        # calcular las normales a cada punto
        self.pointcloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=30))
        self.pointcloud_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals_ground_plane,
                                                                              max_nn=30))
        self.pointcloud_non_ground_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                                              max_nn=30))




    def segment_plane(self, height=-0.5, thresholdA=0.01, thresholdB=0.1):
        """
        filter roughly the points that may belong to the plane.
        then estimate the plane with these points.
        find the distance of the points to the plane and classify
        """
        # find a plane by removing some of the points at a given height
        # this best estimates a ground plane.
        points = np.asarray(self.pointcloud.points)
        idx = points[:, 2] < height
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(points[idx])
        plane_model, inliers = pcd_plane.segment_plane(distance_threshold=thresholdA, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        points = np.asarray(self.pointcloud.points)
        inliers_final = []
        for i in range(len(points)):
            dist = np.abs(a*points[i, 0] + b*points[i, 1] + c*points[i, 2] + d)/np.sqrt(a*a+b*b+c*c)
            if dist < thresholdB:
                inliers_final.append(i)

        # now select the final pointclouds
        plane_cloud = self.pointcloud.select_by_index(inliers_final)
        non_plane_cloud = self.pointcloud.select_by_index(inliers_final, invert=True)
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
            other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform,
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
        debug = True
        # if debug:
        #     other.draw_registration_result(self, initial_transform)

        # compute a first transform for tz, alfa, gamma, using ground planes
        reg_p2pa = o3d.pipelines.registration.registration_icp(
            other.pointcloud_ground_plane, self.pointcloud_ground_plane, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2pa)
        # if debug:
        #     other.draw_registration_result(self, reg_p2pa.transformation)
        # compute second transformation using the whole pointclouds. CAUTION: failures in ground plane segmentation
        # do affect this transform if computed with some parts of ground
        reg_p2pb = o3d.pipelines.registration.registration_icp(
            other.pointcloud, self.pointcloud, self.icp_threshold, initial_transform,
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


    def global_registrationA(self, other):
        """
        Method based on finding correspondences on depth images.
        Once correspondences are found, a least squares method is used to compute R and t.
        In low distintive environments does not work well.
        """
        debug = True
        image1 = self.project_pointcloud()
        image2 = other.project_pointcloud()
        plt.figure()
        plt.imshow(image1, cmap=plt.get_cmap('gray'))
        plt.figure()
        plt.imshow(image2, cmap=plt.get_cmap('gray'))
        rcd1, rcd2 = self.compute_correspondences_orb3D(image1, image2)
        pts1 = self.unproject_pointcloud(rcd1)
        pts2 = self.unproject_pointcloud(rcd2)

        # cloud1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts1))
        # cloud2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts2))

        # o3d.visualization.draw_geometries([cloud1])
        # o3d.visualization.draw_geometries([cloud2])
        T = best_fit_transform(pts1, pts2)

        if debug:
            other.draw_registration_result(self, T.array)
        return T

    def compute_correspondences_orb3D(self, image1, image2):
        view_images = True
        # convert to cv uint8 and scale up
        imagecv1 = 20*image1.astype(np.uint8)
        imagecv2 = 20*image2.astype(np.uint8)
        # cv2.imshow("imagecv1", imagecv1)
        # cv2.waitKey(0)
        #
        # cv2.imshow("imagecv2", imagecv2)
        # cv2.waitKey(0)
        orb = cv2.ORB_create(nfeatures=100)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp1, des1 = orb.detectAndCompute(imagecv1, None)
        kp2, des2 = orb.detectAndCompute(imagecv2, None)

        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(imagecv1, kp1, imagecv2, kp2, matches, None)
        cv2.imshow("Output1-low-response", img3)
        cv2.waitKey(0)
        # return a list of r, c, d values
        # For each match...
        list_kp1 = []
        list_kp2 = []
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            # x - columns         # y - rows
            # Get the coordinates
            (c1, r1) = kp1[img1_idx].pt
            (c2, r2) = kp2[img2_idx].pt
            # CAUTION: recovering depth from original float images
            d1 = image1[int(r1), int(c1)]
            d2 = image2[int(r2), int(c2)]
            # Append to each list
            list_kp1.append([r1, c1, d1])
            list_kp2.append([r2, c2, d2])
            # list_kp1.append([r1, c1, 0])
            # list_kp2.append([r2, c2, 0])
        return np.array(list_kp1), np.array(list_kp2)

    def project_pointcloud(self):
        """
        Computes a depth image from the pointcloud
        """
        nr = 500
        nc = 500
        fov_phi = 2*np.pi
        fov_th = np.pi
        kr = nr/fov_th
        kc = nc/fov_phi
        # kdepth = 255.0/50.0
        image = np.zeros((nr, nc))
        points = np.asarray(self.pointcloud.points)
        # project to image using spherical coordinates
        for i in range(len(points)):
            point_spher = cartesian_to_spherical(points[i, :])
            [phi, th, depth] = point_spher
            phi = phi + fov_phi/2
            r = int(np.round(kr*th))
            c = int(np.round(kc*phi))
            # depth = np.sqrt(x*x + y*y + z*z)
            if r >= 0 and r < nr and c >= 0 and c < nc:
                image[r, c] = depth
        return image

    def unproject_pointcloud(self, rcd_points):
        nr = 500
        nc = 500
        fov_phi = 2 * np.pi
        fov_th = np.pi
        kr = nr / fov_th
        kc = nc / fov_phi
        points = []
        # unproject points from image to xyz asuming spherical coordinates
        for rcd in rcd_points:
            [r, c, depth] = rcd
            phi = c/kc-fov_phi/2
            th = r/kr
            point = spherical_to_cartesian([phi, th, depth])
            points.append(point)
        return np.array(points)

    def unproject_image(self, image):
        shape = image.shape
        nr = shape[0]
        nc = shape[1]
        fov_phi = 2 * np.pi
        fov_th = np.pi
        kr = nr / fov_th
        kc = nc / fov_phi
        points = []
        # unproject points from image to xyz asuming spherical coordinates
        for r in range(nr):
            for c in range(nc):
                depth = image[r, c]
                if depth == 0:
                    continue
                phi = c/kc-fov_phi/2
                th = r/kr
                point = spherical_to_cartesian([phi, th, depth])
                points.append(point)
        return np.array(points)


    def global_registrationB(self, other):
        """
        Tries to find correspondences using fpfp features.
        Similar to registration A, does not work well in an environment with low distintive features.
        """
        debug = True
        radius_feature = 0.3

        if debug:
            other.draw_registration_result(self, np.eye(4))

        # extraer los Fast Point Feature Histograms
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.pointcloud_non_ground_plane,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                       radius=radius_feature,
                                                                       max_nn=100))
        otherpcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(other.pointcloud_non_ground_plane,
                                                                        o3d.geometry.KDTreeSearchParamHybrid(
                                                                        radius=radius_feature,
                                                                        max_nn=100))

        # self.view_correspondences(pcd, pcd_fpfh, otherpcd, otherpcd_fpfh)
        #     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        max_correspondence_distance = 10
        global_transform = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            other.pointcloud_non_ground_plane, self.pointcloud_non_ground_plane, otherpcd_fpfh, pcd_fpfh,
            mutual_filter=True,
            max_correspondence_distance=max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            ransac_n=10,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.99))
        if debug:
            other.draw_registration_result(self, global_transform.transformation)
        return HomogeneousMatrix(global_transform.transformation)

    # def view_correspondences(self, pcd, pcd_fpfh, otherpcd, otherpcd_fpfh):
    #     correspondences = []
    #     feature_dists = []
    #     fpfh_tree = o3d.geometry.KDTreeFlann(pcd_fpfh)
    #     for i in range(len(otherpcd.points)):
    #         [_, idx, _] = fpfh_tree.search_knn_vector_xd(otherpcd_fpfh.data[:, i], 1)
    #         feature_dist = np.linalg.norm(otherpcd_fpfh.data[:, i]-pcd_fpfh.data[:, idx])
    #         feature_dists.append(feature_dist)
    #         dis = np.linalg.norm(otherpcd.points[i] - pcd.points[idx[0]])
    #         c = (0.2 - np.fmin(dis, 0.2)) / 0.2
    #         pcd.colors[i] = [c, c, c]
    #         correspondences.append([i, idx[0]])
    #     print(correspondences)
    #     print(np.mean(feature_dists))



    def global_registrationC(self, other):
        """
        Method based on finding correspondences on an occupancy grid.
        After projecting the clouds to a 2D occupancy grid, ORB correspondences are found.
        did not show promising results.
        """
        debug = True
        image1 = compute_occupancy_grid(points=np.asarray(self.pointcloud_non_ground_plane.points))
        image2 = compute_occupancy_grid(points=np.asarray(other.pointcloud_non_ground_plane.points))
        plt.figure()
        plt.imshow(image1, cmap=plt.get_cmap('gray'))
        plt.figure()
        plt.imshow(image2, cmap=plt.get_cmap('gray'))
        rcd1, rcd2 = self.compute_correspondences_orb2D(image1, image2)

        nr = 1500
        max_m = 20
        # max_z = 20 # max height
        km = nr / (2 * max_m)
        rcd1 = rcd1/km + max_m
        rcd2 = rcd2/km + max_m

        # pts1 = self.unproject_pointcloud(rcd1)
        # pts2 = self.unproject_pointcloud(rcd2)

        # cloud1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts1))
        # cloud2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts2))

        # o3d.visualization.draw_geometries([cloud1])
        # o3d.visualization.draw_geometries([cloud2])
        T = best_fit_transform(rcd1, rcd2)

        if debug:
            other.draw_registration_result(self, T.array)
        return T

    def compute_correspondences_orb2D(self, image1, image2):
        view_images = True
        # convert to cv uint8 and scale up
        imagecv1 = 20*image1.astype(np.uint8)
        imagecv2 = 20*image2.astype(np.uint8)
        # cv2.imshow("imagecv1", imagecv1)
        # cv2.waitKey(0)
        #
        # cv2.imshow("imagecv2", imagecv2)
        # cv2.waitKey(0)
        orb = cv2.ORB_create(nfeatures=100)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp1, des1 = orb.detectAndCompute(imagecv1, None)
        kp2, des2 = orb.detectAndCompute(imagecv2, None)

        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(imagecv1, kp1, imagecv2, kp2, matches, None)
        cv2.imshow("Output1-low-response", img3)
        cv2.waitKey(0)
        # return a list of r, c, d values
        # For each match...
        list_kp1 = []
        list_kp2 = []
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            # x - columns         # y - rows
            # Get the coordinates
            (c1, r1) = kp1[img1_idx].pt
            (c2, r2) = kp2[img2_idx].pt
            # CAUTION: recovering depth from original float images
            # d1 = image1[int(r1), int(c1)]
            # d2 = image2[int(r2), int(c2)]
            # Append to each list
            # list_kp1.append([r1, c1, d1])
            # list_kp2.append([r2, c2, d2])
            list_kp1.append([r1, c1, 0])
            list_kp2.append([r2, c2, 0])
        return np.array(list_kp1), np.array(list_kp2)


    def global_registrationD(self, other):
        """
        Method based on Scan Context plus correlation.
        Two scan context descriptors are found.
        """
        debug = True
        if debug:
            other.draw_registration_result(self, np.eye(4))

        sc1 = self.scdescriptor.compute_descriptor(self.pointcloud.points)
        sc2 = other.scdescriptor.compute_descriptor(other.pointcloud.points)

        # plt.figure(0)
        # plt.imshow(sc1)
        # plt.figure(1)
        # plt.imshow(sc2)

        dist, gamma = self.scdescriptor.maximize_correlation(other.scdescriptor)
        # assuming a rough SE2 transformation here
        T = HomogeneousMatrix(np.array([0, 0, 0]), Euler([0, 0, gamma]))

        if debug:
            other.draw_registration_result(self, T.array)
        return T




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
        T = HomogeneousMatrix(self.transform)
        pointcloud = self.pointcloud.uniform_down_sample(every_k_points=point_cloud_sampling)
        return pointcloud.transform(T.array)

    def transform_by_T(self, T):
        """
            Use open3d to fast transform to global coordinates.
            Returns the pointcloud in global coordinates
        """
        return self.pointcloud.transform(T.array)










