import numpy as np
import matplotlib.pyplot as plt
from tools.conversions import mod_2pi
import gtsam
import gtsam.utils.plot as gtsam_plot


class DataAssociation():
    def __init__(self, graphslam, delta_index=7, xi2_th=40.26, d_th=6):
        """
        xi2_th=16.26
        """
        self.graphslam = graphslam
        # look for data associations that are delta_index back in time
        self.delta_index = delta_index
        self.xi2_threshold = xi2_th
        self.euclidean_distance_threshold = d_th

    def perform_data_association(self):
        """
        The function i
        """
        distances = []
        candidates = []
        i = self.graphslam.current_index-1
        # start checking delta_index indexes before i
        # check from i-delta_index up to the first node
        # caution, must reach index 0, first pose in graph
        # Computing Mahalanobis distance
        # for j in range(i-self.delta_index):
        #     # caution, using only the x, y pose
        #     d2 = self.joint_mahalanobis(j, i, only_position=True)
        #     distances.append(d2)
        #     # print('Mahalanobis distance between: ', i, j, d2)
        #     if d2 < self.xi2_threshold:
        #         candidates.append([i, j])
        for j in range(i-self.delta_index):
            d = self.euclidean_distance(i, j)
            distances.append(d)
            if d < self.euclidean_distance_threshold:
                candidates.append([i, j])
        return candidates

    def marginal_covariance(self, i):
        # init initial estimate, read from self.current_solution
        initial_estimate = gtsam.Values()
        k = 0
        for pose2 in self.graphslam.current_solution:
            initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
            k = k+1
        marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
        cov = marginals.marginalCovariance(i)
        return cov

    def joint_marginal_covariance(self, i, j):
        # init initial estimate, read from self.current_solution
        initial_estimate = gtsam.Values()
        k = 0
        for pose2 in self.graphslam.current_solution:
            initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
            k = k+1
        marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
        keyvector = gtsam.utilities.createKeyVector([i, j])
        jm = marginals.jointMarginalCovariance(variables=keyvector).at(iVariable=i, jVariable=j)
        return jm

    def marginal_information(self, i):
        # init initial estimate, read from self.current_solution
        initial_estimate = gtsam.Values()
        k = 0
        for pose2 in self.graphslam.current_solution:
            initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
            k = k+1
        marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
        cov = marginals.marginalInformation(i)
        return cov

    def joint_marginal_information(self, i, j):
        # init initial estimate, read from self.current_solution
        initial_estimate = gtsam.Values()
        k = 0
        for pose2 in self.graphslam.current_solution:
            initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
            k = k+1
        marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
        keyvector = gtsam.utilities.createKeyVector([i, j])
        jm = marginals.jointMarginalInformation(variables=keyvector).at(iVariable=i, jVariable=j)
        return jm

    def joint_mahalanobis(self, i, j, only_position=False):
        """
        Using an approximation for the joint conditional probability of node i and j
        """
        Oii = self.marginal_information(i)
        Ojj = self.marginal_information(j)
        Inf_joint = Oii + Ojj
        muii = self.graphslam.current_estimate[i]
        mujj = self.graphslam.current_estimate[j]
        mu = mujj-muii
        mu[2] = mod_2pi(mu[2])
        # do not consider orientation
        if only_position:
            mu[2] = 0.0
        d2 = np.abs(np.dot(mu.T, np.dot(Inf_joint, mu)))
        return d2

    def euclidean_distance(self, i, j):
        """
        Compute Euclidean distance between nodes i and j in the solution
        """
        matrixii = self.graphslam.current_estimate.atPose3(i).matrix()
        matrixjj = self.graphslam.current_estimate.atPose3(j).matrix()
        muii = matrixii[:3, 3]
        mujj = matrixjj[:3, 3]
        dist = np.linalg.norm(mujj-muii)
        return dist

    def test_conditional_probabilities(self):
        """
        """
        muii = self.graphslam.current_estimate[13]
        Sii = self.marginal_covariance(13)

        Sjj = self.marginal_covariance(1)
        mujj = self.graphslam.current_estimate[1]

        Sij = self.joint_marginal_covariance(1, 13)
        Sji = self.joint_marginal_covariance(13, 1)

        Sii_ = Sii - np.dot(Sij, np.dot(np.linalg.inv(Sjj), Sij.T))
        Sjj_ = Sjj - np.dot(Sij.T, np.dot(np.linalg.inv(Sii), Sij))


        # product, joint probability
        Sca = np.linalg.inv(np.linalg.inv(Sii) + np.linalg.inv(Sjj))
        Scb = Sii + Sjj
        a1 = np.dot(np.linalg.inv(Sii), muii)
        a2 = np.dot(np.linalg.inv(Sjj), mujj)
        mc = np.dot(Sca, a1+a2)

        # gtsam_plot.plot_pose2(0, gtsam.Pose2(muii[0], muii[1], muii[2]), 0.5, Sii)
        # gtsam_plot.plot_pose2(0, gtsam.Pose2(mujj[0], mujj[1], mujj[2]), 0.5, Sjj)
        # gtsam_plot.plot_pose2(0, gtsam.Pose2(0, 0, 0), 0.5, Sii_)
        # gtsam_plot.plot_pose2(0, gtsam.Pose2(0, 1.5, 0), 0.5, Sjj_)
        # gtsam_plot.plot_pose2(0, gtsam.Pose2(mc[0], mc[1], mc[2]), 0.5, Sca)

        for i in range(10):
            mu = 0.5*(mujj + muii)
            mu[2] = 0
            muij = mujj - muii
            muij[2] = 0

            gtsam_plot.plot_pose2(0, gtsam.Pose2(muii[0], muii[1], muii[2]), 0.5, Sii)
            gtsam_plot.plot_pose2(0, gtsam.Pose2(mujj[0], mujj[1], mujj[2]), 0.5, Sjj)
            gtsam_plot.plot_pose2(0, gtsam.Pose2(mu[0], mu[1], mu[2]), 0.5, Sca)
            gtsam_plot.plot_pose2(0, gtsam.Pose2(mu[0], mu[1], mu[2]), 0.5, Scb)

            d0 = np.dot(muij.T, np.dot(np.linalg.inv(Sca), muij))
            d1 = np.dot(muij.T, np.dot(np.linalg.inv(Scb), muij))
            # d2 = np.dot(muij.T, np.dot(np.linalg.inv(Sii_), muij))
            # d3 = np.dot(muij.T, np.dot(np.linalg.inv(Sjj_), muij))

            muii += np.array([0.2, 0, 0])
        return True

    def view_full_information_matrix(self):
        """
        The function i
        """
        n = self.graphslam.current_index + 1
        H = np.zeros((3*n, 3*n))

        for i in range(n):
            Hii = self.marginal_information(i)
            print(i, i)
            print(Hii)
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Hii

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Hij = self.joint_marginal_information(i, j)
                print(i, j)
                print(Hij)
                H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = Hij

        plt.figure()
        plt.matshow(H)
        plt.show()
        return True

    def view_full_covariance_matrix(self):
        """
        """
        n = self.graphslam.current_index + 1
        H = np.zeros((3*n, 3*n))

        for i in range(n):
            Hii = self.marginal_covariance(i)
            print(i, i)
            print(Hii)
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Hii

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Hij = self.joint_marginal_covariance(i, j)
                print(i, j)
                print(Hij)
                H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = Hij

        plt.figure()
        plt.matshow(H)
        plt.show()
        return True


