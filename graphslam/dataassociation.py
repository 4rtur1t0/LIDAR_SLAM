import numpy as np
import matplotlib.pyplot as plt
from tools.conversions import mod_2pi
import gtsam
import gtsam.utils.plot as gtsam_plot

from tools.outlier_rejection import outlier_rejection


class DataAssociation():
    def __init__(self, graphslam,  icp_noise, delta_index=7, xi2_th=40.26, d_th=6):
        """
        xi2_th=16.26
        """
        self.graphslam = graphslam
        # look for data associations that are delta_index back in time
        self.delta_index = delta_index
        self.xi2_threshold = xi2_th
        self.euclidean_distance_threshold = d_th
        self.icp_noise = icp_noise

    def find_initial_candidates(self):
        """
        The function finds an initial set of candidates based on euclidean distance
        """
        distances = []
        candidates = []
        i = self.graphslam.current_index-1
        for j in range(i-self.delta_index):
            d = self.euclidean_distance(i, j)
            distances.append(d)
            if d < self.euclidean_distance_threshold:
                candidates.append([i, j])
        return candidates

    def filter_data_associations(self, dijskstra, candidates, transformations):
        """
        The function completes the whole data association process using:
            - Finding an initial set of candidate hypotheses with kmax = 8 and kmin=5
            - Next computing a set of pairwise probabilities.
                in order to do so, taking the hypotheses in pairs, we try to form a loop. A probability factor Aij is computed
                based.
            - Next, a set of the hypotheses is selected

        Conditions:
            - The number of candidates must b > 8 in order to be considered for consistency.
            - The pairwise consistency matrix must have a dominant eigenvalue, indicating that the data can be explained
              in a single way (lambda1/lambda2 > 2).

        """
        if len(candidates) < 8:
            return []
        Sab, Tab = dijskstra.compute_dijkstra_path(candidates, transformations)
        A = self.compute_pairwise_consistency_matrix(candidates, transformations)
        # given the pairwise consistency matrix, find the set of hypotheses that best support the associations
        confidence, v = outlier_rejection(A)
        # in this case, the solution is not robust and we should wait to find more candidates.
        if not confidence:
            return []
        filtered_candidates = []
        for i in range(len(v)):
            if v[i]:
                filtered_candidates.append(candidates[i])
        return filtered_candidates



    def compute_pairwise_consistency_matrix(self, candidates):
        """
        Selects association hypothesis in pairs.
        For each pair, compute a loop closing transformation and an associated probability
        """
        N = len(candidates)
        A = np.zeros(N, N)
        # find all combinations of i and j of the hypotheses
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                # compute the rigid transformation that connects the candidates.
                t, S = self.rigid_transformation(candidates[i], candidates[j])
                A[i][j] = np.exp(np.dot(t, np.dot(S, t.T)))
        return A


    def simulate_observations_SE2(x_gt, observations):
        """
        x_gt: ground truth poses
        A series of relative observations are generated from the ground truth solution x_gt
        """
        N = len(observations)
        edges = []
        sx = self.icp_noise[0][0]
        sy = self.icp_noise[1][1]
        sth = self.icp_noise[2][2]
        for k in range(N):
            i = observations[k][0]
            j = observations[k][1]
            ti = np.hstack((x_gt[i, 0:2], 0))
            tj = np.hstack((x_gt[j, 0:2], 0))
            Ti = HomogeneousMatrix(ti, Euler([0, 0, x_gt[i, 2]]))
            Tj = HomogeneousMatrix(tj, Euler([0, 0, x_gt[j, 2]]))
            Tiinv = Ti.inv()
            zij = Tiinv * Tj
            zij = zij.t2v()
            # add noise to the observatoins
            zij = zij + np.array([np.random.normal(0, sx, 1)[0],
                                  np.random.normal(0, sy, 1)[0],
                                  np.random.normal(0, sth, 1)[0]])
            # np.random.normal([0, 0, 0], [sx, sy, sth], 1)
            zij[2] = mod_2pi(zij[2])
            Tij = HomogeneousMatrix([zij[0], zij[1], 0], Euler([0, 0, zij[2]]))
            edges.append(Tij)
        return edges


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


