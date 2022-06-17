import numpy as np
from tools.dijkstra_projection import DijkstraProjectionRelative
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix


class DataAssociation():
    def __init__(self, graphslam,  icp_noise, delta_index=7, xi2_th=3, d_euclidean_th=8):
        """
        xi2_th=16.26
        """
        self.graphslam = graphslam
        # look for data associations that are delta_index back in time
        self.delta_index = delta_index
        self.xi2_threshold = xi2_th
        self.euclidean_distance_threshold = d_euclidean_th
        self.icp_noise = icp_noise
        self.dijkstra_algorithm = DijkstraProjectionRelative(icp_noise)

    def find_initial_candidates(self):
        """
        The function finds an initial set of candidates based on euclidean distance
        """
        candidates = []
        i = self.graphslam.current_index-1
        for j in range(i-self.delta_index):
            d = self.euclidean_distance(i, j)
            # distances.append(d)
            if d < self.euclidean_distance_threshold:
                candidates.append([i, j])
        return candidates

    def filter_data_associations(self, candidates, transformations, min_number_of_candidates=5):
        """
        The function completes the whole data association process using:
            - Finding an initial set of candidate hypotheses with kmax = 8 and kmin=5
            - Next computing a set of pairwise probabilities.
                in order to do so, taking the hypotheses in pairs, we try to form a loop. A probability factor Aij is computed
                based.
            - Next, a set of the hypotheses is selected

        Conditions:
            - The number of candidates must b > min_number in order to be considered for consistency. i. e. min number
                of candidates
            - The pairwise consistency matrix is analyzed. In particular, the matrix is formed by the Mahalanobis distance
            i-j elements. Each element is filtered
        """
        # considering only if the number of candidates is above a threshold
        if len(candidates) < min_number_of_candidates:
            return [], []
        # TODO: FILTER USING DESCRIPTOR (i. e. Scan Context)
        filtered_candidates, filtered_transformations = self.filter_pairwise_candidates(candidates, transformations)
        return filtered_candidates, filtered_transformations

    def filter_pairwise_candidates(self, candidates, atbs):
        """
        Candidates are filtered in pairs.
        For each one, a loop is computed. In this loop, the uncertainty is computed, (at two different paths) using
        the Dijkstra Projection.
        A Mahalanobis distance is then computed for the variable s in the loop.
        Please, take note that only a SE2 transformation is considered here.
        """
        # Compute the compatibility matrix of the pairs in this method
        [A, D] = self.compute_pairwise_consistency_matrix(candidates, atbs)

        # # # print values
        # for i in range(0, len(candidates)):
        #     for j in range(i + 1, len(candidates)):
        #         if i == j:
        #             continue
        #         print('Candidate: ', i, 'with candidate: ', j)
        #         print(D[i][j])

        # now select the valid hypotheses based on the joint Mahalanobis distances
        # wrong measurements will not add likelihood in terms of i or j
        likelihood_candidates = np.zeros(len(candidates))
        for i in range(0, len(candidates)):
            for j in range(0, len(candidates)):
                if i == j:
                    continue
                # whenever the distance is consistent within a xi2 Mahalanobis distance
                # we increase the log likelihood of i and j by 1
                if D[i][j] < self.xi2_threshold:
                    likelihood_candidates[i] += 1
                    likelihood_candidates[j] += 1
        # filter good candidates
        print('Likelihood candidates: ', likelihood_candidates)
        idx = np.nonzero(likelihood_candidates)[0]
        candidates = np.array(candidates)
        atbs = np.array(atbs)
        filtered_candidates = candidates[idx]
        filtered_transformations = atbs[idx]
        return filtered_candidates, filtered_transformations

    def compute_pairwise_consistency_matrix(self, candidates, atbs):
        """
        Given a set of candidate hypotheses with N > 1. Each data association is accompanied by a computed transformation
        i.e. using ICP.
        Arrange the correspondences in pairs.
        Compute a loop closing transformation for each pair.
        Compute a Covariance matrix S for the transformation based on Sab (Dijkstra Projection) and a linear error
        propagation formula.
        Compute A mahalanobis distance s.T*S*s
        """
        N = len(candidates)
        A = np.zeros((N, N))
        D = np.zeros((N, N))
        for i in range(0, N):
            for j in range(i + 1, N):
                if i == j:
                    continue
                # select pairs of candidates and transforms. Then compute consistency for each pair
                pair_candidates = [candidates[i], candidates[j]]
                pair_transforms = [atbs[i], atbs[j]]
                # computes pairwise consistency between two candidates!
                [Aij, Dij] = self.compute_pairwise_consistency(pair_candidates, pair_transforms)
                A[i][j] = Aij
                A[j][i] = Aij
                D[i][j] = Dij
                D[j][i] = Dij
        return A, D

    def compute_pairwise_consistency(self, candidates, atbs):
        """
        This method computes the pairwise consistency value of two candidates given two transformations.
        Consider, for example, that the associations are:
        [[11, 1],
        [13,3]]
        A transformation loop is formed in this way, with {}^iT_j :T (i, j) (from i to j)
        T_total = T(11, 1)T(1, 3)T(3, 13)T(13, 11)

        Please, note that, in this case, T(3, 13) = T(13,3)^{-1}
        The relative transformation and relative uncertainty of the nodes 1, 3 and 13,11 is computed using the Dijkstra
        Projection.

        This total transformation equals to T(11, 11) and should be equal to the Identity.
        The relative covariance of this transformation is propagated and computed as
        Sab = J1^T*Sa*J1 +

        """
        # the first measurement corresponds to the first correspondence hypothesis
        Tm1 = atbs[0]
        # the second measurement corresponds to the second correspondence hypothesis
        # it is inverted, given the order in which the transformations are performed
        Tm2 = atbs[1].inv()
        # set the absolute poses for each node in the dijkstra algorithm
        self.dijkstra_algorithm.set_poses(self.graphslam.current_estimate)
        source = candidates[0][1]
        finish = candidates[1][1]
        # select source and finish nodes, obtain, the path, relative transformation and relative uncertainty using
        # the Dijkstra projection
        shortest_path1, Tab1, Sab1 = self.dijkstra_algorithm.compute_shortest_path(source=source, finish=finish)
        # again, select source and finish nodes, obtain, the path, relative transformation and relative uncertainty
        # using a Dijkstra projection
        source = candidates[1][0]
        finish = candidates[0][0]
        shortest_path2, Tab2, Sab2 = self.dijkstra_algorithm.compute_shortest_path(source=source, finish=finish)

        # print('Shortest path found 1: ', shortest_path1)
        # print('Shortest path found 2: ', shortest_path2)
        # print('Sigma AB1: ', Sab1)
        # print('Sigma AB2: ', Sab2)
        # print('Transformation AB: ', Tab1)
        # print('Transformation AB: ', Tab2)
        # compute the loop closing tranformation, that includes: a measured transformation corresponding with the
        # data association. A transformation based on Dijkstra path. Another measured data association
        I = Tm1 * Tab1 * Tm2 * Tab2
        s = I.t2v()
        # print('Loop closing constraint: I:', I)
        print('Loop closing constraint: s:', s)
        t0 = np.array([0, 0, 0])
        t1 = Tm1.t2v(n=3)
        t2 = (Tm1 * Tab1).t2v()
        t3 = (Tm1 * Tab1 * Tm2).t2v()
        t4 = (Tm1 * Tab1 * Tm2 * Tab2).t2v()
        # icp noise matrix for measurements
        Sij = self.icp_noise # np.diag([icp_xyz_sigma, icp_xyz_sigma, icp_rpy_sigma])
        # propagate uncertainty, Measured, Dijkstra, Measured, Dijstra
        S0 = self.propagate_uncertainty(ti=t0, tj=t1, Si=np.diag([0, 0, 0]), Sij=Sij)
        S1 = self.propagate_uncertainty(ti=t1, tj=t2, Si=S0, Sij=Sab1)
        S2 = self.propagate_uncertainty(ti=t2, tj=t3, Si=S1, Sij=Sij)
        S4 = self.propagate_uncertainty(ti=t3, tj=t4, Si=S2, Sij=Sab2)

        # compute pairwise consistency. Mahalanobis distance
        Dij = np.dot(s, np.dot(np.linalg.inv(S4), s.T))
        Aij = np.exp(Dij)
        # print('Loop closing distance: D:', I)
        return Aij, Dij

    def propagate_uncertainty(self, ti, tj, Si, Sij):
        """
        Computes a Gaussian linear propagation law based on the uncertainty on node u and the relative uncertainty
        between u and v
        """
        Ti = HomogeneousMatrix(np.array([ti[0], ti[1], 0]), Euler([0, 0, ti[2]]))
        Tj = HomogeneousMatrix(np.array([tj[0], tj[1], 0]), Euler([0, 0, tj[2]]))
        Tij = Ti.inv() * Tj
        tij = Tij.t2v()
        [J1, J2] = self.compute_jacobians(ti, tij)
        Sj = np.dot(J1, np.dot(Si, J1.T)) + np.dot(J2, np.dot(Sij, J2.T))
        return Sj

    def compute_jacobians(self, ti, tij):
        """
        Computes Jacobians for the variables in ti and tij
        """
        ci = np.cos(ti[2])
        si = np.sin(ti[2])
        xij = tij[0]
        yij = tij[1]
        J1 = np.array([[1, 0, - si * xij - ci * yij],
                       [0, 1, ci * xij - si * yij],
                       [0, 0, 1]])
        J2 = np.array([[ci, - si, 0],
                       [si, ci, 0],
                       [0, 0, 1]])
        return J1, J2

    #
    # def compute_pairwise_consistency_matrix(self, candidates):
    #     """
    #     Selects association hypothesis in pairs.
    #     For each pair, compute a loop closing transformation and an associated probability
    #     """
    #     N = len(candidates)
    #     A = np.zeros(N, N)
    #     # find all combinations of i and j of the hypotheses
    #     for i in range(len(candidates)):
    #         for j in range(len(candidates)):
    #             if i == j:
    #                 continue
    #             # compute the rigid transformation that connects the candidates.
    #             t, S = self.rigid_transformation(candidates[i], candidates[j])
    #             A[i][j] = np.exp(np.dot(t, np.dot(S, t.T)))
    #     return A

    #
    # def simulate_observations_SE2(x_gt, observations):
    #     """
    #     x_gt: ground truth poses
    #     A series of relative observations are generated from the ground truth solution x_gt
    #     """
    #     N = len(observations)
    #     edges = []
    #     sx = self.icp_noise[0][0]
    #     sy = self.icp_noise[1][1]
    #     sth = self.icp_noise[2][2]
    #     for k in range(N):
    #         i = observations[k][0]
    #         j = observations[k][1]
    #         ti = np.hstack((x_gt[i, 0:2], 0))
    #         tj = np.hstack((x_gt[j, 0:2], 0))
    #         Ti = HomogeneousMatrix(ti, Euler([0, 0, x_gt[i, 2]]))
    #         Tj = HomogeneousMatrix(tj, Euler([0, 0, x_gt[j, 2]]))
    #         Tiinv = Ti.inv()
    #         zij = Tiinv * Tj
    #         zij = zij.t2v()
    #         # add noise to the observatoins
    #         zij = zij + np.array([np.random.normal(0, sx, 1)[0],
    #                               np.random.normal(0, sy, 1)[0],
    #                               np.random.normal(0, sth, 1)[0]])
    #         # np.random.normal([0, 0, 0], [sx, sy, sth], 1)
    #         zij[2] = mod_2pi(zij[2])
    #         Tij = HomogeneousMatrix([zij[0], zij[1], 0], Euler([0, 0, zij[2]]))
    #         edges.append(Tij)
    #     return edges
    #

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

    # def marginal_covariance(self, i):
    #     # init initial estimate, read from self.current_solution
    #     initial_estimate = gtsam.Values()
    #     k = 0
    #     for pose2 in self.graphslam.current_solution:
    #         initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
    #         k = k+1
    #     marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
    #     cov = marginals.marginalCovariance(i)
    #     return cov
    #
    # def joint_marginal_covariance(self, i, j):
    #     # init initial estimate, read from self.current_solution
    #     initial_estimate = gtsam.Values()
    #     k = 0
    #     for pose2 in self.graphslam.current_solution:
    #         initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
    #         k = k+1
    #     marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
    #     keyvector = gtsam.utilities.createKeyVector([i, j])
    #     jm = marginals.jointMarginalCovariance(variables=keyvector).at(iVariable=i, jVariable=j)
    #     return jm
    #
    # def marginal_information(self, i):
    #     # init initial estimate, read from self.current_solution
    #     initial_estimate = gtsam.Values()
    #     k = 0
    #     for pose2 in self.graphslam.current_solution:
    #         initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
    #         k = k+1
    #     marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
    #     cov = marginals.marginalInformation(i)
    #     return cov
    #
    # def joint_marginal_information(self, i, j):
    #     # init initial estimate, read from self.current_solution
    #     initial_estimate = gtsam.Values()
    #     k = 0
    #     for pose2 in self.graphslam.current_solution:
    #         initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
    #         k = k+1
    #     marginals = gtsam.Marginals(self.graphslam.graph, initial_estimate)
    #     keyvector = gtsam.utilities.createKeyVector([i, j])
    #     jm = marginals.jointMarginalInformation(variables=keyvector).at(iVariable=i, jVariable=j)
    #     return jm
    #
    # def joint_mahalanobis(self, i, j, only_position=False):
    #     """
    #     Using an approximation for the joint conditional probability of node i and j
    #     """
    #     Oii = self.marginal_information(i)
    #     Ojj = self.marginal_information(j)
    #     Inf_joint = Oii + Ojj
    #     muii = self.graphslam.current_estimate[i]
    #     mujj = self.graphslam.current_estimate[j]
    #     mu = mujj-muii
    #     mu[2] = mod_2pi(mu[2])
    #     # do not consider orientation
    #     if only_position:
    #         mu[2] = 0.0
    #     d2 = np.abs(np.dot(mu.T, np.dot(Inf_joint, mu)))
    #     return d2
    #

    #
    # def test_conditional_probabilities(self):
    #     """
    #     """
    #     muii = self.graphslam.current_estimate[13]
    #     Sii = self.marginal_covariance(13)
    #
    #     Sjj = self.marginal_covariance(1)
    #     mujj = self.graphslam.current_estimate[1]
    #
    #     Sij = self.joint_marginal_covariance(1, 13)
    #     Sji = self.joint_marginal_covariance(13, 1)
    #
    #     Sii_ = Sii - np.dot(Sij, np.dot(np.linalg.inv(Sjj), Sij.T))
    #     Sjj_ = Sjj - np.dot(Sij.T, np.dot(np.linalg.inv(Sii), Sij))
    #
    #
    #     # product, joint probability
    #     Sca = np.linalg.inv(np.linalg.inv(Sii) + np.linalg.inv(Sjj))
    #     Scb = Sii + Sjj
    #     a1 = np.dot(np.linalg.inv(Sii), muii)
    #     a2 = np.dot(np.linalg.inv(Sjj), mujj)
    #     mc = np.dot(Sca, a1+a2)
    #
    #     # gtsam_plot.plot_pose2(0, gtsam.Pose2(muii[0], muii[1], muii[2]), 0.5, Sii)
    #     # gtsam_plot.plot_pose2(0, gtsam.Pose2(mujj[0], mujj[1], mujj[2]), 0.5, Sjj)
    #     # gtsam_plot.plot_pose2(0, gtsam.Pose2(0, 0, 0), 0.5, Sii_)
    #     # gtsam_plot.plot_pose2(0, gtsam.Pose2(0, 1.5, 0), 0.5, Sjj_)
    #     # gtsam_plot.plot_pose2(0, gtsam.Pose2(mc[0], mc[1], mc[2]), 0.5, Sca)
    #
    #     for i in range(10):
    #         mu = 0.5*(mujj + muii)
    #         mu[2] = 0
    #         muij = mujj - muii
    #         muij[2] = 0
    #
    #         gtsam_plot.plot_pose2(0, gtsam.Pose2(muii[0], muii[1], muii[2]), 0.5, Sii)
    #         gtsam_plot.plot_pose2(0, gtsam.Pose2(mujj[0], mujj[1], mujj[2]), 0.5, Sjj)
    #         gtsam_plot.plot_pose2(0, gtsam.Pose2(mu[0], mu[1], mu[2]), 0.5, Sca)
    #         gtsam_plot.plot_pose2(0, gtsam.Pose2(mu[0], mu[1], mu[2]), 0.5, Scb)
    #
    #         d0 = np.dot(muij.T, np.dot(np.linalg.inv(Sca), muij))
    #         d1 = np.dot(muij.T, np.dot(np.linalg.inv(Scb), muij))
    #         # d2 = np.dot(muij.T, np.dot(np.linalg.inv(Sii_), muij))
    #         # d3 = np.dot(muij.T, np.dot(np.linalg.inv(Sjj_), muij))
    #
    #         muii += np.array([0.2, 0, 0])
    #     return True
    #
    # def view_full_information_matrix(self):
    #     """
    #     The function i
    #     """
    #     n = self.graphslam.current_index + 1
    #     H = np.zeros((3*n, 3*n))
    #
    #     for i in range(n):
    #         Hii = self.marginal_information(i)
    #         print(i, i)
    #         print(Hii)
    #         H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Hii
    #
    #     for i in range(n):
    #         for j in range(n):
    #             if i == j:
    #                 continue
    #             Hij = self.joint_marginal_information(i, j)
    #             print(i, j)
    #             print(Hij)
    #             H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = Hij
    #
    #     plt.figure()
    #     plt.matshow(H)
    #     plt.show()
    #     return True
    #
    # def view_full_covariance_matrix(self):
    #     """
    #     """
    #     n = self.graphslam.current_index + 1
    #     H = np.zeros((3*n, 3*n))
    #
    #     for i in range(n):
    #         Hii = self.marginal_covariance(i)
    #         print(i, i)
    #         print(Hii)
    #         H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Hii
    #
    #     for i in range(n):
    #         for j in range(n):
    #             if i == j:
    #                 continue
    #             Hij = self.joint_marginal_covariance(i, j)
    #             print(i, j)
    #             print(Hij)
    #             H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = Hij
    #
    #     plt.figure()
    #     plt.matshow(H)
    #     plt.show()
    #     return True


