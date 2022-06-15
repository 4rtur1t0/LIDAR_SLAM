"""
Simple experiment using GTSAM in a GraphSLAM context.

The slam_tutorial presents the case in which a robot moves along the environment and is able to compute the
relative transformations between the poses in the graph (a fact that is frequent, for example, if we can perform
a scanmatching between the scans captured at each of the poses).

In this second example DA is considered to be unknown. At each time step, a Sigma_ji conditioned covariance matrix
is computed. The data association is associated to the past

The simulate_function considers that the robot travels around a set of poses. The robot is capable of observing the
relative transformation between consecutive poses. In addition, the robot is able to observe poses that were visited
in the past (loop closing). Next, the function simulate_observations_SE2 is able to compute the transformation from
the reference system of node i when viewed from node j. A noise (ICP_NOISE) is added to that observation.

The gtsam library is used in a SLAM context. When a new laser scan is received:
- The relative transformation is computed with respect to the previous scan.
- A new vertex is created in the graph (graph.add_consecutive_observation()).
- Also graph.add_consecutive_observation() adds an edge between the pose i-1 and i.
- The data association is executed. If the data association decides that the pose j can be observed from the pose i.
    - graphslam.add_non_consecutive_observation(i, j, atb) is called.
    - this method creates an edge between vertices i and j.
    - whenever an edge is created between non-consecutive edges, we perform an optimization of the graph. (graph.optimize)

"""
import numpy as np
from graphslam.dataassociation import DataAssociation
from tools.dijkstra_projection import DijkstraProjectionRelative
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
from graphslam.graphslam import GraphSLAM
import gtsam
from tools.conversions import mod_2pi

prior_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 0.02
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
icp_xyz_sigma = 0.01
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

x_gt = np.array([[0, 0, 0],  # 0
             [5, 0, 0],  # 1
             [10, 0, 0],  # 2
             [10, 0, np.pi / 2],  # 3
             [10, 5, np.pi / 2],  # 4
             [10, 10, np.pi / 2],  # 5
             [10, 10, np.pi],  # 6
             [5, 10, np.pi],  # 7
             [0, 10, np.pi],  # 8
             [0, 10, 3 * np.pi / 2],  # 9
             [0, 5, 3 * np.pi / 2],  # 10
             [0, 0, 3 * np.pi / 2],  # 11
             [0, 0, 0],  # 12
             [5, 0, 0],  # 13
             [10, 0, 0],  # 14
             [10, 0, np.pi/2],
                 [10, 5, np.pi / 2],  # 4
                 [10, 10, np.pi / 2],  # 5
                 [10, 10, np.pi],  # 6
                 [5, 10, np.pi],  # 7
                 [0, 10, np.pi],  # 8
                 [0, 10, 3 * np.pi / 2],  # 9
                 [0, 5, 3 * np.pi / 2],  # 10
                 [0, 0, 3 * np.pi / 2],  # 11
                 [0, 0, 0],  # 12
                 [5, 0, 0]])  # 15


def simulate_observations_SE2(x_gt, observations):
    """
    x_gt: ground truth poses
    A series of relative observations are generated from the ground truth solution x_gt
    """
    N = len(observations)
    edges = []
    sx = icp_xyz_sigma
    sy = icp_xyz_sigma
    sth = icp_rpy_sigma
    for k in range(N):
        i = observations[k][0]
        j = observations[k][1]
        ti = np.hstack((x_gt[i, 0:2], 0))
        tj = np.hstack((x_gt[j, 0:2], 0))
        Ti = HomogeneousMatrix(ti, Euler([0, 0, x_gt[i, 2]]))
        Tj = HomogeneousMatrix(tj, Euler([0, 0, x_gt[j, 2]]))
        Tiinv = Ti.inv()
        zij = Tiinv*Tj
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


def compute_pairwise_consistency_matrix(graphslam, dijkstra_algorithm, candidates, atbs):
    """
    Given a set of candidate hypotheses with N > 1. Each data association is accompanied by a computed transformation
    i.e. using ICP.
    Arrange the correspondences in pairs.
    Compute a loop closing transformation.
    Compute A mahalanobis distance s.TSs
    """
    N = len(candidates)
    A = np.zeros((N, N))
    D = np.zeros((N, N))
    for i in range(0, N):
        for j in range(i+1, N):
            if i == j:
                continue
            pair_candidates = [candidates[i], candidates[j]]
            pair_transforms = [atbs[i], atbs[j]]
            # computes pairwise consistency between two candidates!
            Aij, Dij = compute_pairwise_consistency(graphslam, dijkstra_algorithm, pair_candidates, pair_transforms)
            A[i][j] = Aij
            A[j][i] = Aij
            D[i][j] = Dij
            D[j][i] = Dij
    return A, D


def compute_pairwise_consistency(graphslam, dijkstra_algorithm, candidates, atbs):
    # the first measurement corresponds to the first correspondence hypothesis
    Tm1 = atbs[0]
    # the second measurement corresponds to the second correspondence hypothesis
    # it is inverted, given the order in which the transformations are performed
    Tm2 = atbs[1].inv()
    # set the absolute poses for each node in the dijkstra algorithm
    dijkstra_algorithm.set_poses(graphslam.current_estimate)
    source = candidates[0][1]
    finish = candidates[1][1]
    # select source and finish nodes, obtain, the path, relative transformation and relative uncertainty using
    # the Dijkstra projection
    shortest_path1, Tab1, Sab1 = dijkstra_algorithm.compute_shortest_path(source=source, finish=finish)
    # again, select source and finish nodes, obtain, the path, relative transformation and relative uncertainty using
    # the Dijkstra projection
    source = candidates[1][0]
    finish = candidates[0][0]
    shortest_path2, Tab2, Sab2 = dijkstra_algorithm.compute_shortest_path(source=source, finish=finish)

    print('Shortest path found 1: ', shortest_path1)
    print('Shortest path found 2: ', shortest_path2)
    print('Sigma AB1: ', Sab1)
    print('Sigma AB2: ', Sab2)
    print('Transformation AB: ', Tab1)
    print('Transformation AB: ', Tab2)
    # compute the loop closing tranformation, that includes: a measured transformation corresponding with the
    # data association. A transformation based on Dijkstra path. Another measured data association
    I = Tm1*Tab1*Tm2*Tab2
    s = I.t2v()
    print('Loop closing constraint: I:', I)
    print('Loop closing constraint: s:', s)
    t0 = np.array([0, 0, 0])
    t1 = Tm1.t2v(n=3)
    t2 = (Tm1*Tab1).t2v()
    t3 = (Tm1*Tab1*Tm2).t2v()
    t4 = (Tm1 * Tab1 * Tm2*Tab2).t2v()
    # icp noise matrix for measurements
    Sij = np.diag([icp_xyz_sigma, icp_xyz_sigma, icp_rpy_sigma])
    # propagate uncertainty, Measured, Dijkstra, Measured, Dijstra
    S0 = propagate_uncertainty(ti=t0, tj=t1, Si=np.diag([0, 0, 0]), Sij=Sij)
    S1 = propagate_uncertainty(ti=t1, tj=t2, Si=S0, Sij=Sab1)
    S2 = propagate_uncertainty(ti=t2, tj=t3, Si=S1, Sij=Sij)
    S4 = propagate_uncertainty(ti=t3, tj=t4, Si=S2, Sij=Sab2)

    # compute pairwise consistency. Mahalanobis distance
    Dij = np.dot(s, np.dot(np.linalg.inv(S4), s.T))
    Aij = np.exp(Dij)
    # print('Loop closing distance: D:', I)
    return Aij, Dij


def propagate_uncertainty(ti, tj, Si, Sij):
    """
    Computes a Gaussian linear propagation law based on the uncertainty on node u and the relative uncertainty
    between u and v
    """
    Ti = HomogeneousMatrix(np.array([ti[0], ti[1], 0]), Euler([0, 0, ti[2]]))
    Tj = HomogeneousMatrix(np.array([tj[0], tj[1], 0]), Euler([0, 0, tj[2]]))
    Tij = Ti.inv()*Tj
    tij = Tij.t2v()
    [J1, J2] = compute_jacobians(ti, tij)
    Sj = np.dot(J1, np.dot(Si, J1.T)) + np.dot(J2, np.dot(Sij, J2.T))
    return Sj


def compute_jacobians(ti, tij):
    """
    Computes Jacobians for the variables in ti and tij
    """
    ci = np.cos(ti[2])
    si = np.sin(ti[2])
    xij = tij[0]
    yij = tij[1]
    J1 = np.array([[1,  0, - si * xij - ci * yij],
                   [0,  1,  ci * xij - si * yij],
                   [0,  0,      1]])
    J2 = np.array([[ci, - si,   0],
                   [si,  ci,    0],
                   [0,  0,     1]])
    return J1, J2



def main():
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, xi2_th=20.0, icp_noise=ICP_NOISE_DA)
    dijkstra_algorithm = DijkstraProjectionRelative(np.diag([icp_xyz_sigma, icp_xyz_sigma, icp_rpy_sigma]))
    dijkstra_algorithm.add_node()

    # Constructs the graph for data association
    # caution, loop closing associations are not included in this graph
    for k in range(1, 16):
        # Vertex j is observed from vertex i
        i = k-1
        j = k
        atb = simulate_observations_SE2(x_gt=x_gt, observations=[[i, j]])
        # consecutive edges. Adds a new node
        graphslam.add_consecutive_observation(atb[0])
        # adding nodes for the Dijkstra projection
        dijkstra_algorithm.add_node()
        dijkstra_algorithm.connect_nodes(i, j)

    graphslam.view_solution2D(skip=1)

    # DATA ASSOCIATION
    #candidates = [[12, 0], [2, 12]]
    candidates = [[11, 1],
                  [11, 0],
                  [11, 3],
                  [13, 3],
                  [15, 0],
                  [15, 1],
                  [14, 0],
                  [14, 3]]
    atbs = simulate_observations_SE2(x_gt=x_gt, observations=candidates)
    # two observations of the set are corrupted
    atbs[0] = HomogeneousMatrix(np.eye(4))
    atbs[2] = HomogeneousMatrix(np.eye(4))
    [A, D] = compute_pairwise_consistency_matrix(graphslam, dijkstra_algorithm, candidates, atbs)

    # print values
    for i in range(0, len(candidates)):
        for j in range(i+1, len(candidates)):
            if i == j:
                continue
            print('Candidate: ', i, 'with candidate: ', j)
            print(D[i][j])

    # now select the valid hypotheses based on the joint Mahalanobis distances
    likelihood_candidates = np.zeros(len(candidates))
    for i in range(0, len(candidates)):
        for j in range(0, len(candidates)):
            if i == j:
                continue
            if D[i][j] < 3:
                likelihood_candidates[i] += 1
                likelihood_candidates[j] += 1
    # filter good candidates
    likelihood_candidates = likelihood_candidates/np.sum(likelihood_candidates)
    idx = np.nonzero(likelihood_candidates)[0]
    candidates = np.array(candidates)
    candidates = candidates[idx]
    print('Likelihood candidates: ', likelihood_candidates)
    print('VALID HYPOTHESES ARE: ')
    print(candidates)


if __name__ == "__main__":
    main()
