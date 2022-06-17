"""
Simple experiment using GTSAM in a GraphSLAM context.

The slam_tutorial presents the case in which a robot moves along the environment and is able to compute the
relative transformations between the poses in the graph (a fact that is frequent, for example, if we can perform
a scanmatching between the scans captured at each of the poses).

This first example considers that data associations are known!

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

from tools.dijkstra_projection import DijkstraProjection
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
from graphslam.graphslam import GraphSLAM
import gtsam

prior_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 0.02
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
icp_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 0.04

PRIOR_NOISE_MATRIX = np.array([prior_rpy_sigma*np.pi/180,
                               prior_rpy_sigma*np.pi/180,
                               prior_rpy_sigma*np.pi/180,
                               prior_xyz_sigma,
                               prior_xyz_sigma,
                               prior_xyz_sigma])
ICP_NOISE_MATRIX = np.array([icp_rpy_sigma*np.pi/180,
                             icp_rpy_sigma*np.pi/180,
                             icp_rpy_sigma*np.pi/180,
                             icp_xyz_sigma,
                             icp_xyz_sigma,
                             icp_xyz_sigma])

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_NOISE_MATRIX)
ICP_NOISE = gtsam.noiseModel.Diagonal.Sigmas(ICP_NOISE_MATRIX)


# x_gt = np.array([[0, 0, 0],  # 0
#                      [5, 0, 0],  # 1
#                      [10, 0, 0],  # 2
#                      [10, 0, np.pi / 2],  # 3
#                      [10, 5, np.pi / 2],  # 4
#                      [10, 10, np.pi / 2],  # 5
#                      [10, 10, np.pi],  # 6
#                      [5, 10, np.pi],  # 7
#                      [0, 10, np.pi],  # 8
#                      [0, 10, 3 * np.pi / 2],  # 9
#                      [0, 5, 3 * np.pi / 2],  # 10
#                      [0, 0, 3 * np.pi / 2],  # 11
#                      [0, 0, 0],  # 12
#                      [5, 0, 0],  # 13
#                      [10, 0, 0],  # 14
#                      [10, 0, np.pi / 2]])  # 15


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
                     [0, 5, 3 * np.pi / 2]])  # 10


def mod_2pi(th):
    """
    Return theta in [-pi, pi]
    """
    thetap = np.arctan2(np.sin(th), np.cos(th))
    return thetap


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


def simulate_experiment():
    print('GRAPHSLAM')
    # SIMULATE MAPPING PROCESS. First, simulate odometry
    # correspondences consecutive observations (imu, odometry, ICP)
    observations = []
    for i in range(len(x_gt) - 1):
        observations.append([i, i + 1])
    # non-CONSECUTIVE observations
    # observations.extend([[5, 0], [6, 2], [15, 3], [11, 0], [12, 0], [13, 1], [14, 2],
    #                      [15, 0], [15, 1], [15, 2], [15, 3]])
    observations.extend([[5, 3], [6, 2], [9, 6], [2, 0]])
    # observations.extend([[5, 0]])
    # assuming ground truth, simulate an icp with noise in tx, ty, and th
    # given a series of relative observations ij
    noise_edges = simulate_observations_SE2(x_gt=x_gt, observations=observations)
    return observations, noise_edges


def main():
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # simulate experiment
    observations, noise_edges = simulate_experiment()
    # for k in range(len(observations)):
    for k in range(10):
        # Vertex j is observed from vertex i
        i = observations[k][0]
        j = observations[k][1]
        atb = noise_edges[k]
        # consecutive edges. Adds a new node
        if j-i == 1:
            graphslam.add_consecutive_observation(atb)
        # non-consecutive edges: i. e. loop closing
        else:
            graphslam.add_loop_closing_observation(i, j, atb)
            # optimizing whenever non_consecutive observations are performed (loop closing)
            graphslam.optimize()
        # or optimizing at every new observation
        # graphslam.optimize()
        graphslam.view_solution2D(skip=1)
    graphslam.optimize()


    # Caution, the dijkstra projection considers a SE2 uncertainty propagation
    dijkstrap = DijkstraProjection(graphslam.current_estimate, observations, Sigmaij=np.diag([icp_xyz_sigma,
                                                                                              icp_xyz_sigma,
                                                                                              icp_rpy_sigma]))
    shortest_path, Sigma_ab = dijkstrap.compute_shortest_path(10, 0)
    print('Shortest uncertainty path: ', shortest_path)
    print('Sigma_ab: ', Sigma_ab)
    # or optimizing when all information is available



if __name__ == "__main__":
    main()
