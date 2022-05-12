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
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
from graphslam.graphslam import GraphSLAM
import gtsam
from tools.conversions import mod_2pi

SIGMAS_PRIOR = gtsam.Point3(0.1, 0.1, 0.01)
SIGMAS_ICP = gtsam.Point3(0.3, 0.3, 0.05)
SIGMAS_ICP_REAL = gtsam.Point3(0.05, 0.05, 0.01)
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas=SIGMAS_PRIOR)
ICP_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas=SIGMAS_ICP)
# add less noise, since gaussians are linearized approximations
ICP_NOISE_REAL = gtsam.noiseModel.Diagonal.Sigmas(sigmas=SIGMAS_ICP_REAL)


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


def simulate_observations_SE2(x_gt, observations, sigmas_icp):
    """
    x_gt: ground truth poses
    A series of relative observations are generated from the ground truth solution x_gt
    """
    N = len(observations)
    edges = []
    sx = sigmas_icp[0]
    sy = sigmas_icp[1]
    sth = sigmas_icp[2]
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
        zij[2] = mod_2pi(zij[2])
        edges.append(zij)
    edges = np.array(edges)
    return edges


def main():
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, xi2_th=20.0)

    for k in range(1, len(x_gt)):
        # Vertex j is observed from vertex i
        i = k-1
        j = k
        atb = simulate_observations_SE2(x_gt=x_gt, observations=[[i, j]], sigmas_icp=SIGMAS_ICP_REAL)
        # consecutive edges. Adds a new node
        graphslam.add_consecutive_observation(atb[0])
        graphslam.view_solution()
        # now check for possible data associations.
        associations = dassoc.perform_data_association()
        # non-consecutive edges
        for assoc in associations:
            # caution: need a true observation! atb
            atb = simulate_observations_SE2(x_gt=x_gt, observations=[assoc], sigmas_icp=SIGMAS_ICP_REAL)
            # i, j. node j observed from node i
            graphslam.add_non_consecutive_observation(assoc[0], assoc[1], atb[0])
        # optimizing whenever non_consecutive observations are performed (loop closing)
        if len(associations) > 0:
            graphslam.optimize()
        # or optimizing at every new observation
        # graphslam.optimize()
        # graphslam.view_solution()
    # or optimizing when all information is available
    graphslam.optimize()
    graphslam.view_solution()


if __name__ == "__main__":
    main()
