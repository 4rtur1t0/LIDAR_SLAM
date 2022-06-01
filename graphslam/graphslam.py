"""

"""
from __future__ import print_function
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix
import numpy as np


class GraphSLAM():
    def __init__(self, prior_noise, icp_noise):
        self.current_index = 0
        self.graph = gtsam.NonlinearFactorGraph()
        # self.graph = gtsam.GaussianFactorGraph()
        # self.current_solution = np.array([odom0])
        # first pose, add now
        # self.graph.add(gtsam.PriorFactorPose2(self.current_index, gtsam.Pose2(odom0[0], odom0[1], odom0[2]), prior_noise))
        # Add a prior on pose x0, with 0.3 rad std on roll,pitch,yaw and 0.1m x,y,z
        # prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        self.graph.add(gtsam.PriorFactorPose3(self.current_index, gtsam.Pose3(), prior_noise))
        self.temp_estimate = gtsam.Values()
        self.temp_estimate.insert(self.current_index, gtsam.Pose3())
        self.current_estimate = self.temp_estimate
        self.n_vertices = 1 # the prior above is an edge
        self.n_edges = 0
        self.ICP_NOISE = icp_noise

        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        # parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

    def add_consecutive_observation(self, atb):
        """
        aTb is a relative transformation from a to b
        Add a vertex considering two consecutive poses
        """
        self.n_vertices = self.n_vertices + 1
        self.n_edges = self.n_edges + 1
        k = self.current_index
        # add consecutive observation
        # self.graph.add(gtsam.BetweenFactorPose2(k, k+1, gtsam.Pose2(atb[0], atb[1], atb[2]), self.ICP_NOISE))
        self.graph.add(gtsam.BetweenFactorPose3(k, k + 1, gtsam.Pose3(atb.array), self.ICP_NOISE))

        # compute next estimation
        next_estimate = self.current_estimate.atPose3(k).compose(gtsam.Pose3(atb.array))
        self.temp_estimate.insert(k + 1, next_estimate)
        self.current_index = k + 1
        # compound relative transformation to the last pose
        # cs = self.current_solution[-1]
        # T = HomogeneousMatrix([cs[0], cs[1], 0], Euler([0, 0, cs[2]]))
        # Trel = HomogeneousMatrix([atb[0], atb[1], 0], Euler([0, 0, atb[2]]))
        # T = T*Trel
        # concatenate prior solution
        # self.current_solution = np.vstack((self.current_solution, T.t2v()))

    def add_loop_closing_observation(self, i, j, aTb):
        """
        aTb is a relative transformation from frame i to frame j
        Add a vertex considering a loop closing observation where i-j > 1
        """
        self.n_edges = self.n_edges + 1
        # add non consecutive observation
        # self.graph.add(gtsam.BetweenFactorPose2(int(i), int(j), gtsam.Pose2(aTb[0], aTb[1], aTb[2]), self.ICP_NOISE))
        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(aTb.array), self.ICP_NOISE))

    def optimize(self):
        self.isam.update(self.graph, self.temp_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.temp_estimate.clear()


        # # init initial estimate, read from self.current_solution
        # initial_estimate = gtsam.Values()
        # k = 0
        # for c_solution_k in self.current_solution:
        #     initial_estimate.insert(k, gtsam.Pose2(c_solution_k[0], c_solution_k[1], c_solution_k[2]))
        #     k = k+1
        # # solver parameters
        # parameters = gtsam.GaussNewtonParams()
        # # Stop iterating once the change in error between steps is less than this value
        # parameters.setRelativeErrorTol(1e-5)
        # # Do not perform more than N iteration steps
        # parameters.setMaxIterations(100)
        # # Create the optimizer ...
        # optimizer = gtsam.GaussNewtonOptimizer(self.graph, initial_estimate, parameters)
        #
        # # ... and optimize
        # result = optimizer.optimize()
        # print("Final Result:\n{}".format(result))

        # print("GRAPH")
        # print(self.graph)

        # 5. Calculate and print marginal covariances for all variables
        # marginals = gtsam.Marginals(self.graph, result)
        # for i in range(self.n_vertices):
        #     print("X{} covariance:\n{}\n".format(i,
        #                                          marginals.marginalCovariance(i)))
        #
        # for i in range(self.n_vertices):
        #     gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5,
        #                           marginals.marginalCovariance(i))
        # plt.axis('equal')
        # plt.show()

        # now save the solution, caution, the current solution is used as initial
        # estimate in subsequent calls to optimize
        # self.current_solution = []
        # for i in range(self.n_vertices):
        #     x = result.atPose2(i).translation()[0]
        #     y = result.atPose2(i).translation()[1]
        #     th = result.atPose2(i).rotation().theta()
        #     self.current_solution.append(np.array([x, y, th]))
        # self.current_solution = np.array(self.current_solution)

    def view_solution(self):
        """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""

        # Print the current estimates computed using iSAM2.
        print("*" * 50 + f"\nInference after State:\n", self.current_index)
        # print(self.current_estimate)

        # Compute the marginals for all states in the graph.
        marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        axes = fig.gca(projection='3d')
        plt.cla()

        i = 0
        while self.current_estimate.exists(i):
            gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 1,
                                  marginals.marginalCovariance(i))
            i += 20

        axes.set_xlim3d(-50, 50)
        axes.set_ylim3d(-50, 50)
        axes.set_zlim3d(-50, 50)
        plt.pause(0.05)

    def view_solution_fast(self, skip=1):
        """
        Plot incremental progress without uncertainty
        """
        # Print the current estimates computed using iSAM2.
        print("*" * 50 + f"\nInference after State:\n", self.current_index)
        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        axes = fig.gca(projection='3d')
        plt.cla()

        i = 0
        while self.current_estimate.exists(i):
            gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 1)
            i += skip

        axes.set_xlim3d(-50, 50)
        axes.set_ylim3d(-50, 50)
        axes.set_zlim3d(-1, 10)
        plt.pause(0.0001)

    def get_solution_transforms(self):
        """
        Get a list of solutions in terms of homogenous matrices in global coordinates.
        """
        # print(self.current_estimate)
        transforms = []
        i = 0
        while self.current_estimate.exists(i): #or i in range(len(self.current_estimate)):
            a = self.current_estimate.atPose3(i)
            # converting from Pose3 to numpy array and Homogenous matrix
            transforms.append(HomogeneousMatrix(a.matrix()))
            i += 1
        return transforms

    def get_solution(self):
        return self.current_estimate


    # def view_solution(self):
    #     # init initial estimate, read from self.current_solution
    #     initial_estimate = gtsam.Values()
    #     k = 0
    #     for pose2 in self.current_solution:
    #         initial_estimate.insert(k, gtsam.Pose2(pose2[0], pose2[1], pose2[2]))
    #         k = k+1
    #     marginals = gtsam.Marginals(self.graph, initial_estimate)
    #     for i in range(self.n_vertices):
    #         gtsam_plot.plot_pose2(0, initial_estimate.atPose2(i), 0.5,
    #                               marginals.marginalCovariance(i))
    #     plt.axis('equal')
    #     plt.show()



