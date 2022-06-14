"""
The Dijkstra algorithm that read information from gtsam graphslam
"""
import sys
import numpy as np
from tools.euler import Euler
from tools.homogeneousmatrix import HomogeneousMatrix


class DijkstraProjection():
    def __init__(self, current_estimate, observations, Sigmaij):
        # number of nodes in the graph (equals the number of poses estimated so far)
        self.nnodes = current_estimate.size()
        # covariance noise between every two nodes
        self.Sigmaij = Sigmaij
        self.poses = []
        self.Sigmas = [None]*self.nnodes
        self.graph = np.zeros((self.nnodes, self.nnodes))
        # self.weights = np.zeros((nnodes, nnodes))
        # In this algorithm, the distance is based on the determinant of the uncertainty between nodes.
        self.dists = np.array([sys.maxsize] * self.nnodes, dtype=float)
        # list of visited/unvisited nodes. All nodes marked as false (not visited)
        self.S = [False] * self.nnodes
        # the shortest paths are stored as a parent array
        self.shortest_paths = np.array([-1] * self.nnodes)
        self.shortest_path = []
        self.init_poses(current_estimate, observations=observations)

    def init_poses(self, current_estimate, observations):
        """
        Initializes a Ti matrix for every node in the graph.
        """
        # Sij i known for every pair of nodes
        # Converting from SE3 to SE2 for uncertainty propagation
        i = 0
        while current_estimate.exists(i):
            pose3 = current_estimate.atPose3(i)
            x = pose3.translation()[0]
            y = pose3.translation()[1]
            th = pose3.rotation().ypr()[0]
            pose2 = HomogeneousMatrix([x, y, 0], Euler([0, 0, th]))
            self.poses.append(pose2)
            i += 1
        # now create the graph. First connect adjacent nodes. then use observations to connect relative nodes.
        for i in range(self.nnodes-1):
            self.connect_nodes(i, i+1)
        for observation in observations:
            self.connect_nodes(observation[0], observation[1])

    def connect_nodes(self, i, j):
        """
        Asumes different costs when going from i to j and viceversa
        """
        self.graph[i, j] = True
        self.graph[j, i] = True

    def is_connected(self, i, j):
        if self.graph[i, j]:
            return True
        else:
            return False

    def print(self):
        print('Graph:')
        print(self.graph)
        print('Distance from source: ')
        for i in range(self.nnodes):
            print('Node ', i, ': ', self.dists[i])
        print('Visited nodes: ')
        for i in range(self.nnodes):
            print('Node ', i, ': ', self.S[i])
        print('Shouts: (yes, the algorithm is trying to grow along shouts based on min distance)')
        for i in range(self.nnodes):
            if self.dists[i] < sys.maxsize and not self.S[i]:
                print('Node ', i, ' min dist: ', self.dists[i])
        print('Parent array of shortest paths: ')
        print(self.shortest_paths)

    def store_shortest_path(self, source, finish):
        """
        Follows the parent array backwards from finish to source
        """
        backwards = [finish]
        next_node = finish
        while next_node != source:
            next_node = self.shortest_paths[next_node]
            backwards.append(next_node)
        backwards.reverse()
        self.shortest_path = np.array(backwards)

    def mindistance(self):
        """
        Selects the minimum distance for the nodes that are not included in S.
        That is, look for the minimum distance so far for the nodes that have not been visited yet.
        """
        min_value = sys.maxsize
        min_index = 0
        for v in range(self.nnodes):
            if self.dists[v] < min_value and not self.S[v]:
                min_value = self.dists[v]
                min_index = v
        return min_index

    def update_distances(self, v):
        """
        Finds nodes adjacent to v and updates the distance in a greedy manner.
        If the distance found is less than the previous distance, the distance is saved.
        """
        # find adjacent nodes of v
        for u in range(self.nnodes):
            if not self.is_connected(v, u):
                continue
            # if connected, propagate uncertainty, compute determinant and sum to total distance at node u
            Sigmau = self.propagate_uncertainty(v, u)
            # caution, in a vanilla Dijkstra's algorithm, we may find here an addition,
            # however, the sum of distances (the propagation via Jacobians) is computed in self.propagate_uncertainty
            # thus, the total distance from node v to node u is actually the determinant of the covariance matrix Sigma_uv
            # temp_dist = self.dists[v] + weight
            temp_dist = np.linalg.det(Sigmau)
            # if a shorter path has been found, store it
            if temp_dist < self.dists[u]:
                self.dists[u] = temp_dist
                # update sigma for the new node!
                self.Sigmas[u] = Sigmau
                # update the parent array
                self.shortest_paths[u] = v

    def propagate_uncertainty(self, u, v):
        """
        Computes a Gaussian linear propagation law based on the uncertainty on node u and the relative uncertainty
        between u and v
        """
        Ti = self.poses[u]
        Tj = self.poses[v]
        Tij = np.dot(Ti.inv(), Tj)
        ti = Ti.t2v()
        tij = Tij.t2v()
        [J1, J2] = self.compute_jacobians(ti, tij)
        Si = self.Sigmas[u]
        Sij = self.Sigmaij
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
        J1 = np.array([[1,  0, - si * xij - ci * yij],
                       [0,  1,  ci * xij - si * yij],
                       [0,  0,      1]])
        J2 = np.array([[ci, - si,   0],
                       [si,  ci,    0],
                       [0,  0,     1]])
        return J1, J2

    def compute_shortest_path(self, source, finish):
        """
        Computes the shortest path in terms of uncertainty. The total distance from source to finish is
        """
        # initialize Sigma[source] as null
        self.Sigmas[source] = np.zeros((3, 3))
        # init the dists array with max values
        self.dists = np.array([sys.maxsize] * self.nnodes, dtype=float)
        # Initialize the source with the min value now
        self.dists[source] = 0
        # while all nodes are not visited
        for i in range(self.nnodes):
            # select the element of Q with the min distance
            v = self.mindistance()
            # mark v as visited
            self.S[v] = True
            if v == finish:
                break
            # for all neighbours, update the distances.
            # the uncertainty matrix at each node (self.Sigmas) is also updated to allow for error propagation.
            self.update_distances(v)
            self.print()
        self.store_shortest_path(source=source, finish=finish)
        return self.shortest_path, self.Sigmas[finish]



class DijkstraProjectionRelative():
    """
    Dijkstra projection storing relative transformations and relative uncertainties
    """
    def __init__(self, Sigmaij):
        """
        Currently, all needed initializations are performed on the computed shortest path method.
        """
        # number of nodes in the graph (equals the number of poses estimated so far)
        self.nnodes = 0
        # covariance noise between every two nodes
        self.Sigmaij = Sigmaij
        # self.poses = []
        self.Sigmas = [] # [None]*self.nnodes
        self.graph = np.zeros((self.nnodes, self.nnodes))
        # self.weights = np.zeros((nnodes, nnodes))
        # In this algorithm, the distance is based on the determinant of the uncertainty between nodes.
        self.dists = [] # np.array([sys.maxsize] * self.nnodes, dtype=float)
        # list of visited/unvisited nodes. All nodes marked as false (not visited)
        self.S = [] # [False] * self.nnodes
        # the shortest paths are stored as a parent array
        self.shortest_paths = [] # np.array([-1] * self.nnodes)
        self.shortest_path = []
        # self.init_poses(current_estimate, observations=observations)
        # self.rel_transforms = {}
        self.poses = []

    # def init_poses(self, current_estimate, observations):
    #     """
    #     Initializes a Ti matrix for every node in the graph.
    #     """
    #     # Sij i known for every pair of nodes
    #     # Converting from SE3 to SE2 for uncertainty propagation
    #     i = 0
    #     while current_estimate.exists(i):
    #         pose3 = current_estimate.atPose3(i)
    #         x = pose3.translation()[0]
    #         y = pose3.translation()[1]
    #         th = pose3.rotation().ypr()[0]
    #         pose2 = HomogeneousMatrix([x, y, 0], Euler([0, 0, th]))
    #         self.poses.append(pose2)
    #         i += 1
    #     # now create the graph. First connect adjacent nodes. then use observations to connect relative nodes.
    #     for i in range(self.nnodes-1):
    #         self.connect_nodes(i, i+1)
    #     for observation in observations:
    #         self.connect_nodes(observation[0], observation[1])

    def add_node(self):
        """
        Adds a new node to the graph.
        Adds a row and column to the graph matrix.
        """
        self.graph = np.pad(self.graph, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        self.nnodes = self.nnodes + 1
        self.poses.append(np.array([0, 0, 0]))
        self.Sigmas.append(np.diag([0, 0, 0]))

    def connect_nodes(self, i, j):
        """
        Asumes different costs when going from i to j and viceversa
        grows the graph
        Store the relative transformation.
        """
        self.graph[i, j] = True
        self.graph[j, i] = True
        # self.add_rel_transforms(i, j, iTj)

    # def add_rel_transforms(self, i, j, iTj):
    #     """
    #     Adds the transformation between node i and node j and viceversa.
    #     """
    #     key = str(i) + ',' + str(j)
    #     self.rel_transforms[key] = iTj
    #     key = str(j) + ',' + str(i)
    #     self.rel_transforms[key] = iTj.inv()

    # def get_rel_transform(self, i, j):
    #     """
    #     Return the transformation between the node i and node j
    #     """
    #     key = str(j) + ',' + str(i)
    #     try:
    #         return self.rel_transforms[key]
    #     except KeyError:
    #         return None

    def is_connected(self, i, j):
        if self.graph[i, j]:
            return True
        else:
            return False

    def print(self):
        print('Graph:')
        print(self.graph)
        print('Distance from source: ')
        for i in range(self.nnodes):
            print('Node ', i, ': ', self.dists[i])
        print('Visited nodes: ')
        for i in range(self.nnodes):
            print('Node ', i, ': ', self.S[i])
        print('Shouts: (yes, the algorithm is trying to grow along shouts based on min distance)')
        for i in range(self.nnodes):
            if self.dists[i] < sys.maxsize and not self.S[i]:
                print('Node ', i, ' min dist: ', self.dists[i])
        print('Parent array of shortest paths: ')
        print(self.shortest_paths)

    def set_poses(self, current_estimate):
        i = 0
        while current_estimate.exists(i):
            pose3 = current_estimate.atPose3(i)
            x = pose3.translation()[0]
            y = pose3.translation()[1]
            th = pose3.rotation().ypr()[0]
            self.poses[i] = np.array([x, y, th])
            i += 1


    def store_shortest_path(self, source, finish):
        """
        Follows the parent array backwards from finish to source
        """
        backwards = [finish]
        next_node = finish
        while next_node != source:
            next_node = self.shortest_paths[next_node]
            backwards.append(next_node)
        backwards.reverse()
        self.shortest_path = np.array(backwards)

    def mindistance(self):
        """
        Selects the minimum distance for the nodes that are not included in S.
        That is, look for the minimum distance so far for the nodes that have not been visited yet.
        """
        min_value = sys.maxsize
        min_index = 0
        for v in range(self.nnodes):
            if self.dists[v] < min_value and not self.S[v]:
                min_value = self.dists[v]
                min_index = v
        return min_index

    def update_distances(self, v):
        """
        Finds nodes adjacent to v and updates the distance in a greedy manner.
        If the distance found is less than the previous distance, the distance is saved.
        """
        # find adjacent nodes of v
        for u in range(self.nnodes):
            if not self.is_connected(v, u):
                continue
            # if connected, propagate uncertainty, compute determinant and sum to total distance at node u
            Sigmau, Tij = self.propagate_uncertainty(v, u)
            # caution, in a vanilla Dijkstra's algorithm, we may find here an addition,
            # however, the sum of distances (the propagation via Jacobians) is computed in self.propagate_uncertainty
            # thus, the total distance from node v to node u is actually the determinant of the covariance matrix Sigma_uv
            # temp_dist = self.dists[v] + weight
            temp_dist = np.linalg.det(Sigmau)
            # if a shorter path has been found, store it
            if temp_dist < self.dists[u]:
                self.dists[u] = temp_dist
                # update sigma for the new node!
                self.Sigmas[u] = Sigmau
                # update the parent array
                self.shortest_paths[u] = v

    def propagate_uncertainty(self, u, v):
        """
        Computes a Gaussian linear propagation law based on the uncertainty on node u and the relative uncertainty
        between u and v
        """
        ti = self.poses[u]
        tj = self.poses[v]
        Ti = HomogeneousMatrix(np.array([ti[0], ti[1], 0]), Euler([0, 0, ti[2]]))
        Tj = HomogeneousMatrix(np.array([tj[0], tj[1], 0]), Euler([0, 0, tj[2]]))
        # Tij = self.get_rel_transform(u, v)
        Tij = Ti.inv()*Tj
        tij = Tij.t2v()
        [J1, J2] = self.compute_jacobians(ti, tij)
        Si = self.Sigmas[u]
        Sij = self.Sigmaij
        Sj = np.dot(J1, np.dot(Si, J1.T)) + np.dot(J2, np.dot(Sij, J2.T))
        return Sj, Tij

    def compute_jacobians(self, ti, tij):
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

    def compute_relative_transformation_path(self, source, finish):
        ta = self.poses[source]
        tb = self.poses[finish]
        Ta = HomogeneousMatrix(np.array([ta[0], ta[1], 0]), Euler([0, 0, ta[2]]))
        Tb = HomogeneousMatrix(np.array([tb[0], tb[1], 0]), Euler([0, 0, tb[2]]))
        Tab = Ta.inv()*Tb
        return Tab

    def compute_shortest_path(self, source, finish):
        """
        Computes the shortest path in terms of uncertainty. The total distance from source to finish is
        """
        # initialize Sigma[source] as null
        self.Sigmas[source] = np.zeros((3, 3))
        # init the dists array with max values
        self.dists = np.array([sys.maxsize] * self.nnodes, dtype=float)
        # Initialize the source with the min value now
        self.dists[source] = 0
        self.S = [False] * self.nnodes
        self.shortest_paths = np.array([-1] * self.nnodes)
        self.shortest_path = []

        # while all nodes are not visited
        for i in range(self.nnodes):
            # select the element of Q with the min distance
            v = self.mindistance()
            # mark v as visited
            self.S[v] = True
            if v == finish:
                break
            # for all neighbours, update the distances.
            # the uncertainty matrix at each node (self.Sigmas) is also updated to allow for error propagation.
            self.update_distances(v)
            self.print()
        # stores the shortest path in self.shortest_path as node indexes
        self.store_shortest_path(source=source, finish=finish)
        # source, finish transformation
        Tab = self.compute_relative_transformation_path(source=source, finish=finish)
        return self.shortest_path, Tab, self.Sigmas[finish]



