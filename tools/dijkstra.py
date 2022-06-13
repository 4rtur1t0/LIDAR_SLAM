"""
A basic implementation of the Dijkstra's algorithm

Some sources:
https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
"""
import sys
import numpy as np


class Dijkstra():
    def __init__(self, nnodes):
        self.nnodes = nnodes
        self.graph = np.zeros((nnodes, nnodes))
        self.weights = np.zeros((nnodes, nnodes))
        self.dists = np.array([sys.maxsize] * self.nnodes, dtype=float)
        # list of visited/unvisited nodes. All nodes marked as false (not visited)
        self.S = [False] * self.nnodes
        # the shortest paths are stored as a parent array
        self.shortest_paths = np.array([-1] * self.nnodes)
        self.shortest_path = []

    def add_weights(self, i, j, weights):
        """
        Asumes different costs when going from i to j and viceversa
        """
        self.graph[i, j] = True
        self.graph[j, i] = True
        self.weights[i, j] = weights[0]
        self.weights[j, i] = weights[1]

    def get_weight(self, i, j):
        if self.graph[i, j] > 0:
            return self.weights[i, j]
        else:
            return None

    def print(self):
        print('Graph:')
        print(self.graph)
        print('Weights: ')
        print(self.weights)
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
        print('Shortest path from source to finish: ')
        print(self.shortest_path)

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
        Selects the minimum distance for the nodes that are not included in S
        """
        min_value = sys.maxsize
        min_index = 0
        for v in range(self.nnodes):
            if self.dists[v] < min_value and not self.S[v]:
                min_value = self.dists[v]
                min_index = v
        return min_index

    def update_distances(self, v):
        # find adjacent nodes of v
        for u in range(self.nnodes):
            weight = self.get_weight(v, u)
            if weight is None:
                continue
            temp_dist = self.dists[v] + weight
            # if a shorter path has been found, store it
            if temp_dist < self.dists[u]:
                self.dists[u] = temp_dist
                # update the parent array
                self.shortest_paths[u] = v

    def compute_shortest_path(self, source, finish):
        # init the dists array with max values
        self.dists = np.array([sys.maxsize] * self.nnodes)
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
            # for all neighbours, update the distances
            self.update_distances(v)
            self.print()
        self.store_shortest_path(source=source, finish=finish)
        return shortest_path


if __name__ == "__main__":
    # provide number of nodes
    dijkstra_algorithm = Dijkstra(nnodes=9)
    # contruct the graph by providing weights > 0 between nodes of the graph
    dijkstra_algorithm.add_weights(0, 1, weights=[4, 4])
    dijkstra_algorithm.add_weights(0, 7, weights=[8, 8])
    dijkstra_algorithm.add_weights(1, 2, weights=[8, 8])
    dijkstra_algorithm.add_weights(1, 7, weights=[1, 1])
    dijkstra_algorithm.add_weights(2, 3, weights=[7, 7])
    dijkstra_algorithm.add_weights(2, 8, weights=[2, 2])
    dijkstra_algorithm.add_weights(2, 5, weights=[4, 4])
    dijkstra_algorithm.add_weights(3, 4, weights=[9, 9])
    dijkstra_algorithm.add_weights(3, 5, weights=[14, 14])
    dijkstra_algorithm.add_weights(4, 5, weights=[10, 10])
    dijkstra_algorithm.add_weights(5, 6, weights=[2, 2])
    dijkstra_algorithm.add_weights(6, 7, weights=[1, 1])
    dijkstra_algorithm.add_weights(6, 8, weights=[6, 6])
    dijkstra_algorithm.add_weights(7, 8, weights=[7, 7])

    # Print the graph
    dijkstra_algorithm.print()
    # select source and finish nodes
    shortest_path = dijkstra_algorithm.compute_shortest_path(source=0, finish=7)
    # Print the result
    dijkstra_algorithm.print()


