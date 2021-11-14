#!/usr/bin/env python
# coding: utf-8
"""
Randomize sweep heuristic algorithm
https://www.jstor.org/stable/3007888
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.spatial import distance
import gurobipy as gp
from gurobipy import GRB

from solver.absolver import ABSolver


class sweepHeuristic(ABSolver):
    """
    This is a class for sweep heuristic solver

    Args:
        depot (array): coordinate of central depot
        loc (array): coordinates of customers
        demand (array): demands of customers
    """

    def solve(self, rand_iter=5):
        """
        A method to solve model

        Args:
            rand_iter (int): number of Randomization

        Returns:
            tuple: best route (list[list]), objective value of best route (float)
        """
        # compute angle
        rel_loc = self.loc - self.depot
        rad = np.arctan2(rel_loc[:,1], rel_loc[:,0])
        deg = (np.rad2deg(rad).astype(int) + 180) % 360 - 180
        #assert np.abs(deg - np.rad2deg(rad).astype(np.int8)).sum() == 0
        # random initial angles
        best_routes = None
        best_obj = float("inf")
        for _ in range(rand_iter):
            # init sets
            nodes = set(range(self.size))
            clusters, cluster = [], []
            # left capacity
            left = 1
            # select a random angle
            angle =  np.random.randint(-180, 180)
            # get clusters
            while nodes:
                # increase angle
                v = self._get_node_from_ange(angle, deg, nodes)
                # go back depot
                if self.demand[v] > left:
                    clusters.append(cluster)
                    # new route cluster
                    cluster = []
                    left = 1
                # add to route
                left -= self.demand[v]
                nodes.remove(v)
                cluster.append(v)
            # add last cluster
            clusters.append(cluster)
            # solve TSP for each cluster
            routes = []
            obj = 0
            for cluster in clusters:
                tour, tspobj = self._solveTSP(cluster)
                routes.append(tour)
                obj += tspobj
            if obj <= best_obj:
                best_routes, best_obj = routes, obj
        return best_routes, best_obj


    @staticmethod
    def _get_node_from_ange(angle, deg, nodes):
        """
        A method to increase angle until it equal to some node
        """
        while True:
            # check euqality
            v_cand = set(np.where(deg == angle)[0]).intersection(nodes)
            if v_cand:
                return list(v_cand)[0]
            angle = ((angle + 1) + 180) % 360 - 180


    def _solveTSP(self, cluster):
        """
        A method to find the optimal TSP tour
        """
        # get coordinates
        loc = self._getLoc(cluster)
        # distance matrix
        dist = distance.cdist(loc, loc, "euclidean")
        # only one customer
        if len(cluster) == 1:
            return cluster, dist[0,0]
        # build model
        tsp_model, x = self._buildTSPModel(dist)
        # solve
        tsp_model.optimize(self._subtourelim)
        # get tour
        tsp_tour = self.getTour(x)
        tour = []
        for i in tsp_tour[1:-1]:
            tour.append(cluster[i-1])
        return tour, tsp_model.objVal


    def _getLoc(self, cluster):
        """
        A method to get coordinates of nodes in cluster and depot
        """
        loc = self.depot.reshape((1,2))
        for i in cluster:
            loc = np.concatenate((loc, self.loc[i:i+1]), axis=0)
        return loc


    @staticmethod
    def _buildTSPModel(dist):
        """
        A method to build TSP MIP model
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        size = dist.shape[0]
        edges = [(i, j) for i in range(size) for j in range(size) if i < j]
        x = m.addVars(edges, name="x", vtype=GRB.BINARY)
        for i, j in edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # objective function
        obj = gp.quicksum(dist[i,j] * x[i,j] for i, j in edges)
        m.setObjective(obj)
        # constraints
        m.addConstrs(x.sum(i, "*") == 2 for i in range(size))  # 2 degree
        # activate lazy constraints
        m._x = x
        m._n = size
        m.Params.lazyConstraints = 1
        # update
        m.update()
        return m, x


    @staticmethod
    def _subtourelim(model, where):
        """
        A Gurobi callback function to add lazy constraints for subtour elimination
        """
        def subtour(selected, n):
            """
            find shortest cycle
            """
            unvisited = list(range(n))
            # init dummy longest cycle
            cycle = range(n + 1)
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [
                        j for i, j in selected.select(current, "*")
                        if j in unvisited
                    ]
                if len(cycle) > len(thiscycle):
                    cycle = thiscycle
            return cycle

        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x.keys() if xvals[i, j] > 1e-2)
            # shortest cycle
            tour = subtour(selected, model._n)
            # add cuts
            if len(tour) < model._n:
                model.cbLazy(
                    gp.quicksum(
                        model._x[i, j]
                        for i, j in combinations(tour, 2)) <= len(tour) - 1)


    @staticmethod
    def getTour(x):
        """
        A method to get a tour from solution
        """
        # active edges
        edges = defaultdict(list)
        for i, j in x:
            if x[i,j].x > 1e-2:
                edges[i].append(j)
        # get tour
        visited = {0}
        tour = [0]
        while len(visited) < len(edges):
            i = tour[-1]
            for j in edges[i]:
                if j not in visited:
                    tour.append(j)
                    visited.add(j)
                    break
        if 0 in edges[tour[-1]]:
            tour.append(0)
        return tour
