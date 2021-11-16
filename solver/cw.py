#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin
"""
Randomized Clarke-Wright Savings Heuristic algorithm
https://www.jstor.org/stable/167703
"""

import numpy as np

from solver.absolver import ABSolver


class cwHeuristic(ABSolver):
    """
    This is an class for Clarke-Wright heuristic solver

    Args:
        depot (array): coordinate of central depot
        loc (array): coordinates of customers
        demand (array): demands of customers
    """

    def solve(self, rand_depth=5, rand_iter=5):
        """
        A method to solve model

        Args:
            rand_depth (int): the number of merger we consider each iteration
            rand_iter (int): the number of times we repeat for each r

        Returns:
            tuple: best route (list[list]), objective value of best route (float)
        """
        # calculate distances
        d2c, c2c = self._calDistance()
        # calculate the saving matrix
        S = d2c.reshape((-1, 1)) + d2c.reshape((1, -1)) - c2c
        # set best sol recorders
        best_obj = 1e5
        best_routes = []
        # lets solve it
        for r in range(1, rand_depth + 1):
            for m in range(rand_iter):
                routes = self._routesInit()
                while True:
                    MS = self._calSaving(S, routes)
                    if (MS > 0).sum() == 0:
                        break
                    mergers = self._topMergers(MS, r)
                    routes = self._merge(routes.copy(), mergers)
                obj = self._calObj(routes, d2c, c2c)
                if obj < best_obj:
                    best_obj = obj
                    best_routes = routes.copy()
        return best_routes, best_obj


    def _calDistance(self):
        """
        A method to calculate the d2c distances and c2c distances, organized as arrays
        """
        # calculate depot-to-custoer distances
        rel_loc = self.loc - self.depot
        d2c = np.sqrt((rel_loc ** 2).sum(axis=1))
        # calculate customer-to-customer distances
        rel_pos = self.loc.reshape((-1, 1, 2)) - self.loc.reshape((1, -1, 2))
        c2c = np.sqrt((rel_pos ** 2).sum(axis=-1))
        return d2c, c2c


    def _routesInit(self):
        """
        A method to initialize the solution (each customer is placed in a different route)
        """
        return [[i] for i in range(self.loc.shape[0])]


    def _calSaving(self, S, routes):
        """
        A method to calculate the pair-wise merge saving for the given routes
        """
        n = len(routes)
        MS = np.zeros((n, n))
        d = [self.demand[route].sum() for route in routes]
        for o, r1 in enumerate(routes):
            for s, r2 in enumerate(routes):
                if o == s:
                    continue
                MS[o, s] = S[r1[-1], r2[0]] if d[o] + d[s] <= 1 else -100
        return MS


    def _topMergers(self, M, r):
        """
        A method to find top mergers
        """
        M_flat = M.flatten()
        n = M.shape[0]
        r = np.min([n**2 - 1, r])
        indices = np.argpartition(M_flat, n ** 2 - r)[-r:]
        indices = [idx for idx in indices if M_flat[idx] > 0]
        return np.array([[int(idx//n), int(idx % n)] for idx in indices])


    def _merge(self, routes, mergers):
        """
        A method to combine the associated routes
        """
        idx = np.random.randint(mergers.shape[0])
        i, j = mergers[idx]
        r1 = routes.pop(i)
        r2 = routes.pop(j) if j < i else routes.pop(j-1)
        routes.append(r1 + r2)
        return routes


    def _calObj(self, routes, d2c, c2c):
        """
        A method to calculate objective value
        """
        obj = 0
        for route in routes:
            obj += d2c[route[0]]
            obj += d2c[route[-1]]
            for idx in range(len(route) - 1):
                obj += c2c[route[idx], route[idx + 1]]
        return obj
