import numpy as np

from solver.absolver import ABSolver


class cwHeuristic(ABSolver):

    '''
    This is an class for CW heuristic solver

    Args:
        depot (array): coordinate of central depot
        loc (array): coordinates of customers
        demand (array): demands of customers
    '''

    def _cal_distance(self):
        '''
        calculate the d2c distances and c2c distances, organized as arrays
        :return: d2c [n], c2c [nxn]
        '''

        # calculate depot-to-custoer distances
        rel_loc = self.loc - self.depot
        d2c = np.sqrt((rel_loc ** 2).sum(axis=1))

        # calculate customer-to-customer distances
        rel_pos = self.loc.reshape((-1, 1, 2)) - self.loc.reshape((1, -1, 2))
        c2c = np.sqrt((rel_pos ** 2).sum(axis=-1))

        return d2c, c2c

    def _routes_init(self):
        '''
        initialize the solution (each customer is placed in a different route)
        :return: a list n route
        '''

        return [[i] for i in range(self.loc.shape[0])]

    def _cal_saving(self, S, routes):
        '''
        calculate the pair-wise merge saving for the given routes
        :param S: the saving matrix
        :param routes: a list of routes
        :return: a saving matrix (array, len(routes) x len(routes))
        '''

        n = len(routes)
        MS = np.zeros((n, n))
        d = [self.demand[route].sum() for route in routes]

        for o, r1 in enumerate(routes):
            for s, r2 in enumerate(routes):
                if o == s:
                    continue
                MS[o, s] = S[r1[-1], r2[0]] if d[o] + d[s] <= 1 else -100

        return MS

    def _top_mergers(self, M, r):

        M_flat = M.flatten()
        n = M.shape[0]
        indices = np.argpartition(M_flat, n ** 2 - r)[-r:]
        indices = [idx for idx in indices if M_flat[idx] > 0]

        return np.array([[int(idx//n), int(idx % n)] for idx in indices])

    def _merge(self, routes, mergers):

        idx = np.random.randint(mergers.shape[0])
        i, j = mergers[idx]

        r1 = routes.pop(i)
        r2 = routes.pop(j) if j < i else routes.pop(j-1)

        routes.append(r1 + r2)

        return routes

    def _cal_obj(self, routes, d2c, c2c):

        obj = 0
        for route in routes:
            obj += d2c[route[0]]
            obj += d2c[route[-1]]

            for idx in range(len(route) - 1):
                obj += c2c[route[idx], route[idx + 1]]

        return obj

    def solve(self, R=5, M=5):

        '''
        :param R (int): the number of merger we consider each iteration
        :param M (int): the number of times we repeat for each r
        :return: best route, best objective value
        '''

        # calculate distances
        d2c, c2c = self._cal_distance()

        # calculate the saving matrix
        S = d2c.reshape((-1, 1)) + d2c.reshape((1, -1)) - c2c

        # set best sol recorders
        best_obj = 1e5
        best_routes = []

        # lets solve it
        for r in range(1, R + 1):
            for m in range(M):
                routes = self._routes_init()
                while True:
                    MS = self._cal_saving(S, routes)
                    if (MS > 0).sum() == 0:
                        break
                    mergers = self._top_mergers(MS, r)
                    routes = self._merge(routes.copy(), mergers)

                obj = self._cal_obj(routes, d2c, c2c)
                if obj < best_obj:
                    best_obj = obj
                    best_routes = routes.copy()

        for route in routes:
            print(self.demand[route].sum())

        return best_routes, best_obj
