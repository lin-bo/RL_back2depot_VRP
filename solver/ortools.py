#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Google OR-Tools
https://developers.google.com/optimization
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial import distance

from solver.absolver import ABSolver
import utils

class googleOR(ABSolver):
    """
    This is a class for using Google OR-Tool

    Args:
        depot (array): coordinate of central depot
        loc (array): coordinates of customers
        demand (array): demands of customers
    """

    def solve(self, solution_limit=100):
        """
        A method to solve model
        """
        # init data
        data, dist = self.createData()

        # create the routing index manager
        manager = pywrapcp.RoutingIndexManager(data["num_nodes"],
                                               data["num_vehicles"],
                                               data["depot"])
        # create routing model
        routing = pywrapcp.RoutingModel(manager)

        # register distance callback for cost
        def distanceCallback(from_index, to_index):
            """
            A callback function to return the distance between the two nodes
            """
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance'][from_node][to_node]
        transit_callback_index = routing.RegisterTransitCallback(distanceCallback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # register demand callback for capacity
        def demandCallback(from_index):
            """
            A callback function to return the demand of the node
            """
            from_node = manager.IndexToNode(from_index)
            return data["demand"][from_node]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demandCallback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index,
                                                0,  # null capacity slack
                                                data["capacities"],  # vehicle maximum capacities
                                                True,  # start cumul to zero
                                                "Capacity")

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.FromSeconds(3)
        search_parameters.solution_limit = solution_limit

        # solve
        sol = routing.SolveWithParameters(search_parameters)
        routes, _ = self.getSolution(data, manager, routing, sol)

        # recalculate objective value
        obj = utils.calObj(routes, dist)

        return routes, obj


    def createData(self):
        """"
        A method to store the data for the problem
        """
        # get coordinates
        loc = self._getLoc()
        # distance matrix
        dist = distance.cdist(loc, loc, "euclidean")
        # build data
        data = {}
        data["distance"] = (dist * 1e4).astype(int)
        data["num_nodes"] = len(dist)
        data["num_vehicles"] = 10
        data["depot"] = 0
        data["demand"] = [0] + (self.demand * 1e3).astype(int).tolist()
        data["capacities"] = [1e3] * data["num_vehicles"]
        return data, dist


    def _getLoc(self):
        """
        A method to get coordinates of nodes in cluster and depot
        """
        loc = self.depot.reshape((1,2))
        loc = np.concatenate((loc, self.loc), axis=0)
        return loc


    @staticmethod
    def getSolution(data, manager, routing, solution):
        """
        A method to get solution
        """
        routes = []
        total_dist = 0
        for vid in range(data["num_vehicles"]):
            tour = []
            ind = routing.Start(vid)
            while not routing.IsEnd(ind):
                node_ind = manager.IndexToNode(ind)
                # not depot
                if node_ind != 0:
                    tour.append(node_ind-1)
                prev_ind, ind = ind, solution.Value(routing.NextVar(ind))
                # calculate distance
                total_dist += routing.GetArcCostForVehicle(prev_ind, ind, vid)
            if tour:
                routes.append(tour)
        return routes, total_dist
