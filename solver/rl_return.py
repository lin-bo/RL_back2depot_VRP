#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Structure2Vec + RL for Return2Depot
"""

import numpy as np
from scipy.spatial import distance
import torch

from solver.absolver import ABSolver
from attention_model import load_routing_agent
from utils import returnState, returnAgent, calObj

class return2Depot(ABSolver):
    """
    This is a class for using hierachy RL

    Args:
        size (int): graph size
        rou_agent_type (str): type of routing agent
    """

    def __init__(self, size, rou_agent_type):
        self.size = size
        self.rou_agent_type = rou_agent_type
        # cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        # load routing agent
        print("\nLoading routing agent...")
        self.rou_agent = load_routing_agent(size=self.size, name=self.rou_agent_type)
        # initialize return agent
        print("\nLoading return2depot agent...")
        self.re_agent = returnAgent(gnn_x_feat=2, gnn_w_feats=1, gnn_e_feats=64)
        self.re_agent.loadModel("./pretrained/vrp-{}.pkl".format(size))

    def solve(self, data, graph):
        # init return state
        rou_state = self.rou_agent.re_init(data)
        # init state
        re_state = returnState(data, graph, self.rou_agent_type)
        # init route
        batch_routes = re_state.v
        # eval
        for t in range(self.size*2):
            # take action
            action = self.re_agent.actionDecode(re_state)
            # update state
            re_state, rou_state = re_state.update(action, self.rou_agent, rou_state, data)
            batch_routes = torch.cat((batch_routes, re_state.v), axis=1)
        batch_routes = self._covertRoutes(batch_routes)
        batch_objs = self._calObjs(batch_routes, data)
        return batch_routes, batch_objs

    def _covertRoutes(self, batch_routes):
        batch_routes = batch_routes.cpu().detach().numpy() - 1
        batch_routes_list = []
        for routes in batch_routes:
            routes_list = []
            tour_list = []
            for i in routes:
                if i == -1 and len(tour_list) != 0:
                    routes_list.append(tour_list)
                    tour_list = []
                if i != -1:
                    tour_list.append(i)
            batch_routes_list.append(routes_list)
        return batch_routes_list

    def _calObjs(self, batch_routes, data):
        batch_dist = self._calDist(data)
        batch_objs = [calObj(routes, dist) for routes, dist in zip(batch_routes, batch_dist)]
        return batch_objs

    def _calDist(self, data):
        # info
        depots = data["depot"].cpu().detach().numpy().reshape((-1,1,2))
        locs = data["loc"].cpu().detach().numpy()
        # calculate
        batch_dist = []
        for i in range(depots.shape[0]):
            coor = np.concatenate((depots[i], locs[i]), axis=0)
            dist = distance.cdist(coor, coor, "euclidean")
            batch_dist.append(dist)
        return batch_dist
