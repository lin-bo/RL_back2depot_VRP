import numpy as np
from prob import VRPDGLDataset
from dgl.dataloading import GraphDataLoader
import torch
from attention_model.attention_utils.functions import load_routing_agent
from solver.absolver import ABSolver


class amVRP:

    def __init__(self, size=20, method='greedy'):
        """
        args:
            size: the number of customers
        """
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.solver = load_routing_agent(size=size, name='vrp')
        self.horizon = size * 2
        self._size = size
        assert method in ['greedy', 'sampling']
        self.method = method

    def solve(self, batch_data):
        if self.method == 'greedy':
            batch_rep, iter_rep = 1, 1
        else:
            batch_rep, iter_rep = 50, 1
        routes, costs = self.solver.sample_many(batch_data, batch_rep=batch_rep, iter_rep=iter_rep)
        routes = self._covertRoutes(routes)
        return routes, costs.detach().cpu().tolist()

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
            if len(tour_list) != 0:
                routes_list.append(tour_list)
            batch_routes_list.append(routes_list)
        return batch_routes_list
