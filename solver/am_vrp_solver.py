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
        self.method = method
        assert method not in ['greedy', 'sampling']

    def solve(self, batch_data):

        if self.method == 'greedy':
            batch_rep, iter_rep = 1, 1
        else:
            batch_rep, iter_rep = 50, 1

        routes, costs = self.solver.sample_many(batch_data, batch_rep=batch_rep, iter_rep=iter_rep)

        return routes, costs