import numpy as np
from prob import VRPDGLDataset
from dgl.dataloading import GraphDataLoader
import torch
from attention_model.attention_utils.functions import load_routing_agent
from solver.absolver import ABSolver


class amVRP:

    def __init__(self, size=20):
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

    def solve(self, batch_data):

        routes, costs = self.solver.sample_many(batch_data, batch_rep=1, iter_rep=1)

        return routes, costs