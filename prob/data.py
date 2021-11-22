#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Dataset
"""

import os
import pickle

import numpy as np
from scipy.spatial import distance
import networkx as nx
from torch.utils.data import Dataset
import torch
import dgl
from tqdm import tqdm

CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

def make_instance(args):
    depot, loc, demand, capacity = args
    instance = {
        "loc": torch.tensor(loc, dtype=torch.float),
        "demand": torch.tensor(demand, dtype=torch.float) / capacity,
        "depot": torch.tensor(depot, dtype=torch.float)
    }
    return instance

class VRPDataset(Dataset):
    """
    This class is VRP Dataset

    Args:
        size (int): number of nodes in graph
        mode (str): dataset mode
        num_samples (int): number of instances
        seed (int): random seed
    """

    def __init__(self, size=50, mode="test", num_samples=1000000, seed=1234):
        # check mode
        assert mode in ["train", "val", "test"], "Invalid dataset mode."
        # init Dataset
        super(VRPDataset, self).__init__()
        # get path
        path = "./data/vrp/vrp{}_{}_seed{}.pkl".format(size, mode, seed)
        # load data
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[:num_samples]]
        # generate data
        else:
            # set random seed
            torch.manual_seed(seed)
            # capacities
            capacity = CAPACITIES[size]
            # data
            self.data = [
                {
                # customer location
                "loc": torch.FloatTensor(size, 2).uniform_(0, 1),
                # customer demand
                "demand": torch.FloatTensor(size).uniform_(1, 10).int().float() / capacity,
                # depot
                "depot": torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]
        # build graph
        for i in tqdm(range(num_samples)):
            g = self._buildGraph(i)
            self.data[i]["graph"] = g


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    def _buildGraph(self, i):
        """
        A method to build dgl graph with distance
        """
        # get distance
        dist = self._getDist(i)
        # build nx graph
        nx_graph = nx.from_numpy_matrix(dist, create_using=nx.DiGraph)
        weights = np.array([nx_graph.edges[e]["weight"] for e in nx_graph.edges],
                           dtype=np.float32)
        # build dgl graph
        g = dgl.from_networkx(nx_graph, idtype=torch.int32)
        g.ndata["x"] = torch.zeros(g.num_nodes(), 1)
        g.edata["w"] = torch.from_numpy(weights)
        return g


    def _getDist(self, i):
        """
        A method to calculate distance matrix
        """
        # get coordinates
        depot = self.data[i]["depot"].detach().numpy().reshape((1,2))
        loc = self.data[i]["loc"].detach().numpy()
        loc = np.concatenate((depot, loc), axis=0)
        # calculate distance
        dist = distance.cdist(loc, loc, "euclidean")
        return dist
