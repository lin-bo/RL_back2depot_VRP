#!/usr/bin/env python
# coding: utf-8
"""
Dataset
"""

import os
import pickle

from torch.utils.data import Dataset
import torch

CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

def make_instance(args):
    depot, loc, demand, capacity = args
    instance = {
        'loc': torch.tensor(loc, dtype=torch.float),
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float)
    }
    return instance

class VRPDataset(Dataset):

    def __init__(self, size=50, mode="test", num_samples=1000000, seed=1234):
        # check mode
        assert mode in ["train", "val", "test"], "Invalid dataset mode."
        # init Dataset
        super(VRPDataset, self).__init__()
        # get path
        path = "./data/vrp/vrp{}_{}_seed{}.pkl".format(size, mode, seed)
        # load data
        if os.path.isfile(path):
            with open(path, 'rb') as f:
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
                'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                # customer demand
                'demand': torch.FloatTensor(size).uniform_(1, 10).int().float() / capacity,
                # depot
                'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
