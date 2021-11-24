#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Routing state in reinforcement learning
"""

import torch

class routingState:
    """
    This class is enviroment state

    Args:
        size (int): dimension of node feature
        w_feats (int): dimension of edge weight
    """

    def __init__(self, batch_data):
        self._batch = len(batch_data["loc"])
        self._size = len(batch_data["demand"][0])
        self.v = torch.zeros(self._batch, dtype=torch.float32)
        self.c = torch.ones(self._batch, dtype=torch.float32)
        self.o = torch.zeros((self._batch, self._size+1), dtype=torch.float32)

    def update(self, action, state, rou_agent, rou_state):
        """
        A fuctiion to update state after action
        """
        # make routing decision
        log_p, mask = rou_agent._get_log_p(rou_agent.fixed, state)
