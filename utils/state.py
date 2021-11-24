#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang & Bo Lin
"""
Routing state in reinforcement learning
"""

import torch

class returnState:
    """
    This class is enviroment state

    Args:
        size (int): dimension of node feature
        w_feats (int): dimension of edge weight
    """

    def __init__(self, batch_data):
        self._batch = len(batch_data["loc"])
        self._size = len(batch_data["demand"][0])
        self._demand = batch_data["demand"]

        # create one hot vectors
        self._one_hot = torch.zeros((self._size + 1, self._size + 1))
        self._one_hot.scatter_(0, torch.arange(0, self._size + 1).reshape((1, -1)), 1)

        self.v = torch.zeros(self._batch, dtype=torch.float32)
        self.c = torch.ones(self._batch, dtype=torch.float32)
        self.o = torch.zeros((self._batch, self._size+1), dtype=torch.float32)

    def update(self, action, rou_agent, rou_state):
        """
        A fuctiion to update state after action
        """
        # make routing decision
        next_nodes = self._routing_decision(rou_agent, rou_state)

        # update return agent state
        self._update_return_state(next_nodes, action)

        # update routing agent state
        rou_state = rou_state.new_update(next_nodes, action)

        return rou_state

    def _routing_decision(self, rou_agent, rou_state):
        """
        make routing decisions based on the given routing agent
        return:
            (bacth_size, ) tensor representing the next nodes to visit for each instance,
            note that if all the customers have already been served, the returned node will always be 1
        """
        # make routing decision
        log_p, mask = rou_agent._get_log_p(rou_agent.fixed, rou_state)
        prob = log_p.exp()
        # check if the demand at each node exceeds the remaining capacity or not, if so, should be masked
        flag_demand = self._demand > self.c.reshape((self._batch, 1))
        mask *= flag_demand.reshape((self._batch, 1, -1))
        # normalize the probability
        prob *= ~mask
        prob /= prob.sum(axis=-1, keepdim=True)
        # decode the next node to visit (based on the routing agent)
        next_nodes = rou_agent._select_node(prob[:, 0, :], mask[:, 0, :])

        return next_nodes

    def _update_return_state(self, next_nodes, action):
        """
        Update returning state
        """
        self.v = (next_nodes + 1) * (1 - action)
        satisfied = self._demand.gather(axis=-1, index=next_nodes.reshape((-1, 1)))[:, 0]
        self.c = 1 * action + (self.c - satisfied) * (1 - action)
        self.o += self._one_hot[next_nodes + 1] * (1 - action.reshape((-1, 1)))
        self.o = torch.minimum(self.o, torch.tensor(1))
