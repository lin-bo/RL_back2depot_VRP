#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin
"""
Naive return
"""


import torch
from attention_model import load_routing_agent


class naiveReturn:

    def __init__(self, size=20, thre=0.0):

        """
        args:
            size: the number of customers
            thre: the threshould that parameterize the algorithm
        """

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.rou_agent = load_routing_agent(size=size)
        self.horizon = size * 2
        self.thre = thre
        self._size = size

        # create one hot vectors
        self._one_hot = torch.zeros((self._size + 1, self._size + 1))
        self._one_hot.scatter_(0, torch.arange(0, self._size + 1).reshape((1, -1)), 1).to(self.device)

    def _set_params(self, batch_data):

        self._batch = len(batch_data["loc"])
        self._demand = batch_data["demand"]
        self._loc = torch.cat((batch_data["depot"].reshape(-1, 1, 2), batch_data["loc"]), axis=1)

        self.v = torch.zeros(self._batch, dtype=torch.int32, device=self.device)
        self.c = torch.ones(self._batch, dtype=torch.float32, device=self.device)
        self.o = torch.zeros((self._batch, self._size + 1), dtype=torch.float32, device=self.device)

        self.routes = self.v.clone().reshape((-1, 1))
        self.dist = torch.zeros(self._batch, dtype=torch.float32, device=self.device)
        self.prev_v = self.v.clone()

    def solve(self, batch_data):
        """
        args:
            batch_data
        return:
            routes: (batch, horizon) tensor
            dist: (batch, ) tensor
        """
        # set problem specific parameters
        self._set_params(batch_data)

        # initialize the state of routing agent
        state = self.rou_agent.re_init(batch_data["loc"])

        # sequentially generate the solutions
        for t in range(self.horizon):
            action = self._get_action()
            next_nodes = self._routing_decision(state)
            self._update_return_state(next_nodes, action)
            state = state.new_update(next_nodes, action)
            self.dist += self._step_dist()

        return self.routes, self.dist

    def _update_return_state(self, next_nodes, action):
        """
        Update returning state
        """
        self.prev_v = self.v.clone()
        self.v = ((next_nodes + 1) * (1 - action))
        satisfied = self._demand.gather(axis=-1, index=next_nodes.reshape((-1, 1)))[:, 0].to(self.device)
        self.c = 1 * action + (self.c - satisfied) * (1 - action)
        self.o += self._one_hot[next_nodes + 1] * (1 - action.reshape((-1, 1)))
        self.o = torch.minimum(self.o, torch.tensor(1))

        self.routes = torch.cat((self.routes, self.v.reshape((-1, 1))), axis=-1)

    def _get_action(self):

        low_cap = (self.c <= self.thre).to(torch.int32).to(self.device)
        all_served = (self.o.sum(axis=1) >= self._size).to(torch.int32).to(self.device)
        cant_serve = ((self._demand > self.c.reshape((-1, 1))).sum(axis=1)
                      + self.o.sum(axis=1)
                      == self._size).to(torch.int32).to(self.device)

        return torch.minimum(low_cap + all_served + cant_serve, torch.tensor(1, device=self.device))

    def _routing_decision(self, rou_state):
        """
        make routing decisions based on the given routing agent
        return:
            (bacth_size, ) tensor representing the next nodes to visit for each instance,
            note that if all the customers have already been served, the returned node will always be 1
        """
        # make routing decision
        log_p, mask = self.rou_agent._get_log_p(self.rou_agent.fixed, rou_state)
        prob = log_p.exp()
        # check if the demand at each node exceeds the remaining capacity or not, if so, should be masked

        flag_demand = self._demand > self.c.reshape((self._batch, 1))
        mask = torch.minimum(mask + flag_demand.reshape((self._batch, 1, -1)), torch.tensor(1, device=self.device))
        # normalize the probability
        prob *= ~mask
        prob /= prob.sum(axis=-1, keepdim=True)
        # decode the next node to visit (based on the routing agent)
        next_nodes = self.rou_agent._select_node(prob[:, 0, :], mask[:, 0, :])

        return next_nodes.to(self.device)

    def _step_dist(self):
        """
        calculate one-step reward
        return:
            (batch, ) tensor (negative value)
        """

        # get locations
        idx = torch.cat((self.prev_v.reshape(-1, 1, 1), self.prev_v.reshape(-1, 1, 1)), axis=-1).to(torch.int64)
        idx = idx.to(self.device)
        prev_loc = self._loc.gather(axis=1, index=idx)[:, 0, :]
        idx = torch.cat((self.v.reshape(-1, 1, 1), self.v.reshape(-1, 1, 1)), axis=-1).to(torch.int64)
        idx = idx.to(self.device)
        curr_loc = self._loc.gather(axis=1, index=idx)[:, 0, :]

        return (prev_loc - curr_loc).norm(dim=-1)
