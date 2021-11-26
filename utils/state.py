#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin & Bo Tang
"""
Routing state in reinforcement learning
"""

import copy

import torch
import dgl

class returnState:
    """
    This class is enviroment state

    Args:
        size (int): dimension of node feature
        w_feats (int): dimension of edge weight
    """

    def __init__(self, batch_data, batch_graph):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        # original data
        self._batch = len(batch_data["loc"])
        self._size = len(batch_data["demand"][0])
        # graph
        self.g = copy.deepcopy(batch_graph)
        # state
        self.v = torch.zeros((self._batch, 1), dtype=torch.int32, device=self.device)
        self.c = torch.ones((self._batch, 1), dtype=torch.float32, device=self.device)
        self.o = torch.zeros((self._batch, self._size+1), dtype=torch.float32, device=self.device)
        self.prev_v = self.v.clone()
        # reward
        self.r = torch.zeros((self._batch, 1), dtype=torch.float32, device=self.device)

    def update(self, action, rou_agent, rou_state, batch_data):
        """
        A method to update state after action
        return:
            rou_state: new state for the routing agent
            reward: (batch, ) tensor specifying the one-step reward for each instance
        """
        # map -1, 1 to 0, 1
        action_flag = ((action + 1) / 2).to(torch.int32).to(self.device)
        # get demand and loc info
        demand = batch_data["demand"]
        loc = torch.cat((batch_data["depot"].reshape(-1, 1, 2), batch_data["loc"]), axis=1)
        # make routing decision
        next_nodes = self._routing_decision(rou_agent, rou_state, demand)
        # update return agent state
        re_state = self._update_return_state(batch_data, next_nodes, action_flag, demand)
        # update routing agent state
        rou_state = rou_state.new_update(next_nodes.reshape((-1, )), action_flag.reshape((-1,)))
        # update reward
        re_state.r = self._cal_reward(loc)
        return re_state, rou_state

    def _update_return_state(self, batch_data, next_nodes, action, demand):
        """
        A method to update returning state

        Args:
            next_nodes: (batch, 1)
            action: (batch, 1)
        """
        # init new state
        new_state = returnState(batch_data, self.g)
        # update current location
        new_state.prev_v = self.v.clone()
        new_state.v = ((next_nodes + 1) * (1 - action)).to(torch.int32).detach()
        # update capacity
        satisfied = demand.gather(axis=-1, index=next_nodes).to(self.device)
        new_state.c = (1 * action + (self.c - satisfied) * (1 - action)).detach()
        # create one hot vectors
        one_hot = torch.zeros((self._size + 1, self._size + 1))
        one_hot = one_hot.scatter(0, torch.arange(0, self._size + 1).reshape(1, -1), 1).to(self.device)
        # update visit history
        new_state.o += one_hot[next_nodes + 1][:,0,:] * (1 - action)
        new_state.o = torch.minimum(new_state.o, torch.tensor(1, device=self.device)).detach()
        # update graph
        x = self.g.ndata["x"].detach().clone()
        new_state.g.ndata["x"] = x
        for i, g in enumerate(dgl.unbatch(new_state.g)):
            g.ndata["x"][:,0] = new_state.o[i]
        return new_state

    def _routing_decision(self, rou_agent, rou_state, demand):
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
        flag_demand = demand > self.c
        mask = torch.minimum(mask + flag_demand.reshape((self._batch, 1, -1)), torch.tensor(1, device=self.device))
        # normalize the probability
        prob *= ~mask
        prob /= prob.sum(axis=-1, keepdim=True)
        # decode the next node to visit (based on the routing agent)
        next_nodes = rou_agent._select_node(prob[:, 0, :], mask[:, 0, :]).reshape(-1, 1)
        return next_nodes

    def _cal_reward(self, loc):
        """
        calculate one-step reward and update the reward recorder
        """
        # get locations
        idx = torch.cat((self.prev_v.reshape(-1, 1, 1), self.prev_v.reshape(-1, 1, 1)), axis=-1)
        idx = idx.to(torch.int64).to(self.device)
        prev_loc = loc.gather(axis=1, index=idx)[:, 0, :]
        idx = torch.cat((self.v.reshape(-1, 1, 1), self.v.reshape(-1, 1, 1)), axis=-1)
        idx = idx.to(torch.int64).to(self.device)
        curr_loc = loc.gather(axis=1, index=idx)[:, 0, :]

        return - (prev_loc - curr_loc).norm(dim=-1, keepdim=True)
