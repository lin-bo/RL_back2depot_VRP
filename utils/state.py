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

    def __init__(self, batch_data, batch_graph, rou_name='tsp'):
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
        # agent name
        self.name = rou_name

    def update(self, action, rou_agent, rou_state, batch_data):
        """
        A method to update state after action
        return:
            rou_state: new state for the routing agent
            reward: (batch, ) tensor specifying the one-step reward for each instance
        """
        # get demand and loc info
        demand = batch_data["demand"]
        loc = torch.cat((batch_data["depot"].reshape(-1, 1, 2), batch_data["loc"]), axis=1)
        # map -1, 1 to 0, 1
        action_flag = ((action + 1) / 2).to(torch.int32)
        # make routing decision
        next_nodes, mask = self._routing_decision(rou_agent, rou_state, demand)
        # update return agent state
        re_state = self._update_return_state(batch_data, next_nodes, action_flag, demand)
        # update routing agent state
        rou_state = self._update_rou_state(rou_state, next_nodes, action_flag)
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
        new_state = returnState(batch_data, self.g, self.name)
        new_state.prev_v = self.v.clone()
        # create one hot vectors
        one_hot = torch.zeros((self._size + 1, self._size + 1), device=self.device)
        one_hot = one_hot.scatter(0, torch.arange(0, self._size + 1, device=self.device).reshape(1, -1), 1)
        if self.name == 'tsp':
            # update current location
            new_state.v = ((next_nodes + 1) * (1 - action)).to(torch.int32).detach()
            # update capacity
            satisfied = demand.gather(axis=-1, index=next_nodes)
            new_state.c = (1 * action + (self.c - satisfied) * (1 - action)).detach()
            # update visit history
            new_state.o = self.o + one_hot[next_nodes + 1][:,0,:] * (1 - action)
            new_state.o = torch.minimum(new_state.o, torch.tensor(1, device=self.device)).detach()
        else:
            # update current location
            new_state.v = ((next_nodes) * (1 - action)).to(torch.int32).detach()
            # update capacity
            zero = torch.zeros((self._batch, 1), dtype=torch.int32, device=self.device)
            satisfied = demand.gather(axis=-1, index=torch.maximum(next_nodes-1, zero)) * (next_nodes != 0)
            back_flag = (new_state.v == 0).to(torch.int32).to(self.device)
            new_state.c = (1 * back_flag + (self.c - satisfied) * (1 - back_flag)).detach()
            # update visit history
            new_state.o = self.o + one_hot[next_nodes][:, 0, :] * (1 - action)
            new_state.o = torch.minimum(new_state.o, torch.tensor(1, device=self.device)).detach()
        # update graph
        x = self.g.ndata["x"].detach().clone()
        new_state.g.ndata["x"] = x
        for i, g in enumerate(dgl.unbatch(new_state.g)):
            g.ndata["x"][:, 0] = new_state.o[i]
        return new_state

    def _update_rou_state(self, rou_state, next_nodes, action_flag):

        if self.name == 'tsp':
            rou_state = rou_state.new_update(next_nodes.reshape((-1,)), action_flag.reshape((-1,)))
        else:
            rou_state = rou_state.update(next_nodes.reshape((-1,)))

        return rou_state

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
        if self.name == 'tsp':
            # check if the demand at each node exceeds the remaining capacity or not, if so, should be masked
            flag_demand = demand > self.c
            mask = torch.minimum(mask + flag_demand.reshape((self._batch, 1, -1)), torch.tensor(1, device=self.device))
        else:
            depot_mask = torch.cat([torch.ones((self._batch, 1), dtype=torch.float32, device=self.device),
                                    torch.zeros((self._batch, self._size), dtype=torch.float32, device=self.device)],
                                   axis=1).reshape((self._batch, 1, -1))
            mask = torch.maximum(depot_mask, mask)
        # normalize the probability
        prob = (prob + 0.001) * (1 - mask)
        prob /= prob.sum(axis=-1, keepdim=True)
        # decode the next node to visit (based on the routing agent)
        next_nodes = rou_agent._select_node(prob[:, 0, :], mask[:, 0, :]).reshape(-1, 1)
        return next_nodes, mask

    def _cal_reward(self, loc):
        """
        calculate one-step reward and update the reward recorder
        """
        # get locations
        idx = torch.cat((self.prev_v.reshape(-1, 1, 1), self.prev_v.reshape(-1, 1, 1)), axis=-1)
        idx = idx.to(torch.int64)
        prev_loc = loc.gather(axis=1, index=idx)[:, 0, :]
        idx = torch.cat((self.v.reshape(-1, 1, 1), self.v.reshape(-1, 1, 1)), axis=-1)
        idx = idx.to(torch.int64)
        curr_loc = loc.gather(axis=1, index=idx)[:, 0, :]
        return - (prev_loc - curr_loc).norm(dim=-1, keepdim=True)
