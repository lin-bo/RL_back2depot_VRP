#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Retuen-to-depot agent
"""

import torch

from model import QGNN

class returnAgent:
    """
    This class is return-to-depot agent
    """

    def __init__(self, gnn_x_feat, gnn_w_feats, gnn_e_feats):
        # nn
        self.q_gnn = QGNN(x_feats=gnn_x_feat, w_feats=gnn_w_feats, e_feats=gnn_e_feats)
        # cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.q_gnn = self.q_gnn.to(self.device)

    def actionDecode(self, batch_graph, state):
        """
        A method to decode action

        Args:
          batch_graph (DGL graph): a batch of graphs
          state (returnState): enviroment state
        """
        # action choice
        batch = batch_graph.batch_size
        action_return = - torch.ones((batch,1), device=self.device)
        action_noreturn = torch.ones((batch,1), device=self.device)
        # calculate Q value
        q_r = self.QValue(batch_graph, state, action_return)
        print(q_r)
        q_n = self.QValue(batch_graph, state, action_noreturn)

    def QValue(self, batch_graph, state, action):
        """
        A method to obtain Q-value

        Args:
          batch_graph (DGL graph): a batch of graphs
          state (returnState): enviroment state
          action(tensor): a bacth of actions
        """
        q = self.q_gnn(batch_graph, state, action)
        return q
