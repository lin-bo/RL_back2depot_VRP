#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Retuen-to-depot agent
"""

import os

import torch
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import QGNN

class returnAgent:
    """
    This class is return-to-depot agent
    """

    def __init__(self, gnn_x_feat, gnn_w_feats, gnn_e_feats, gamma, lr):
        # nn
        self.q_gnn = QGNN(x_feats=gnn_x_feat, w_feats=gnn_w_feats, e_feats=gnn_e_feats)
        # cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.q_gnn = self.q_gnn.to(self.device)
        # recay rate
        self.gamma = gamma
        # optimizer
        self.optim = optim.Adam(self.q_gnn.parameters(), lr=lr)
        # scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, 1000, gamma=0.95)
        # loss
        self.criterion = nn.MSELoss(reduction="mean")
        # tensorboard
        self.cnt = 0
        self.writer = SummaryWriter(log_dir="./logs")
        # model dir
        self.model_dir = "./pretrained"
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def actionDecode(self, batch_graph, state):
        """
        A method to decode action

        Args:
          batch_graph (DGL graph): a batch of graphs
          state (returnState): enviroment state
        """
        self.q_gnn.eval()
        action, _ = self.getMaxQ(batch_graph, state)
        return action

    def getMaxQ(self, batch_graph, state):
        """
        A method to decode action

        Args:
          batch_graph (DGL graph): a batch of graphs
          state (returnState): enviroment state
        """
        # action choice
        batch = batch_graph.batch_size
        action_noreturn = - torch.ones((batch,1), device=self.device)
        action_return = torch.ones((batch,1), device=self.device)
        # calculate Q value
        q_n = self.q_gnn(state, action_noreturn)
        q_r = self.q_gnn(state, action_return)
        q = torch.cat((q_n, q_r), 1)
        # max value
        qind = torch.argmax(q, dim=1).reshape(batch, 1)
        # force to not return on depot
        for i in range(batch):
            if state.v[i].item() == 0 and not torch.all(state.o[i,1:]).item():
                qind[i,0] = 0
        action = (qind - 0.5) * 2
        qvalue = q.gather(dim=1, index=qind)
        return action, qvalue

    def updateModel(self, record):
        """
        A method to update model by SDG

        Args:
          record (namedtuple): a record of MDP steps
        """
        # calculate loss
        s_p, a_p, r_pt, s_t = record.s_p, record.a_p, record.r_pt, record.s_t
        self.q_gnn.train()
        q_p = self.q_gnn(s_p, a_p)
        _, max_q_t = self.getMaxQ(s_t.g, s_t)
        y = r_pt + self.gamma * max_q_t
        loss = self.criterion(q_p, y)
        # backward pass
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        # tensorboard log
        self.writer.add_scalar('Loss', loss.item(), self.cnt)
        self.cnt += 1
        return loss.item()
