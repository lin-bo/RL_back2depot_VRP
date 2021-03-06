#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Graph neural networks model
"""

import torch
from torch import nn
from torch.nn import functional as f
import dgl
import dgl.function as fn
from dgl.nn import SumPooling

def copy_v(edges):
    return {'m': edges.dst['h']}

class QGNN(nn.Module):
    """
    This class is gnn

    Args:
        x_feats (int): dimension of node feature
        w_feats (int): dimension of edge weight
        e_feats (int): dimension of embedding
    """
    def __init__(self, x_feats=2, w_feats=1, e_feats=64):
        super(QGNN, self).__init__()
        self._x_feats = x_feats
        self._w_feats = w_feats
        self._e_feats = e_feats
        # layers
        self.layers1 = structure2Vec(x_feats=self._x_feats,
                                     w_feats=self._w_feats,
                                     in_feats=self._x_feats,
                                     out_feats=self._e_feats)
        self.layers2 = structure2Vec(x_feats=self._x_feats,
                                     w_feats=self._w_feats,
                                     in_feats=self._e_feats,
                                     out_feats=self._e_feats)
        self.q_fuction = QFuction(e_feats=self._e_feats)

    def forward(self, state, action):
        """
        A method for forward pass

        Args:
          graph (DGL graph): a batch of graphs
          state (returnState): enviroment state
          action(tensor): a bacth of actions
        """
        h = self.layers1(state.g, state.g.ndata["x"])
        h = self.layers2(state.g, h)
        q = self.q_fuction(h, state, action)
        return q


class structure2Vec(nn.Module):
    """
    This class is structure2vec embedding

    Args:
        x_feats (int): dimension of node feature
        w_feats (int): dimension of edge weight
        in_feats (int): dimension of input feature
        out_feats (int): dimension of output feature
    """

    def __init__(self, x_feats, w_feats, in_feats, out_feats):
        super(structure2Vec, self).__init__()
        self._in_feats = in_feats
        self._x_feats = x_feats
        self._w_feats = w_feats
        self._out_feats = out_feats
        # fc
        self._xfc = nn.Linear(self._x_feats, self._out_feats, bias=False)
        self._wfc = nn.Linear(self._out_feats, self._out_feats, bias=False)
        self._ffc = nn.Linear(self._in_feats, self._out_feats)
        # multiplication weights
        self.weights = nn.Parameter(torch.normal(mean=torch.zeros(self._out_feats),
                                                 std=1e-2))

    def forward(self, graph, feat):
        """
        A method for forward pass

        Args:
          graph (DGL graph): a batch of graphs
          feat (tensor): a batch of embedding features
        """
        h = self._xfc(graph.ndata["x"]) + self._aggw(graph) + self._aggf(graph, feat)
        h = f.relu(h)
        return h

    def _aggw(self, graph):
        with graph.local_scope():
            g = graph.edata["w"].reshape(-1,1) * self.weights
            graph.edata["g"] = f.relu(g)
            graph.update_all(fn.copy_e("g", "m"), fn.sum("m", "h_new"))
            h = graph.ndata["h_new"]
            return self._wfc(h)

    def _aggf(self, graph, feat):
        with graph.local_scope():
            graph.ndata["h"] = feat
            graph.update_all(copy_v, fn.sum("m", "h_new"))
            h = graph.ndata["h_new"]
            return self._ffc(h)


class QFuction(nn.Module):
    """
    This class is Q-function

    Args:
        e_feats (int): dimension of embedding
    """

    def __init__(self, e_feats=64):
        super(QFuction, self).__init__()
        self._e_feats = e_feats
        # pool
        self.sumpool = SumPooling()
        # fc
        self._theta5fc = nn.Linear(self._e_feats*3, 1)
        self._theta6fc = nn.Linear(self._e_feats, self._e_feats)
        self._theta7fc = nn.Linear(self._e_feats, self._e_feats)
        self._theta8fc = nn.Linear(1, self._e_feats)

    def forward(self, feat, state, action):
        """
        A method for forward pass

        Args:
          feat (tensor): a batch of embedding features
          state (returnState): enviroment state
          action(tensor): a bacth of actions
        """
        h1 = self._agglob(feat, state)
        h2 = self._aggcur(feat, state)
        h3 = self._aggact(action)
        h = f.relu(torch.cat((h1, h2, h3), 1))
        q = self._theta5fc(h)
        return q

    def _agglob(self, feat, state):
        feat_sum = self.sumpool(state.g, feat)
        h = self._theta6fc(feat_sum)
        return h

    def _aggcur(self, feat, state):
        cur_feat = self._getCurFeat(state.g, feat, state.v)
        h = self._theta7fc(cur_feat)
        return h

    def _aggact(self, action):
        h = self._theta8fc(action)
        return h

    def _getCurFeat(self, graph, feat, cur_node):
        feat = self._unbatchFeat(graph, feat)
        ind = cur_node.repeat(1, self._e_feats).reshape(-1, 1, self._e_feats).to(torch.int64)
        cur_feat = feat.gather(dim=1, index=ind)
        return cur_feat.reshape(-1, self._e_feats)

    def _unbatchFeat(self, graph, feat):
        return feat.reshape(graph.batch_size, -1, self._e_feats)
