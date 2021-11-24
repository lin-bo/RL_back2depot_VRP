#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Graph neural networks model
"""

import torch
from torch import nn
from torch.nn import functional as f
import dgl.function as fn


class GNN(nn.Module):
    """
    This class is gnn

    Args:
        x_feats (int): dimension of node feature
        w_feats (int): dimension of edge weight
        e_feats (int): dimension of embedding
    """
    def __init__(self, x_feats=2, w_feats=1, e_feats=64):
        super(GNN, self).__init__()
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

    def forward(self, graph):
        h = self.layers1(graph, graph.ndata["x"])
        h = self.layers2(graph, h)
        return h


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
        self.weights = nn.Parameter(torch.Tensor(self._out_feats))

    def forward(self, graph, feat):
        h = self._xfc(graph.ndata["x"]) + self._aggw(graph) + self._aggf(graph, feat)
        return f.relu(h)

    def _aggw(self, graph):
        with graph.local_scope():
            g = torch.stack([w * self.weights for w in graph.edata["w"]])
            graph.edata["g"] = f.relu(g)
            graph.update_all(fn.copy_e("g", "m"), fn.sum("m", "h_new"))
            h = graph.ndata["h_new"]
            return self._wfc(h)

    def _aggf(self, graph, feat):
        with graph.local_scope():
            graph.ndata["h"] = feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h_new"))
            h = graph.ndata["h_new"]
            return self._ffc(h)
