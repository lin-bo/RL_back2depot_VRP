#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Calculation
"""

import torch

def calObj(routes, dist):
    """
    A function to calculate objective value
    """
    obj = 0
    for r in routes:
        prev_ind = 0
        for v in r:
            ind = v + 1
            obj += dist[prev_ind, ind]
            prev_ind = ind
        obj += dist[ind, 0]
    return obj

def rewardCal(step, sa_queue):
    """
    A function to calculate n-step rewards
    """
    r = torch.sum(torch.stack([s.r for s, _ in list(sa_queue.queue)]), dim=0).detach()
    return r
