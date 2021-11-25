#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Calculation
"""

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
    A function to calculate n-step rewards + qval
    """
    r = torch.zeros(qval.shape, dtype=torch.float32, device=qval.device)
    for (s, _) in list(sa_queue.queue)[-step - 1: -1]:
        r += s.r
    return r
