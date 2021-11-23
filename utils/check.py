#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Feasibility checking
"""

def checkValid(routes, depot, loc, demand):
    """
    A function to check solution feasibility
    """
    # init set
    nodes = set(range(len(loc)))
    # check
    for r in routes:
        left = 1
        for v in r:
            # unvisted nodes
            if v not in nodes:
                print("Node {} is visited or not existed.".format(v))
                return False
            # unsatisfied demand
            left -= demand[v]
            if left < -1e-7:
                print("Demand of node {} cannot be satisfied.".format(v))
                print("Left capacity: {}".format(left))
                return False
            # remove
            nodes.remove(v)
    # go through all node
    if nodes:
        print("Nodes", ", ".join(map(str, list(nodes))), "are not visited.")
        return False
    else:
        return True
