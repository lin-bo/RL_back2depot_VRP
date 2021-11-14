#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Abstract solver
"""

from abc import ABC, abstractmethod
import numpy as np

class ABSolver(ABC):
    """
    This is an abstract class for VRP solver

    Args:
        depot (int): coordinate of central depot
        loc (str): coordinates of customers
        demand (int): demands of customers
    """

    def __init__(self, depot, loc, demand, seed=135):
        self.depot = depot
        self.loc = loc
        self.demand = demand
        # random seed
        np.random.seed(135)
        # graph size
        self.size = len(self.loc)

    @abstractmethod
    def solve(self):
        """
        An abstract method to solve model
        """
        raise NotImplementedError
