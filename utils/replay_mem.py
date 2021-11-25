#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Momory for experience replay
"""

from collections import namedtuple
import random

class replayMem:
    """
    This clas is memeory for experience replay

    Args:
      mem_size (int): size of memory
    """

    def __init__(self, mem_size, seed=135):
        random.seed(seed)
        self.mem_size = mem_size
        self.tuple = namedtuple("Record", ["s_p", "a_p", "r_pt", "s_t"])
        self.records = []

    def update(self, re_state_prev, action_prev, reward, re_state):
        """
        A method to update memory record

        Args:
          re_state_prev (returnState): state t-n
          action_prev (tensor): action t-n
          reward (tensor): cumulative reward
          re_state (returnState): state t
        """
        self.records.append(self.tuple(re_state_prev, action_prev, reward, re_state))
        self.records = self.records[-self.mem_size:]

    def sample(self):
        record = random.choice(self.records)
        return record
