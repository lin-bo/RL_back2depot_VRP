#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Momory for experience replay
"""

from collections import namedtuple
import random

import torch

class replayMem:
    """
    This clas is memeory for experience replay

    Args:
      mem_size (int): size of memory
    """

    def __init__(self, mem_size, seed=135):
        random.seed(seed)
        self.mem_size = mem_size
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
        assert reward.shape[0] == 1, "Only allow one instance, current batch size {}".format(reward.shape[0])
        self.records.append({"s_p":re_state_prev,
                             "a_p":action_prev,
                             "r_pt":reward,
                             "s_t":re_state})
        self.records = self.records[-self.mem_size:]

    def sample(self, batch):
        records = random.sample(self.records, k=min(batch,len(self.records)))
        s_p = records[0]["s_p"]
        a_p = records[0]["a_p"]
        r_pt = records[0]["r_pt"]
        s_t = records[0]["s_t"]
        for record in records[1:]:
            s_p = s_p.cat(record["s_p"])
            a_p = torch.cat((a_p, record["a_p"]), dim=0)
            r_pt = torch.cat((r_pt, record["r_pt"]), dim=0)
            s_t = s_t.cat(record["s_t"])
        return {"s_p":s_p, "a_p":a_p, "r_pt":r_pt, "s_t":s_t}
