#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin & Bo Tang
"""
Deep Q-Learning
"""

import queue

from torch import optim
from dgl.dataloading import GraphDataLoader
from prob import VRPDGLDataset


def train(model, size, step=10, lr=1e-4, batch=64, num_samples=1000, seed=135):
    """
    A function to train back2depot DQN

    Args:
        model(torch.nn): graph neural network
        size(int): graph size
        step(int): step length
        lr(float): learning rate
        batch(int): batch size
        epoch(int): number of epochs
        num_samples(int): number of training instances
    """
    # load dataset
    print("Generating dataset...")
    data = VRPDGLDataset(num_samples=num_samples, seed=seed)
    dataloader = GraphDataLoader(data, batch_size=batch)
    # init memory
    mem = memInit()
    # set optimizer
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    # set time horizon
    horizon = 2 * size
    print("Training model...")
    for batch_graph in dataloader:
        # set state-action queue
        sa_queue = queue.Queue()
        # init state
        state = stateInit()
        for t in range(horizon):
            # take action
            action = actionDecode(batch_graph, state)
            # update state
            state = stateUpdate(action)
            # put into queue
            sa_queue.put((state, action))
            if t >= step - 1:
                # get reward
                reward = rewardObtain(list(sa_queue.queue))
                sa_queue.get()
                # update memory
                mem = memUpdate(mem)
                # update model parameters
                modelUpdate(model, mem)


def memInit():
    """
    A function to initialize memory
    """
    pass


def memUpdate(action):
    """
    A fuctiion to update memory
    """
    pass


def stateInit():
    """
    A function to initialize state
    """
    pass


def stateUpdate(action):
    """
    A fuctiion to update state after action
    """
    pass


def actionDecode(batch_graph, state):
    """
    A function to decode depot-return action
    """
    pass


def rewardObtain(sa_queue):
    """
    A function to calculate reward
    """
    pass


def modelUpdate(model, mem):
    """
    A function to decrease loss with SDG
    """
    pass
