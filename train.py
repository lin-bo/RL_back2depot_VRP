#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin & Bo Tang
"""
Deep Q-Learning
"""

import queue

import torch

from dgl.dataloading import GraphDataLoader
from prob import VRPDGLDataset
from attention_model import load_routing_agent
from utils.state import returnState

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
    # cuda
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)
    # load dataset
    print("Generating dataset...")
    data = VRPDGLDataset(num_samples=num_samples, seed=seed)
    dataloader = GraphDataLoader(data, batch_size=batch)
    # init memory
    mem = memInit()
    # set optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # set time horizon
    horizon = 2 * size
    # load routing agent
    print("\nLoad routing agent...")
    rou_agent = load_routing_agent(size=size)
    print("\nTraining model...")
    for batch_data, batch_graph in dataloader:
        # to device
        batch_graph.to(device)
        batch_data["loc"] = batch_data["loc"].to(device)
        batch_data["demand"] = batch_data["demand"].to(device)
        batch_data["depot"] = batch_data["depot"].to(device)
        # set state-action queue
        sa_queue = queue.Queue()
        # initialize routing state
        rou_state = rou_agent.re_init(batch_data['loc'])
        # init state
        state = returnState(batch_data)
        for t in range(horizon):
            # take action
            action = actionDecode(batch_graph, state)
            # update state
            state.update(action, state, rou_agent, rou_state)
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
            break


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
