#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin & Bo Tang
"""
Deep Q-Learning
"""

import queue
import time

import torch
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from prob import VRPDGLDataset
from attention_model import load_routing_agent
from utils import returnState, returnAgent, rewardCal


def train(size, step=10, lr=1e-4, batch=64, num_samples=1000, seed=135):
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
    # device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # load dataset
    print("\nGenerating dataset...")
    time.sleep(1)
    data = VRPDGLDataset(num_samples=num_samples, seed=seed)
    dataloader = GraphDataLoader(data, batch_size=batch)
    # init memory
    mem = memInit()
    # set time horizon
    horizon = 2 * size
    # load routing agent
    print("\nLoading routing agent...")
    rou_agent = load_routing_agent(size=size)
    # initialize return agent
    print("\nLoading return2depot agent...")
    re_agent = returnAgent(gnn_x_feat=2, gnn_w_feats=1, gnn_e_feats=64)
    print("\nTraining model...")
    time.sleep(1)
    for batch_data, batch_graph in tqdm(dataloader):
        # to device
        batch_graph = batch_graph.to(device)
        batch_data["loc"] = batch_data["loc"].to(device)
        batch_data["demand"] = batch_data["demand"].to(device)
        batch_data["depot"] = batch_data["depot"].to(device)
        # set state-action queue
        sa_queue = queue.Queue()
        # initialize return state
        rou_state = rou_agent.re_init(batch_data['loc'])
        # init state
        re_state = returnState(batch_data)
        for t in range(horizon):
            # take action
            action, _ = re_agent.actionDecode(batch_graph, re_state)
            # update state
            re_state, rou_state = re_state.update(action, rou_agent, rou_state, batch_data)
            # put into queue
            sa_queue.put((re_state, action))
            if t >= step:
                # get reward
                sa_queue.get()
                reward = rewardCal(step, sa_queue)
                # update memory
                mem = memUpdate(mem)
                # update model parameters
                re_agent.updateModel()

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


def modelUpdate(model, mem):
    """
    A function to decrease loss with SDG
    """
    pass
