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
from utils import returnState, returnAgent, replayMem, rewardCal


def train(size, step=1, lr=1e-4, batch=64, num_samples=1000, seed=135):
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
    data = VRPDGLDataset(size=size, num_samples=num_samples, seed=seed)
    dataloader = GraphDataLoader(data, batch_size=batch)
    # init memory
    mem = replayMem(mem_size=1000, seed=seed)
    # set time horizon
    horizon = 2 * size
    # load routing agent
    print("\nLoading routing agent...")
    rou_agent = load_routing_agent(size=size)
    # initialize return agent
    print("\nLoading return2depot agent...")
    re_agent = returnAgent(gnn_x_feat=2,
                           gnn_w_feats=1,
                           gnn_e_feats=64,
                           gamma=0.99,
                           lr=lr,
                           seed=seed,
                           logdir="./logs/{}/".format(size))
    print("\nTraining model...")
    time.sleep(1)
    pbar = tqdm(dataloader)
    # init count
    iters = 0
    for batch_data, batch_graph in pbar:
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
        re_state = returnState(batch_data, batch_graph)
        for t in range(horizon):
            # take action
            action = re_agent.actionDecode(batch_graph, re_state, sim=True)
            # update state
            re_state, rou_state = re_state.update(action, rou_agent, rou_state, batch_data)
            # put into queue
            sa_queue.put((re_state, action))
            if t >= step:
                # get reward
                re_state_prev, action_prev = sa_queue.get()
                reward = rewardCal(step, sa_queue)
                # update memory
                mem.update(re_state_prev, action_prev, reward, re_state)
                # update model parameters
                record = mem.sample()
                loss = re_agent.updateModel(record)
                # tqdm log
                desc = "Iter {}, Loss: {:.4f}".format(iters, loss)
                pbar.set_description(desc)
                iters += 1
    # save model
    filename = "vrp-{}.pkl".format(size)
    print("\nSaving model...")
    print("  ./pretrained/{}".format(filename))
    re_agent.saveModel(filename)
