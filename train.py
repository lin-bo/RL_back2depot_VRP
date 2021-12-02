#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin & Bo Tang
"""
Deep Q-Learning
"""

import argparse
import queue
import time

import torch
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader

from prob import VRPDGLDataset
from attention_model import load_routing_agent
from utils import returnState, returnAgent, replayMem, rewardCal


def train(size, rou_agent_type="vrp", distr="uniform", step=1, lr=1e-4, batch=64, num_samples=10000, seed=135):
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
    data = VRPDGLDataset(size=size, distr=distr, num_samples=num_samples, seed=seed)
    dataloader = GraphDataLoader(data, batch_size=1)
    # init memory
    mem = replayMem(mem_size=10000, seed=seed)
    # set time horizon
    horizon = 2 * size
    # load routing agent
    print("\nLoading routing agent...")
    rou_agent = load_routing_agent(size=size, name=rou_agent_type)
    # initialize return agent
    print("\nLoading return2depot agent...")
    re_agent = returnAgent(gnn_x_feat=4,
                           gnn_w_feats=1,
                           gnn_e_feats=64,
                           gamma=0.99,
                           epsilon=0.6,
                           lr=lr,
                           seed=seed,
                           logdir="./logs/{}/{}/{}/".format(rou_agent_type, distr, size))
    print("\nTraining model...")
    time.sleep(1)
    pbar = tqdm(dataloader)
    # init count
    iters = 0
    for i, (batch_data, batch_graph) in enumerate(pbar):
        # to device
        batch_graph = batch_graph.to(device)
        batch_data["loc"] = batch_data["loc"].to(device)
        batch_data["demand"] = batch_data["demand"].to(device)
        batch_data["depot"] = batch_data["depot"].to(device)
        # set state-action queue
        sa_queue = queue.Queue()
        # initialize return state
        rou_state = rou_agent.re_init(batch_data)
        # init state
        re_state = returnState(batch_data, batch_graph, rou_agent_type)
        for t in range(horizon):
            # take action
            action = re_agent.actionDecode(re_state, explore=True)
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
                record = mem.sample(batch)
                loss = re_agent.updateModel(record)
                # tqdm log
                desc = "Iter {}, Loss: {:.4f}".format(iters, loss)
                pbar.set_description(desc)
                iters += 1
                # all visited
                if torch.all(re_state.o[:,1:]).item():
                    break
        re_agent.epsilon = 0.1 + 0.5 * 0.97 ** i
        if i and i % 10 == 0:
            filename = "{}-{}-{}.pkl".format(rou_agent_type, distr, size)
            re_agent.saveModel(filename)
    # save model
    filename = "{}-{}-{}.pkl".format(rou_agent_type, distr, size)
    print("\nSaving model...")
    print("  ./pretrained/{}".format(filename))
    re_agent.saveModel(filename)

if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument("--size",
                        type=int,
                        choices=[20, 50, 100],
                        help="graph size")
    parser.add_argument("--distr",
                        type=str,
                        choices=["uniform", "cluster"],
                        help="data distribution")
    parser.add_argument("--step",
                        type=int,
                        default=3,
                        help="step length")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("--batch",
                        type=int,
                        default=64,
                        help="batch size")
    # get configuration
    config = parser.parse_args()
    # run
    train(size=config.size, distr=config.distr, rou_agent_type="vrp", step=config.step,
          lr=config.lr, batch=64, num_samples=1000, seed=135)
