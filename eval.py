#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Evaluation
"""

import argparse
import os
import time

import pandas as pd
from dgl.dataloading import GraphDataLoader
import torch
from tqdm import tqdm

from prob import VRPDGLDataset
from solver import return2Depot

def eval(size, distr, rou_agent_type):
    """
    A function to evaluate different algorithms

    Args:
        size(int): graph size
        distr (str): data distribution
    """
    # device
    print("Device:")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("  {}".format(device))
    # load test data
    print("\nLoad data...")
    print("  Graph size: {}".format(size))
    batch = 64
    data = VRPDGLDataset(size=size, distr=distr, num_samples=1000)
    dataloader = GraphDataLoader(data, batch_size=batch, shuffle=False)
    # create table
    path = "./res/{}".format(distr)
    if rou_agent_type == "tsp":
        file = "n{}-r2d-tsp.csv".format(size)
    else:
        file = "n{}-r2d.csv".format(size)
    if not os.path.isdir(path):
        os.mkdir(path)
    print("Save result to " + path + "/" + file)
    df = pd.DataFrame(columns=["Obj", "Routes", "Vehicles", "Elapsed"])
    # init solver
    solver = return2Depot(size, distr, rou_agent_type)
    # solve
    print("\nEvaluating:")
    pbar = tqdm(dataloader)
    for batch_data, batch_graph in pbar:
        # to device
        batch_graph = batch_graph.to(device)
        batch_data["loc"] = batch_data["loc"].to(device)
        batch_data["demand"] = batch_data["demand"].to(device)
        batch_data["depot"] = batch_data["depot"].to(device)
        tick = time.time()
        batch_routes, batch_objs = solver.solve(batch_data, batch_graph)
        tock = time.time()
        # save result
        for i in range(batch_graph.batch_size):
            obj = batch_objs[i]
            routes = batch_routes[i]
            depot = batch_data["depot"].cpu().detach().numpy()[i]
            loc = batch_data["loc"].cpu().detach().numpy()[i]
            demand = batch_data["demand"].cpu().detach().numpy()[i]
            # check valid
            # assert checkValid(routes, depot, loc, demand), "Infeasible solution."
            routes_str = ";".join([",".join(map(str, r)) for r in routes])
            num_veh = len(routes)
            elpased = (tock - tick) / batch
            row = {"Obj":obj, "Routes":routes_str, "Vehicles":num_veh, "Elapsed":elpased}
            df = df.append(row, ignore_index=True)
            df.to_csv(path+"/"+file, index=False)

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
    parser.add_argument("--agent",
                        type=str,
                        choices=["tsp", "vrp"],
                        help="route agent type")
    config = parser.parse_args()
    # run
    eval(config.size, config.distr, config.agent)
