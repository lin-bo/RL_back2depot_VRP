#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Baseline evaluation
"""

import argparse
import os
import time

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from prob import VRPDataset
from solver import cwHeuristic, sweepHeuristic, googleOR, naiveReturn, amVRP
from utils import routesPlot, checkValid

def eval(size, distr, algo, solver_args):
    """
    A function to evaluate different algorithms

    Args:
        size(int): graph size
        distr(str): data distribution
        algo (str): name of algorithm
        solver_args (tuple): args of solver
    """
    # load test data
    print("Load data...")
    print("Graph size: {}".format(size))
    data = VRPDataset(size=size, distr=distr, num_samples=1000)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    print()
    # select solver
    print("Select solver:")
    if algo == "cw":
        print("  Randomized Clarke-Wright savings heuristic")
        solver = cwHeuristic
        print("  Random depth: {}, random iter: {}".format(solver_args[0], solver_args[1]))
        args = {"rand_depth":solver_args[0], "rand_iter":solver_args[1]}
    if algo == "sw":
        print("  Randomized sweep heuristic")
        solver = sweepHeuristic
        print("  Random iter: {}".format(solver_args[0]))
        args = {"rand_iter":solver_args[0]}
    if algo == "gg":
        print("  Google OR Tools")
        solver = googleOR
        print("  Solution limit: {}".format(solver_args[0]))
        args = {"solution_limit":solver_args[0]}
    if algo == "nr":
        print("  Naive Return")
        solver = naiveReturn
        print("  Capacity threshold: {}".format(solver_args[0]))
        args = {"thre":solver_args[0]}
        prob = solver(size=size, thre=args["thre"])
    if algo == "am":
        print("  Attention Model")
        solver = amVRP
        args = {"method":None}
        if solver_args[0] == 0:
            args["method"] = "greedy"
        if solver_args[0] == 1:
            args["method"] = "sampling"
        print("  Method: {}".format(args["method"]))
        prob = solver(size=size, method=args["method"])
    print()
    # create table
    path = "./res/{}".format(distr)
    file = "n{}-".format(size) + algo + "-" + "_".join(map(str, solver_args)) + ".csv"
    if not os.path.isdir(path):
        os.mkdir(path)
    print("Save result to " + path + "/" + file)
    df = pd.DataFrame(columns=["Obj", "Routes", "Vehicles", "Elapsed"])
    # run
    print("Run solver...")
    for ins in tqdm(dataloader):
        if algo == "nr":
            # cuda
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            ins["depot"] = ins["depot"].to(device)
            ins["loc"] = ins["loc"].to(device)
            ins["demand"] = ins["demand"].to(device)
            # run solver
            tick = time.time()
            routes, obj = prob.solve(ins)
            tock = time.time()
        elif algo == "am":
            # cuda
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            ins["depot"] = ins["depot"].to(device)
            ins["loc"] = ins["loc"].to(device)
            ins["demand"] = ins["demand"].to(device)
            # run solver
            tick = time.time()
            routes, obj = prob.solve(ins)
            routes, obj = routes[0], obj[0]
            tock = time.time()
        else:
            # get info
            depot = ins["depot"].detach().numpy()[0]
            loc = ins["loc"].detach().numpy()[0]
            demand = ins["demand"].detach().numpy()[0]
            # run solver
            prob = solver(depot, loc, demand)
            tick = time.time()
            routes, obj = prob.solve(**args)
            tock = time.time()
        # check valid
        # assert checkValid(routes, depot, loc, demand), "Infeasible solution."
        # table row
        routes_str = ";".join([",".join(map(str, r)) for r in routes])
        num_veh = len(routes)
        elpased = tock - tick
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
    parser.add_argument("--algo",
                        type=str,
                        choices=["cw", "sw", "gg", "nr", "am"],
                        help="algorithm")
    parser.add_argument("--args",
                        type=int,
                        nargs='*',
                        default=[],
                        help="args of solver")
    # get configuration
    config = parser.parse_args()
    # run
    eval(config.size, config.distr, config.algo, config.args)
