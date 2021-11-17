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
from tqdm import tqdm

from prob import VRPDataset
from solver import cwHeuristic, sweepHeuristic, googleOR
from utils import routesPlot, checkValid

def eval(size, algo, solver_args):
    """
    A function to evaluate different algorithms

    Args:
        size(int): graph size
        algo (str): name of algorithm
        solver_args (tuple): args of solver
    """
    # load test data
    print("Load data...")
    print("Graph size: {}".format(size))
    data = VRPDataset(size=size, num_samples=10000)
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
    print()
    # create table
    path = "./res"
    file = "n{}-".format(size) + algo + "-" + "_".join(map(str, solver_args)) + ".csv"
    if not os.path.isdir(path):
        os.mkdir(path)
    print("Save result to " + path + "/" + file)
    df = pd.DataFrame(columns=["Obj", "Routes", "Vehicles", "Elapsed"])
    # run
    print("Run solver...")
    for ins in tqdm(data):
        # get info
        depot = ins["depot"].detach().numpy()
        loc = ins["loc"].detach().numpy()
        demand = ins["demand"].detach().numpy()
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
    parser.add_argument("--algo",
                        type=str,
                        choices=["cw", "sw", "gg"],
                        help="algorithm")
    parser.add_argument("--args",
                        type=int,
                        nargs='*',
                        default=[],
                        help="args of solver")
    # get configuration
    config = parser.parse_args()
    # run
    eval(config.size, config.algo, config.args)
