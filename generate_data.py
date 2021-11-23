#!/usr/bin/env python
# coding: utf-8
# Author: Wouter Kool
"""
Data generation
"""

import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()
    problem = "vrp"

    assert opts.filename is None or len(opts.graph_sizes) == 1, \
    "Can only specify filename when generating a single dataset"

    for graph_size in opts.graph_sizes:
        datadir = os.path.join(opts.data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if opts.filename is None:
            filename = os.path.join(datadir, "{}{}_{}_seed{}.pkl".format(
                problem, graph_size, opts.name, opts.seed))
        else:
            filename = check_extension(opts.filename)

        assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

        np.random.seed(opts.seed)
        dataset = generate_vrp_data(opts.dataset_size, graph_size)

        print(dataset[0])

        save_dataset(dataset, filename)
