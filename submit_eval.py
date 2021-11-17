#!/usr/bin/env python
# coding: utf-8
# Author: Bo Tang
"""
Submit task to Compute Canada
"""

import argparse
import sys
sys.path.append("~/projects/def-khalile2/botang/rl_vrp/")

import submitit

from eval import eval

# init parser
parser = argparse.ArgumentParser()

# configuration
parser.add_argument("--size",
                    type=int,
                    choices=[20, 50, 100],
                    help="graph size")
parser.add_argument("--algo",
                    type=str,
                    choices=["cw", "sw"],
                    help="algorithm")
parser.add_argument("--args",
                    type=int,
                    nargs='*',
                    default=[],
                    help="args of solver")

# get configuration
config = parser.parse_args()

# job submission parameters
instance_logs_path = "slurm_logs_spotest"
if config.algo == "cw":
    if config.size == 20:
        timeout_min = 10 * config.args[0] * config.args[1]
    if config.size == 50:
        timeout_min = 40 * config.args[0] * config.args[1]
    if config.size == 100:
        timeout_min = 100 * config.args[0] * config.args[1]
if config.algo == "sw":
    if config.size == 20:
        timeout_min = 15 * config.args[0]
    if config.size == 50:
        timeout_min = 30 * config.args[0]
    if config.size == 100:
        timeout_min = 60 * config.args[0]
mem_gb = 1
num_cpus = 1

# create executor
executor = submitit.AutoExecutor(folder=instance_logs_path)
executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                           timeout_min=timeout_min,
                           mem_gb=mem_gb,
                           cpus_per_task=num_cpus)

# submit job
job = executor.submit(eval, config.size, config.algo, config.args)
print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}" \
      .format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))

# get result
#job.result()
