#!/usr/bin/env python
# coding: utf-8
from solver.cw import cwHeuristic
from solver.sweep import sweepHeuristic
from solver.ortools import googleOR
from solver.naive_return import naiveReturn
from solver.rl_return import return2Depot
from solver.am_vrp_solver import amVRP