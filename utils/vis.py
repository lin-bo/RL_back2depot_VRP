#!/usr/bin/env python
# coding: utf-8
"""
Visualization
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def routesPlot(routes, depot, loc):
    """
    This is a mthod to draw plot of routes

    Args:
        routes: (list[list]): VRP solution
        depot (array): coordinate of central depot
        loc (array): coordinates of customers
    """
    # use different colors
    cmap = cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(routes)))
    # figure size
    plt.figure(figsize=(8, 8))
    # axis range
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # coordinates of routes
    for i, r in enumerate(routes):
        x, y = [depot[0]], [depot[1]]
        for v in r:
            x.append(loc[v,0])
            y.append(loc[v,1])
        x.append(depot[0])
        y.append(depot[1])
        # plot route
        plt.plot(x, y, c=colors[i], ls="-", lw=1, marker='.', ms=8, alpha=0.9)
    # plot depot
    plt.scatter(depot[0], depot[1], c='b', marker='p', s=150)
    plt.show()
