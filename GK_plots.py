#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import importlib
import Greenwald_Khanna
importlib.reload(Greenwald_Khanna)
from Greenwald_Khanna import GK

random.seed(2021)

def GK_plot(alpha, n, streams, labels):
    np_gks = []
    for i in range(len(labels)):
        np_gks.append(GK(alpha))
        for item in streams[i]:
            np_gks[i].insert(item)
    qs = np.arange(0, 1, 0.05)
    vs = []
    for i in range(len(labels)):
        values = []
        for q in np.arange(0, 1, 0.05):
            res_index, res = np_gks[i].quantile(q)
            values.append(res)
        vs.append(np.array(values))
    for i in range(len(labels)):
        plt.plot(qs, vs[i], label=labels[i])
    plt.xlabel(r"Value")
    plt.ylabel(r"Percentile")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("images/gk_{0}_{1}_{2}.png".format(alpha, n, '_'.join(labels)))
    plt.gcf().clear()

if __name__ == "__main__":
    np_gk = GK(0.01)
    for item in np.random.uniform(0, 1, 10000):
        np_gk.insert(item)
    res_real = np_gk.quantile(0.5)
    res_dp = np_gk.dp_exp_mech_quantile(0.5, 10)
    print(res_real, res_dp)
    res_real = np_gk.quantile(0.2)
    res_dp = np_gk.dp_exp_mech_quantile(0.2, 10)
    print(res_real, res_dp)
    res_dp = np_gk.dp_exp_mech_quantile(0.2, 1)
    print(res_real, res_dp)
    res_dp = np_gk.dp_exp_mech_quantile(0.2, 0.0001)
    print(res_real, res_dp)

    grad = 0.001
    np_gk = GK(0.01)
    for item in np.random.uniform(0, 1, 10000):
        np_gk.insert(math.floor(item*1/grad)*grad)
    res_real = np_gk.quantile(0.2)
    res_dp = np_gk.dp_exp_mech_quantile(0.2, 1)
    print(res_real, res_dp)
    print(res_real, np_gk.dp_histogram_quantile(0.2, 1))
    res_real = np_gk.quantile(0.5)
    res_dp = np_gk.dp_exp_mech_quantile(0.5, 1)
    print(res_real, res_dp)
    print(res_real, np_gk.dp_histogram_quantile(0.5, 1))
    '''
    for a in [0.1, 0.01, 0.001, 0.0001]:
        for num_items in [1000, 10000, 100000, 100000]:
            unif_data = np.random.uniform(0, 1, num_items)
            normal_data = np.random.normal(0.5, 1/12., num_items)
            GK_plot(a, num_items, [unif_data, normal_data], [r"Unif(0,1)", r"Normal(0.5,0.0833)"])
    '''
    
