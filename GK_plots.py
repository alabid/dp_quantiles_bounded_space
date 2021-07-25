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
        plt.plot(vs[i], qs, label=labels[i])
    plt.ylabel(r"Percentile")
    plt.xlabel(r"Value")
    plt.legend()
    plt.savefig("images/gk_{0}_{1}_{2}.png".format(alpha, n, '_'.join(labels)))
    plt.gcf().clear()

def DP_GK_plot(alpha, n, eps, streams, labels, num_trials):
    eps = float(eps)

    np_gks = []
    for i in range(len(labels)):
        np_gks.append(GK(alpha))
        for item in streams[i]:
            np_gks[i].insert(item)
    qs = np.arange(0, 1, 0.05)
    vs = []
    for i in range(len(labels)):
        values = [[], []]
        for q in np.arange(0, 1, 0.05):
            res_index, res = np_gks[i].quantile(q)
            diff1 = 0
            diff2 = 0
            for j in range(num_trials):
                res1 = np_gks[i].dp_exp(q, eps)
                diff1 += abs(res1-res)
                res2 = 0
                res2 = np_gks[i].dp_hist(q, eps, 0, alpha/2, 1/(2*0.05))
                diff2 += abs(res2-res)
            values[0].append(diff1/num_trials)
            values[1].append(diff2/num_trials)
        vs.append(np.array(values))
    for i in range(len(labels)):
        plt.plot(qs, vs[i][0], label="Exp"+labels[i])
        plt.plot(qs, vs[i][1], label="Hist"+labels[i])
    plt.ylabel(r"Absolute Error")
    plt.xlabel(r"Percentile")
    plt.legend()
    plt.savefig("images/p_gk_{0}_{1}_{2}.png".format(alpha, n, '_'.join(labels)))
    print("Saving images/p_gk_{0}_{1}_{2}.png".format(alpha, n, '_'.join(labels)))
    plt.gcf().clear()

if __name__ == "__main__":

    ### Vary n for DPExpGKGumb and DPHistGK
    num_trials = 1000
    labels = [r"Unif(0,1)", r"Normal(0.5,1/12)"]
    ## Vary n for large alpha
    a = 0.1
    eps = 1
    xs = [100, 1000, 10000, 100000]
    ys_exp = [[], []]
    ys_hist = [[], []]
    full_ys_exp = [[], []]
    full_ys_hist = [[], []]
    q = 0.5
    for num_items in xs:
        print("n = ", num_items)
        unif_data = np.random.uniform(0, 1, num_items)
        normal_data = np.random.normal(0.5, math.sqrt(1/12.), num_items)

        np_gk_unif = GK(a)
        full_np_gk_unif = GK(1.0/num_items)
        for item in np.random.uniform(0, 1, num_items):
            np_gk_unif.insert(item)
            full_np_gk_unif.insert(item)
        np_gk_normal = GK(a)
        full_np_gk_normal = GK(1.0/num_items)
        for item in np.random.normal(0.5, math.sqrt(1/12.), num_items):
            np_gk_normal.insert(item)
            full_np_gk_normal.insert(item)

        res_index, res_unif = np_gk_unif.quantile(q)
        res_index, res_normal = np_gk_normal.quantile(q)
        res_index, full_res_unif = full_np_gk_unif.quantile(q)
        res_index, full_res_normal = np_gk_normal.quantile(q)

        diff_unif_exp = 0
        diff_normal_exp = 0
        diff_full_unif_exp = 0
        diff_full_normal_exp = 0
        diff_unif_hist = 0
        diff_normal_hist = 0
        diff_full_unif_hist = 0
        diff_full_normal_hist = 0

        for j in range(num_trials):
            # Exponential Mechanism
            diff_unif_exp += abs(res_unif - np_gk_unif.dp_exp(q, eps))
            diff_normal_exp += abs(res_unif - np_gk_normal.dp_exp(q, eps))
            diff_full_unif_exp += abs(full_res_unif - full_np_gk_unif.dp_exp(q, eps))
            diff_full_normal_exp += abs(full_res_unif - full_np_gk_normal.dp_exp(q, eps))

            # Histogram
            diff_unif_hist += abs(res_unif - np_gk_unif.dp_hist(q, eps))
            diff_normal_hist += abs(res_unif - np_gk_normal.dp_hist(q, eps))
            diff_full_unif_hist += abs(full_res_unif - full_np_gk_unif.dp_hist(q, eps))
            diff_full_normal_hist += abs(full_res_unif - full_np_gk_normal.dp_hist(q, eps))

        ys_exp[0].append(diff_unif_exp/num_trials)
        ys_exp[1].append(diff_normal_exp/num_trials)
        full_ys_exp[0].append(diff_full_unif_exp/num_trials)
        full_ys_exp[1].append(diff_full_normal_exp/num_trials)
        ys_hist[0].append(diff_unif_hist/num_trials)
        ys_hist[1].append(diff_normal_hist/num_trials)
        full_ys_hist[0].append(diff_full_unif_hist/num_trials)
        full_ys_hist[1].append(diff_full_normal_hist/num_trials)
    # DPExpGKGump: vary n for large alpha
    bar_width = 0.25
    pos1 = np.arange(len(ys_exp[0]))
    pos2 = [x + bar_width for x in pos1]

    plt.bar(pos1, ys_exp[0], color='r', width=bar_width, label=labels[0])
    plt.bar(pos2, ys_exp[1], color='g', width=bar_width, label=labels[1])
    plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=25, fontweight="bold")
    plt.ylabel(r"Absolute Error", fontsize=25, fontweight="bold")
    plt.legend()
    plt.show()
    plt.savefig("new_images/dp_gk_{0}_{1}_{2}.png".format(alpha, eps, q))
    plt.gcf().clear()
    '''
    ## Vary n for DPExpGKGumb and DPHistGK
    a = 0.0001
    eps = 1
    xs = [10000, 100000, 1000000, 10000000]
    ys = []
    np_gks = []
    for num_items in xs:
        unif_data = np.random.uniform(0, 1, num_items)
        normal_data = np.random.normal(0.5, math.sqrt(1/12.), num_items)
        np_gks.append(GK(alpha))
        for item in streams[i]:
            np_gks[i].insert(item)
    '''
    
