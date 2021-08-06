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

if __name__ == "__main__":

    ### Vary n for DPExpGKGumb (for large and small alpha)
    for a in [0.01, 0.001]:
        print("alpha = ", a)
        xs = [1000000, 10000000]
        print("sizes = ", xs)
        num_trials = 10
        ## Vary n for large alpha
        eps = 1
        ys_exp = [[], []]
        q = 0.5
        for num_items in xs:
            unif_data = np.random.uniform(0, 10, num_items)
            normal_data = np.random.normal(5, 1, num_items)

            np_gk_unif = GK(a)
            for item in unif_data:
                np_gk_unif.insert(item)
            np_gk_normal = GK(a)
            for item in normal_data:
                np_gk_normal.insert(item)

            res_index, res_unif = np_gk_unif.quantile(q)
            res_index, res_normal = np_gk_normal.quantile(q)

            diff_unif_exp = 0
            diff_normal_exp = 0

            for j in range(num_trials):
                print("trial number: ", j)
                # Exponential Mechanism
                diff_unif_exp += abs(res_unif - np_gk_unif.dp_exp(q, eps))
                diff_normal_exp += abs(res_normal - np_gk_normal.dp_exp(q, eps))

            ys_exp[0].append(diff_unif_exp/num_trials)
            ys_exp[1].append(diff_normal_exp/num_trials)

        print(ys_exp[0])
        print(ys_exp[1])

