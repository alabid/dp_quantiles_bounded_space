#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import importlib
import csv
import Greenwald_Khanna
importlib.reload(Greenwald_Khanna)
from Greenwald_Khanna import GK, FullSpace

random.seed(2021)

# TAXI_ID, TIMESTAMP, TRIP_ID: taxi.csv
# sensor readings (16 channels): ethylene_CO.txt

def load_gas_data(frac):
    x = []
    with open("./data/ethylene_CO.txt") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        first = False
        for row in reader:
            row = [item for item in row if item != '']
            if not first:
                first = True
                continue
            x.append(float(row[3]))

    assert len(x) > 1
    x = np.random.choice(x, size=math.floor(frac*len(x)), replace=False)
    return len(x), x

def load_taxi_data(frac):
    x = []
    with open("./data/taxi.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(float(row["TAXI_ID"]))

    assert len(x) > 1
    x =  np.random.choice(x, size=math.floor(frac*len(x)), replace=False)
    return len(x), x

if __name__ == "__main__":

    # load gas data
    num_items, gas_data = load_gas_data(0.2)
    print("gas_data -> n = ", num_items)
    # load taxis data
    num_items, taxi_data = load_taxi_data(0.2)
    print("taxi_data -> n = ", num_items)

    ### Vary epsilon for DPExpGKGumb (for large and small alpha)
    for a in [0.01, 0.001]:
        print("alpha = ", a)
        xs = []
        epsilons = [0.1, 0.5, 1, 5]

        num_trials = 100
        ## Vary n for large alpha
        ys_exp = [[], []]
        ys_exp_space = [[], []]
        full_ys_exp = [[], []]
        full_ys_exp_space = [[], []]
        q = 0.5

        for eps in epsilons:
            print("eps = ", eps)
            np_gk_unif = GK(a)
            full_np_gk_unif = FullSpace()
            for item in gas_data:
                np_gk_unif.insert(item)
                full_np_gk_unif.insert(item)
            np_gk_normal = GK(a)
            full_np_gk_normal = FullSpace()
            for item in taxi_data:
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

            for j in range(num_trials):
                # Exponential Mechanism
                diff_unif_exp += abs(res_unif - np_gk_unif.dp_exp(q, eps))
                diff_normal_exp += abs(res_normal - np_gk_normal.dp_exp(q, eps))
                diff_full_unif_exp += abs(full_res_unif - full_np_gk_unif.dp_exp(q, eps))
                diff_full_normal_exp += abs(full_res_normal - full_np_gk_normal.dp_exp(q, eps))

            ys_exp[0].append(diff_unif_exp/num_trials)
            ys_exp_space[0].append(len(np_gk_unif.S))
            ys_exp[1].append(diff_normal_exp/num_trials)
            ys_exp_space[1].append(len(np_gk_normal.S))
            full_ys_exp[0].append(diff_full_unif_exp/num_trials)
            full_ys_exp_space[0].append(len(full_np_gk_unif.S))
            full_ys_exp[1].append(diff_full_normal_exp/num_trials)
            full_ys_exp_space[1].append(len(full_np_gk_normal.S))

        print("epsilons=", epsilons)
        print("a=", a, "; q=", q)
        labels = [r"DPExpGKGumb", r"DPExpFull"]
        ## DPExpGKGump (vary epsilon for large and small alpha)
        # DPExpGK vs. DPExpFull: Accuracy (gas sensor)
        print("Accuracies for Gas Sensor")
        print(labels[0])
        print(ys_exp[0])
        print(labels[1])
        print(full_ys_exp[0])
        # DPExpGK vs. DPExpFull: Space (gas sensor)
        print("Space for Gas Sensor")
        print(labels[0])
        print(ys_exp_space[0])
        print(labels[1])
        print(full_ys_exp_space[0])
        # DPExpGK vs. DPExpFull: Accuracy (taxi)
        print("Accuracies for Taxi")
        print(labels[0])
        print(ys_exp[1])
        print(labels[1])
        print(full_ys_exp[1])
        # DPExpGK vs. DPExpFull: Space (taxi)
        print("Space for Taxi")
        print(labels[0])
        print(ys_exp_space[1])
        print(labels[1])
        print(full_ys_exp_space[1])
        print("="*50)
        print("")
