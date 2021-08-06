#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import importlib
import Greenwald_Khanna
importlib.reload(Greenwald_Khanna)
from Greenwald_Khanna import GK, FullSpace

random.seed(2021)

if __name__ == "__main__":

    ### Vary n for DPExpGKGumb (for large and small alpha)
    for a in [0.1, 0.01, 0.001, 0.0001]:
        print("alpha = ", a)
        xs = np.arange(1, 5)*math.ceil((1/0.00001))

        num_trials = 1000
        ## Vary n for large alpha
        eps = 1
        ys_exp = [[], []]
        ys_exp_space = [[], []]
        full_ys_exp = [[], []]
        full_ys_exp_space = [[], []]
        q = 0.5
        for num_items in xs:
            print("starting n = ", num_items)
            unif_data = np.random.uniform(0, 10, num_items)
            normal_data = np.random.normal(5, 1, num_items)

            np_gk_unif = GK(a)
            full_np_gk_unif = FullSpace()
            for item in unif_data:
                np_gk_unif.insert(item)
                full_np_gk_unif.insert(item)
            np_gk_normal = GK(a)
            full_np_gk_normal = FullSpace()
            for item in normal_data:
                np_gk_normal.insert(item)
                full_np_gk_normal.insert(item)

            res_index, res_unif = np_gk_unif.quantile(q)
            res_index, res_normal = np_gk_normal.quantile(q)
            res_index, full_res_unif = full_np_gk_unif.quantile(q)
            res_index, full_res_normal = full_np_gk_normal.quantile(q)

            diff_unif_exp = 0
            diff_normal_exp = 0
            diff_full_unif_exp = 0
            diff_full_normal_exp = 0

            for j in range(num_trials):
                # Exponential Mechanism
                diff_unif_exp += abs(res_unif - np_gk_unif.dp_exp(q, eps))
                diff_normal_exp += abs(res_unif - np_gk_normal.dp_exp(q, eps))
                diff_full_unif_exp += abs(full_res_unif - full_np_gk_unif.dp_exp(q, eps))
                diff_full_normal_exp += abs(full_res_unif - full_np_gk_normal.dp_exp(q, eps))

            ys_exp[0].append(diff_unif_exp/num_trials)
            ys_exp_space[0].append(len(np_gk_unif.S))
            ys_exp[1].append(diff_normal_exp/num_trials)
            ys_exp_space[1].append(len(np_gk_normal.S))
            full_ys_exp[0].append(diff_full_unif_exp/num_trials)
            full_ys_exp_space[0].append(len(full_np_gk_unif.S))
            full_ys_exp[1].append(diff_full_normal_exp/num_trials)
            full_ys_exp_space[1].append(len(full_np_gk_unif.S))

            print("done with n = ", num_items)

        ## DPExpGKGump (vary n for large alpha)
        bar_width = 0.25
        labels = [r"Unif(0, 10)", r"Normal(5, 1)"]
        # DPExp (uniform vs. normal)
        pos1 = np.arange(len(ys_exp[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, ys_exp[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], xs)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_{0}_{1}_{2}.png".format(a, eps, q))
        np.save("new_vectors/dp_exp_gk_{0}_{1}_{2}_unif.npy".format(a, eps, q), ys_exp[0])
        np.save("new_vectors/dp_exp_gk_{0}_{1}_{2}_normal.npy".format(a, eps, q), ys_exp[1])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Accuracy (uniform)
        labels = [r"DPExpGKGumb", r"DPExpFull"]
        pos1 = np.arange(len(ys_exp[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp[0], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], xs)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_full_unif_{0}_{1}_{2}.png".format(a, eps, q))
        np.save("new_images/dp_exp_gk_full_unif_{0}_{1}_{2}_gk.npy".format(a, eps, q), ys_exp[0])
        np.save("new_images/dp_exp_gk_full_unif_{0}_{1}_{2}_full.npy".format(a, eps, q), full_ys_exp[0])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Space (uniform)
        labels = [r"DPExpGKGumb", r"DPExpFull"]
        pos1 = np.arange(len(ys_exp_space[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp_space[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp_space[0], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], xs)
        plt.legend(fontsize=18)
        plt.yscale('log')
        plt.savefig("new_images/dp_exp_gk_full_unif_space_{0}_{1}_{2}.png".format(a, eps, q))
        np.save("new_vectors/dp_exp_gk_full_unif_space_{0}_{1}_{2}_gk.npy".format(a, eps, q), ys_exp_space[0])
        np.save("new_vectors/dp_exp_gk_full_unif_space_{0}_{1}_{2}_full.npy".format(a, eps, q), full_ys_exp_space[0])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull (normal)
        pos1 = np.arange(len(ys_exp[1]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[1], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[1]))], xs)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_full_normal_{0}_{1}_{2}.png".format(a, eps, q))
        np.save("new_vectors/dp_exp_gk_full_normal_{0}_{1}_{2}_gk.npy".format(a, eps, q), ys_exp[1])
        np.save("new_vectors/dp_exp_gk_full_normal_{0}_{1}_{2}_full.npy".format(a, eps, q), full_ys_exp[1])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Space (normal)
        pos1 = np.arange(len(ys_exp_space[1]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp_space[1], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp_space[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Dataset/Stream Size ($n$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp_space[1]))], xs)
        plt.legend(fontsize=18)
        plt.yscale('log')
        plt.savefig("new_images/dp_exp_gk_full_normal_space_{0}_{1}_{2}.png".format(a, eps, q))
        np.save("new_vectors/dp_exp_gk_full_normal_space_{0}_{1}_{2}_gk.npy".format(a, eps, q), ys_exp_space[1])
        np.save("new_vectors/dp_exp_gk_full_normal_space_{0}_{1}_{2}_full.npy".format(a, eps, q), full_ys_exp_space[1])
        plt.gcf().clear()

    ### Vary epsilon for DPExpGKGumb (for large and small alpha)
    for a in [0.1, 0.01, 0.001, 0.0001]:
        print("alpha = ", a)
        xs = []
        epsilons = [0.1, 0.5, 1, 5]

        num_trials = 1000
        ## Vary n for large alpha
        ys_exp = [[], []]
        ys_exp_space = [[], []]
        full_ys_exp = [[], []]
        full_ys_exp_space = [[], []]
        q = 0.5
        num_items = math.ceil((1/0.00001))
        print("n = ", num_items)
        unif_data = np.random.uniform(0, 10, num_items)
        normal_data = np.random.normal(5, 1, num_items)

        for eps in epsilons:
            print("eps = ", eps)
            np_gk_unif = GK(a)
            full_np_gk_unif = FullSpace()
            for item in unif_data:
                np_gk_unif.insert(item)
                full_np_gk_unif.insert(item)
            np_gk_normal = GK(a)
            full_np_gk_normal = FullSpace()
            for item in normal_data:
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

        ## DPExpGKGump (vary epsilon for large and small alpha)
        bar_width = 0.25
        labels = [r"Unif(0, 10)", r"Normal(5, 1)"]
        # DPExp (uniform vs. normal)
        pos1 = np.arange(len(ys_exp[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, ys_exp[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Privacy Parameter ($\epsilon$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], epsilons)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_{0}_{1}_{2}.png".format(a, num_items, q))
        np.save("new_vectors/dp_exp_gk_{0}_{1}_{2}_unif.npy".format(a, num_items, q), ys_exp[0])
        np.save("new_vectors/dp_exp_gk_{0}_{1}_{2}_normal.npy".format(a, num_items, q), ys_exp[1])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Accuracy (uniform)
        labels = [r"DPExpGKGumb", r"DPExpFull"]
        pos1 = np.arange(len(ys_exp[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp[0], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Privacy Parameter ($\epsilon$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], epsilons)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_full_unif_{0}_{1}_{2}.png".format(a, num_items, q))
        np.save("new_vectors/dp_exp_gk_full_unif_{0}_{1}_{2}_gk.npy".format(a, num_items, q), ys_exp[0])
        np.save("new_vectors/dp_exp_gk_full_unif_{0}_{1}_{2}_full.npy".format(a, num_items, q), full_ys_exp[0])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Space (uniform)
        labels = [r"DPExpGKGumb", r"DPExpFull"]
        pos1 = np.arange(len(ys_exp_space[0]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp_space[0], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp_space[0], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Privacy Parameter ($\epsilon$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[0]))], epsilons)
        plt.legend(fontsize=18)
        plt.yscale('log')
        plt.savefig("new_images/dp_exp_gk_full_unif_space_{0}_{1}_{2}.png".format(a, num_items, q))
        np.save("new_vectors/dp_exp_gk_full_unif_space_{0}_{1}_{2}_gk.npy".format(a, num_items, q), ys_exp_space[0])
        np.save("new_vectors/dp_exp_gk_full_unif_space_{0}_{1}_{2}_full.npy".format(a, num_items, q), full_ys_exp_space[0])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull (normal)
        pos1 = np.arange(len(ys_exp[1]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp[1], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Privacy Parameter ($\epsilon$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"Absolute Error", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp[1]))], epsilons)
        plt.legend(fontsize=18)
        plt.savefig("new_images/dp_exp_gk_full_normal_{0}_{1}_{2}.png".format(a, num_items, q))
        np.save("new_vectors/dp_exp_gk_full_normal_{0}_{1}_{2}_gk.npy".format(a, num_items, q), ys_exp[1])
        np.save("new_vectors/dp_exp_gk_full_normal_{0}_{1}_{2}_full.npy".format(a, num_items, q), full_ys_exp[1])
        plt.gcf().clear()
        # DPExpGK vs. DPExpFull: Space (normal)
        pos1 = np.arange(len(ys_exp_space[1]))
        pos2 = [x + bar_width for x in pos1]
        plt.bar(pos1, ys_exp_space[1], color='r', width=bar_width, label=labels[0])
        plt.bar(pos2, full_ys_exp_space[1], color='g', width=bar_width, label=labels[1])
        plt.xlabel(r"Privacy Parameter ($\epsilon$)", fontsize=20, fontweight="bold")
        plt.ylabel(r"", fontsize=20, fontweight="bold")
        plt.xticks([r+bar_width/2 for r in range(len(ys_exp_space[1]))], epsilons)
        plt.legend(fontsize=18)
        plt.yscale('log')
        plt.savefig("new_images/dp_exp_gk_full_normal_space_{0}_{1}_{2}.png".format(a, num_items, q))
        np.save("new_images/dp_exp_gk_full_normal_space_{0}_{1}_{2}_gk.npy".format(a, num_items, q), ys_exp_space[1])
        np.save("new_images/dp_exp_gk_full_normal_space_{0}_{1}_{2}_full.npy".format(a, num_items, q), full_ys_exp_space[1])
        plt.gcf().clear()
    
    
