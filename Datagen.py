import numpy as np
import math
import random
import importlib
import statistics
from tqdm import tqdm
from Greenwald_Khanna import GK, FullSpace
import pickle
import sys

random.seed(2021)

#1. Uniform and normal distribution, DPExpGKGumb's absolute error vs data stream length n = 100k, 200k, 300k, 400k, alpha =1E-1
#2. DPExpGKGumb & DPExpFull, absolute error (a) & space (b) vs data stream length n = 100k, 200k, 300k, 400k, alpha = 1E-1, uniform data,
#3. DPExpGKGumb & DPExpFull, absolute error (a) & space (b) vs data stream length n = 100k, 200k, 300k, 400k, alpha = 1E-1, normal data

check = 0

if __name__ == "__main__":

    alpha = 1E-4
    #alpha_large = 1E-1

    np_gk_unif = GK(alpha)
    full_np_gk_unif = FullSpace()
    np_gk_norm = GK(alpha)
    full_np_gk_norm = FullSpace()

    data_size = int(4E2)
    data_step_size = int(1E2)

    dbfile_d_unif = open('Uniform_data'+str(data_size), 'ab')
    dbfile_d_norm = open('Normal_data'+str(data_size), 'ab')

    uniform = np.random.uniform(0, 10, data_size).tolist()
    normal = np.random.normal(5, 1, data_size).tolist()

    for i in range(4):
        print("Processing uniform data set\n")
        for item in tqdm(uniform[-data_step_size:]):
            np_gk_unif.insert(item)
            full_np_gk_unif.insert(item)

        dbfile_np_gk_unif = open("np_gk_unif_" + str((i+1)*data_step_size), 'ab')
        pickle.dump(np_gk_unif, dbfile_np_gk_unif)
        dbfile_np_gk_unif.close()
        dbfile_full_np_gk_unif = open("full_np_gk_unif_" + str((i+1)*data_step_size), 'ab')
        pickle.dump(full_np_gk_unif, dbfile_full_np_gk_unif)
        dbfile_full_np_gk_unif.close()

        print("Processing normal data set\n")
        for item in tqdm(normal[-data_step_size:]):
            np_gk_norm.insert(item)
            full_np_gk_norm.insert(item)

        dbfile_np_gk_norm = open("np_gk_norm_" + str((i+1)*data_step_size), 'ab')
        pickle.dump(np_gk_norm, dbfile_np_gk_norm)
        dbfile_np_gk_norm.close()
        dbfile_full_np_gk_norm = open("full_np_gk_norm_" + str((i+1)*data_step_size), 'ab')
        pickle.dump(full_np_gk_norm, dbfile_full_np_gk_norm)
        dbfile_full_np_gk_norm.close()


    pickle.dump(uniform, dbfile_d_unif)
    pickle.dump(normal, dbfile_d_norm)
    dbfile_d_unif.close()
    dbfile_d_norm.close()
