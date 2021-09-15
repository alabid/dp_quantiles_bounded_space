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

dbfile_d_unif = open('Uniform_data400', 'rb')
dbfile_d_norm = open('Normal_data400', 'rb')
total_uniform_data = pickle.load(dbfile_d_unif)
total_normal_data = pickle.load(dbfile_d_norm)

dbfile_d_unif.close()
dbfile_d_norm.close()

if __name__ == "__main__":
    if sys.argv[1] == "small":
        check = 0
        alpha = 1E-4        #approximation parameter
        eps = 1             #privacy parameter
        q = 0.5             #quantile to be queried
        data_step_size = int(1E5)
        dbfile_e = open('Small_approx_error', 'wb')
        dbfile_s = open('Small_approx_sizes', 'wb')
    elif sys.argv[1] == "large":
        check = 0
        alpha = 1E-1         #approximation parameter
        eps = 1             #privacy parameter
        q = 0.5             #quantile to be queried
        data_step_size = int(1E5)
        dbfile_e = open('Large_approx_error', 'wb')
        dbfile_s = open('Large_approx_sizes', 'wb')
    elif sys.argv[1] == "privsmall":
        check = 1
        alpha = 1E-4         #approximation parameter
        epses = [0.1,0.5,1,5]#privacy parameters
        q = 0.5             #quantile to be queried
        data_step_size = int(1E5)
        dbfile_e = open('Small_approx_error_varyeps', 'wb')
    elif sys.argv[1] == "privlarge":
        check = 1
        alpha = 1E-1         #approximation parameter
        epses = [0.1,0.5,1,5]#privacy parameters
        q = 0.5             #quantile to be queried
        data_step_size = int(1E5)
        dbfile_e = open('Large_approx_error_varyeps', 'wb')

    num_trials = 100

    if check==0: #experiment to compare error and size of DPExpGKGumb and DPExpFull
        unif_data = []      #list of raw data distributed uniformly at random
        normal_data = []    #list of raw data distributed as a normal distribution

        results = []
        sizes = []
        index_errors = []

        # initialise data structures
        np_gk_unif = GK(alpha)
        full_np_gk_unif = FullSpace()
        np_gk_normal = GK(alpha)
        full_np_gk_normal = FullSpace()

        print("Dataset size range \n")
        for i in tqdm(range(4)):
            #concatenate random data to that already generated
            unif_data = total_uniform_data[0:(i+1)*data_step_size]
            normal_data = total_normal_data[0:(i+1)*data_step_size]
            print("Also lengths",len(unif_data),len(normal_data))
            #normal_data += np.random.normal(5, 1, data_step_size).tolist()

            #add new data to data structures
            print("Processing uniform data set\n")
            for item in tqdm(unif_data[-data_step_size:]):
                np_gk_unif.insert(item)
                full_np_gk_unif.insert(item)

            print("Processing normal data set\n")
            for item in tqdm(normal_data[-data_step_size:]):
                np_gk_normal.insert(item)
                full_np_gk_normal.insert(item)

            #record data structure sizes
            sizes.append([len(np_gk_unif.S)*3, len(full_np_gk_unif.S),
            len(np_gk_normal.S)*3, len(full_np_gk_normal.S)])

            #derive non-private quantile estimates
            res_index_unif_app, res_unif = np_gk_unif.quantile(q)
            res_index_norm_app, res_normal = np_gk_normal.quantile(q)
            res_index_unif, full_res_unif = full_np_gk_unif.quantile(q)
            res_index_normal, full_res_normal = full_np_gk_normal.quantile(q)

            #define lists to record experimental results
            res_unif_exp = []
            res_normal_exp = []
            res_full_unif_exp = []
            res_full_normal_exp = []

            print("Private queries")
            for j in tqdm(range(num_trials)):
                # Exponential Mechanism
                res_unif_exp.append(np_gk_unif.dp_exp(q, eps))
                res_normal_exp.append(np_gk_normal.dp_exp(q, eps))
                res_full_unif_exp.append(full_np_gk_unif.dp_exp(q, eps))
                res_full_normal_exp.append(full_np_gk_normal.dp_exp(q, eps))

            #concatenate results
            results.append([data_step_size*(i+1),
                            [full_res_unif, res_unif, res_unif_exp, res_full_unif_exp],
                            [full_res_normal, res_normal, res_normal_exp, res_full_normal_exp]])


    elif check==1:#Experiment to compare error of DPExpGKGumb across distributions
        #generate data
        #unif_data = np.random.uniform(0, 10, int(1E5)).tolist()
        #normal_data = np.random.normal(5, 1, int(1E5)).tolist()

        unif_data = total_uniform_data[0:int(1E5)]
        normal_data = total_normal_data[0:int(1E5)]

        results = []

        #initialise data structures
        print("Approx. factor is ", alpha)
        np_gk_unif = GK(alpha)
        full_np_gk_unif = FullSpace()
        np_gk_normal = GK(alpha)
        full_np_gk_normal = FullSpace()

        #add random data to data structures
        print("Processing uniform data set\n")
        for item in tqdm(unif_data[-data_step_size:]):
            np_gk_unif.insert(item)
            full_np_gk_unif.insert(item)

        print("Processing normal data set\n")
        for item in tqdm(normal_data[-data_step_size:]):
            np_gk_normal.insert(item)
            full_np_gk_normal.insert(item)

        #run experiments across values of eps
        for eps in tqdm(epses):
            #derive non-private quantile estimates
            res_index_unif_app, res_unif = np_gk_unif.quantile(q)
            res_index_norm_app, res_normal = np_gk_normal.quantile(q)
            res_index_unif, full_res_unif = full_np_gk_unif.quantile(q)
            res_index_normal, full_res_normal = full_np_gk_normal.quantile(q)

            res_unif_exp = []
            res_normal_exp = []
            res_full_unif_exp = []
            res_full_normal_exp = []

            print("Private queries")
            for j in tqdm(range(num_trials)):
                # Exponential Mechanism
                res_unif_exp.append(np_gk_unif.dp_exp(q, eps))
                res_normal_exp.append(np_gk_normal.dp_exp(q, eps))
                res_full_unif_exp.append(full_np_gk_unif.dp_exp(q, eps))
                res_full_normal_exp.append(full_np_gk_normal.dp_exp(q, eps))

            #concatenate results
            results.append([eps,
                            [full_res_unif, res_unif, res_unif_exp, res_full_unif_exp],
                            [full_res_normal, res_normal, res_normal_exp, res_full_normal_exp]])

    #save results to file
    if check == 0:
        pickle.dump(results, dbfile_e)
        pickle.dump(sizes, dbfile_s)
        dbfile_s.close()
    if check == 1:
        pickle.dump(results, dbfile_e)

    dbfile_e.close()
