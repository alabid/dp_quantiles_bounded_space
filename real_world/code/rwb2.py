import numpy as np
import math
import random
import importlib
import statistics
from tqdm import tqdm
from GK import GK, FullSpace
import pickle
import sys

random.seed(2021)

def sPrint(a):
    print(format(a,'.6f'),"\t", end='')

structs = []
approxes = [1E-2, 1E-3, 1E-4, 1E-5]
for approx in approxes:
    f = open('../sketches/1eth_CO_alpha_' + str(approx), 'rb')
    struct = pickle.load(f)
    structs.append(struct)
    print(struct.n, len(struct.S))
    struct.alpha = approx
    struct.l = -1E4
    struct.u= 1E4
    f.close

f = open('../sketches/1fs_eth_CO', 'rb')
fs_taxi = pickle.load(f)
fs_taxi.l = -1E4
fs_taxi.u = 1E4
f.close
print(fs_taxi.n, len(fs_taxi.S))

#1. Taxi dataset error and dataset size varying eps for small and large approximation
#2. Gas sensor dataset error and dataset size varying eps for small and large approximation
#3. Continual observation setting normal and taxi error curve across varying epsilon

num_trials = 100
quantile = 0
results = []
fs_results = []
sizes = []
q = 0.5
epses = [0.1, 0.5, 1, 5]

#record data structure sizes
# for struct in structs:
#     sizes.append(len(struct.S)*3)
# sizes.append(len(fs_taxi.S))

#derive non-private quantile estimates
for struct in structs:
    ind, app = struct.quantile(q)
    #nonPrivate.append([ind, app])
ind, quantile = fs_taxi.quantile(q)
print("The true quantile is ", quantile)
#nonPrivate.append([ind, quantile])

for eps in epses:
    #define lists to record experimental results
    results_eps = []
    for struct in structs:
        results_eps.append([])
    fs_results_eps = []

    print("\n Private queries for eps = ", eps, "\n")
    for j in tqdm(range(num_trials)):
        # Exponential Mechanism
        for i,struct in enumerate(structs):
            #print(struct.dp_exp(q,eps))
            results_eps[i].append(abs(struct.dp_exp(q,eps) - quantile))
        #print(fs_taxi.dp_exp(q,eps))
        fs_results_eps.append(abs(fs_taxi.dp_exp(q,eps) - quantile))

    #concatenate results
    results.append(results_eps)
    fs_results.append(fs_results_eps)

for i,approx in enumerate(approxes):
    print("\n Approximation ", approxes[i], "\n")
    for j,eps in enumerate(epses):
        print("\n", epses[j], "\t", end='')
        sPrint(np.mean(results[j][i]))
        sPrint(np.quantile(results[j][i], 0.1))
        sPrint(np.quantile(results[j][i], 0.9))


print("\n Fullspace \n")
for j,eps in enumerate(epses):
    print("\n", epses[j], "\t", end='')
    sPrint(np.mean(fs_results[j]))
    sPrint(np.quantile(fs_results[j], 0.1))
    sPrint(np.quantile(fs_results[j], 0.9))


dbfile_taxi_results = open('eth_CO_results', 'wb')
pickle.dump(results, dbfile_taxi_results)
#dbfile_taxi_sizes = open('taxi_sizes', 'wb')
#pickle.dump(sizes, dbfile_taxi_sizes)
