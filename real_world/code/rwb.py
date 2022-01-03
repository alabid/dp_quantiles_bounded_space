import csv
from GK import GK, FullSpace
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import pickle

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

df = pd.read_csv("../data/ethylene_CO.csv", skiprows = [0], delim_whitespace = True, error_bad_lines = False, engine='python', usecols = [3])
print(df.info())
print(df.describe())

structs = []
approxes = [1E-2, 1E-3, 1E-4, 1E-5]
for approx in approxes:
    struct = GK(approx, -1E4, 1E4)
    structs.append(struct)
fs_taxi = FullSpace(-1E4, 1E4)

count = 0
for ind in tqdm(df.index):
    for struct in structs:
        struct.insert(float(df.iloc[ind]))
    fs_taxi.insert(float(df.iloc[ind]))
    # count = count+1
    # if count>10000:
    #     break

for struct in structs:
    print("Number of items processed", count, "size of data structure ", struct.n, " approximation ", struct.alpha, "number of items ", len(struct.S))
    f = open('../sketches/eth_CO_alpha_'+str(struct.alpha), 'wb')
    pickle.dump(struct, f)
    f.close()

print("Number of items processed ", count, " size of data structure ", fs_taxi.n, " number of items " , len(fs_taxi.S))
f = open('../sketches/fs_eth_CO', 'wb')
pickle.dump(fs_taxi, f)
f.close()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
