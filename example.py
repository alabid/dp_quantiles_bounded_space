#!/usr/bin/env python3

import numpy as np
from Greenwald_Khanna import GK
import random
random.seed(2021)


if __name__ == "__main__":
    alpha = 0.01
    np_gk = GK(alpha)
    for i in range(10000):
        np_gk.insert(random.random())
    for q in np.arange(0, 1, 0.05):
        print("(percentile, approx. value): ", (q, np_gk.quantile(q)))
    print("========================================================")
    alpha = 0.001
    np_gk = GK(alpha)
    for i in range(100000):
        np_gk.insert(random.random())
    for q in np.arange(0, 1, 0.05):
        print("(percentile, approx. value): ", (q, np_gk.quantile(q)))
    print("========================================================")
    alpha = 0.0001
    np_gk = GK(alpha)
    for i in range(100000):
        np_gk.insert(random.random())
    for q in np.arange(0, 1, 0.05):
        print("(percentile, approx. value): ", (q, np_gk.quantile(q)))

