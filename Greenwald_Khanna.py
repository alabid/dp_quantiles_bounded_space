#!/usr/bin/env python3

import numpy as np
import math

# defined constants
MAX_BAND = 1000000000

class GKTuple:
    def __init__(self, v, g, Delta):
        self.v = v
        self.g = g
        self.Delta = Delta

class GK:
    def __init__(self, alpha):
        self.alpha = alpha
        self.n = 0
        self.S = []

    def insert(self, v):
        if self.n % int(1/(2*self.alpha)) == 0:
            self.__compress()
        self.__internal_insert(v)
        self.n += 1

    def quantile(self, phi):
        r = math.ceil(phi*self.n)
        rmin = 0
        for i in range(len(self.S)):
            rmin += self.S[i].g
            rmax = rmin + self.S[i].Delta
            if max(r-rmin, rmax-r) <= self.alpha * self.n:
                print(self.alpha * self.n, r-rmin, rmax-r)
                return (i, self.S[i].v)
        return None

    def dp_exp_mech_quantile(self, phi, eps):
        eps = float(eps)
        r = math.ceil(phi*self.n)
        res_index, res = self.quantile(phi)

        max_value = -np.inf
        max_index = -1
        for i in range(1, len(self.S)):
            score = math.log(self.S[i].g, 2) - eps/2 * math.ceil(abs(i-res_index))
            n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
            if n_score > max_value:
                max_index = i
                max_value = n_score

        return np.random.uniform(low=self.S[max_index-1].v, high=self.S[max_index].v)

    def dp_histogram_quantile(self, phi, eps):
        eps = float(eps)
        hist = []
        cur = 0
        for t in self.S:
            noisy_g = t.g + np.random.laplace(0., 2/eps)
            hist.append((t.v, cur + noisy_g))
            cur = cur + noisy_g
        r = math.ceil(phi*self.n)
        for (v, rank) in hist:
            if r < rank: return v
        return hist[len(hist)-1][0]

    def __compress(self):
        if self.n < 1/(2*self.alpha): return

        max_Delta = 2*self.alpha*self.n
        bands = GK.__band_lookup_table(int(max_Delta))

        i = len(self.S)-2
        while i >= 1:
            if bands[self.S[i].Delta] <= bands[self.S[i+1].Delta]:
                start = i
                g_i_star = self.S[i].g;
                while start >= 2 and bands[self.S[start-1].Delta] < bands[self.S[i].Delta]:
                    start -= 1
                    g_i_star += self.S[start].g
                if g_i_star + self.S[i+1].g + self.S[i+1].Delta < max_Delta:
                    merged = GKTuple(self.S[i+1].v, g_i_star + self.S[i+1].g, self.S[i+1].Delta)
                    self.S = self.S[:start] + [merged] + self.S[2+i:]
                    i = start
            i -= 1

    def __internal_insert(self, v):
        # find index for insertion
        i = 0
        while i < len(self.S) and v >= self.S[i].v:
            i += 1

        # determine Delta
        Delta = 0
        if self.n >= int(1/(2*self.alpha)) and i > 0 and i < len(self.S):
            Delta = math.floor(2*self.alpha*self.n)-1
        
        # form Tuple and add to storage
        self.S = self.S[:i] + [GKTuple(v, 1, Delta)] + self.S[i:]
            
    @staticmethod
    def __band_lookup_table(max_Delta):
        bands = [None]*(max_Delta+1)
        bands[0] = MAX_BAND  # reserve Delta = 0 for its own band
        bands[max_Delta] = 0 # In GK paper, Delta = max_Delta is band 0

        p = math.floor(max_Delta)
        tau = 1
        while tau <= math.ceil(math.log(max_Delta, 2)):
            A = math.pow(2, tau-1)
            B = math.pow(2, tau)
            lower = p - B - (p % B)
            if lower < 0: lower = 0
            upper = p - A - (p % A)
            i = int(lower+1)
            while i <= upper:
                bands[i] = tau
                i += 1

            tau += 1

        return bands

    
