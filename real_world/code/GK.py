#!/usr/bin/env python3

import numpy as np
import math
import bisect

# defined constants
MAX_BAND = 1000000000

class FullSpace:
    def __init__(self, l, u):
        #works with the promise that the data set to be added lies entirely between l and u
        self.n = 0
        self.S = []
        self.l = l
        self.u = u
        self.insert(l)
        self.insert(u)

    def insert(self, v):
        bisect.insort(self.S, v)
        self.n += 1

    def quantile(self, q):
        index = math.floor(self.n*q)
        return index, self.S[index]

    def dp_exp(self, q, eps):
        eps = float(eps)
        r = math.ceil(q*self.n)
        #res_index, res = self.quantile(q)

        max_value = -np.inf
        max_index = -1
        rmin = 0

        #run exp_mech_gumb iterate on tuple element 0 which is always rank 0 element
        u = - abs(0-r)
        score = (eps/2)*u
        n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
        max_index = 0
        max_value = n_score

        for i in range(1, len(self.S)):
            #run exp_mech_gumb iterate on interval between i-1 and i

            if self.S[i]-self.S[i-1] > 0:
                if r>i:
                    u = -abs(r-i)
                else:
                    u = -abs(r-(i-1))
                score = math.log(1E8*(self.S[i]-self.S[i-1]), 2) + (eps/2)*u
                n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
                if n_score > max_value:
                    max_index = i
                    max_value = n_score
                    interval = True

            #run exp_mech_gumb iterate on tuple element i
            u = -abs(i-r)
            score = eps/2*u
            n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
            if n_score > max_value:
                max_index = i
                max_value = n_score
                interval = False

        if interval == True:
            return np.clip(np.random.uniform(low=self.S[max_index-1], high=self.S[max_index]),self.l,self.u)
        else:
            return np.clip(self.S[max_index],self.l,self.u)

    def __getstate__(self):
        return [self.n, self.S]

    def __setstate__(self, state):
        self.n = state[0]
        self.S = state[1]



class GKTuple:
    def __init__(self, v, g, Delta):
        self.v = v
        self.g = g
        self.Delta = Delta

    def __getstate__(self):
        return [self.v, self.g, self.Delta]

    def __setstate__(self, state):
        self.v = state[0]
        self.g = state[1]
        self.Delta = state[2]

class GK:
    def __init__(self, alpha,l,u):
        #works with the promise that the data set to be added lies entirely between l and u
        self.alpha = alpha
        self.n = 0
        self.S = []
        self.l = l
        self.u = u
        self.insert(l)
        self.insert(u)

    def insert(self, v):
        if self.n % int(1/(2*self.alpha)) == 0:
            self.__compress()
        self.__internal_insert(v)
        self.n += 1


    def quantile(self, q):
        r = math.ceil(q*self.n)

        if q == 0 and len(self.S) > 0: return (0, self.S[0].v)

        rmin = 0
        for i in range(len(self.S)):
            rmin += self.S[i].g
            rmax = rmin + self.S[i].Delta
            if max(r-rmin, rmax-r) <= self.alpha * self.n:
                return (rmin, self.S[i].v)
        print("Couldn't find element for quantile ", q, " and rank ", r)
        return None

    def dp_exp(self, q, eps):
        #works with the promise that the data set lies entirely between -10 and 10
        eps = float(eps)
        r = math.ceil(q*self.n)
        res_index, res = self.quantile(q)

        max_value = -np.inf
        max_index = -1
        rmin = 0
        interval = False

        #run exp_mech_gumb iterate on tuple element 0 which is always rank 0 element
        u = - abs(0-r)
        score = (eps/2)*u
        n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
        max_index = 0
        max_value = n_score

        for i in range(1, len(self.S)):
            #run exp_mech_gumb iterate on interval between i-1 and i
            rmax = rmin + self.S[i].g + self.S[i].Delta

            if self.S[i].v-self.S[i-1].v > 0:
                u = -min(abs(rmin-r), abs(rmax-r))
                score = math.log(1E8*(self.S[i].v-self.S[i-1].v), 2) + (eps/2)*u
                n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
                if n_score > max_value:
                    max_index = i
                    max_value = n_score
                    interval = True

            #run exp_mech_gumb iterate on tuple element i
            rmin += self.S[i].g
            u = -min(abs(rmin-r), abs(rmax-r))
            score = eps/2*u
            n_score = score + np.random.gumbel(loc=0.0, scale=1.0)
            if n_score > max_value:
                max_index = i
                max_value = n_score
                interval = False

        if interval == True:
            return np.clip(np.random.uniform(low=self.S[max_index-1].v, high=self.S[max_index].v),self.l,self.u)
        else:
            return np.clip(self.S[max_index].v,self.l,self.u)

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
        #OLD:
        # i = 0
        # while i < len(self.S) and v >= self.S[i].v:
        #     i += 1
        #NEW:


        i = int(0)
        if len(self.S) == 0:
            i = 0
        else:
            i = 0
            l = int(0)
            u = int(len(self.S)-1)
            m = int((u+l)/2)
            if v>=self.S[u].v:
                i = u+1
            elif v<self.S[l].v:
                i = 0
            else:
                while u-l>2:
                    m = int((u+l)/2)
                    if v>self.S[m].v:
                        l = m
                    else:
                        u = m
                i = m


        # determine Delta
        Delta = 0
        if self.n >= int(1/(2*self.alpha)) and i > 0 and i < len(self.S):
            Delta = math.floor(2*self.alpha*self.n)

        # form Tuple and add to storage
        self.S.insert(i, GKTuple(v, 1, Delta))
        #self.S = self.S[:i] + [] + self.S[i:]

    def __getstate__(self):
        return [self.n, self.S]

    def __setstate__(self, state):
        self.n = state[0]
        self.S = state[1]

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
