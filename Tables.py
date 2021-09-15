import numpy as np
import math
import random
import importlib
import statistics
import Greenwald_Khanna
importlib.reload(Greenwald_Khanna)
from Greenwald_Khanna import GK, FullSpace
from tqdm import tqdm
import pickle
import sys


if __name__=="__main__":
    check = 0

    if sys.argv[1] == "small":
        dbfile_e = open('Small_approx_error', 'rb')
        dbfile_s = open('Small_approx_sizes', 'rb')
        results = pickle.load(dbfile_e)
        sizes = pickle.load(dbfile_s)
        dbfile_e.close()
        dbfile_s.close()
    elif sys.argv[1] == "large":
        dbfile_e = open('Large_approx_error', 'rb')
        dbfile_s = open('Large_approx_sizes', 'rb')
        results = pickle.load(dbfile_e)
        sizes = pickle.load(dbfile_s)
        dbfile_e.close()
        dbfile_s.close()
    elif sys.argv[1] =="privsmall":
        dbfile_e = open('Small_approx_error_varyeps', 'rb')
        results = pickle.load(dbfile_e)
        dbfile_e.close()
        sizes = []
        check = 1
    elif sys.argv[1] =="privlarge":
        dbfile_e = open('Large_approx_error_varyeps', 'rb')
        results = pickle.load(dbfile_e)
        dbfile_e.close()
        check = 1
        sizes = []

    print("Results",len(results))
    print("Sizes",sizes)

    #data = np.array(results[0][1][2])
    #print(data)
    #data = np.array(results[0][1][3])
    #print(data)

    if check == 0:
        print("Uniform, GKExp")

        for result in results:
            #print(np.quantile(result[1][2],0.05))
            #print("(",result[0],",",format(abs(statistics.mean(result[1][2])-result[1][0]), '.6f'),") +- (",format(np.quantile(result[1][2],0.05)-result[1][0], '.6f'),",",format(np.quantile(result[1][2],0.95)-result[1][0], '.6f'),")")
            errors = np.absolute(result[1][2]-result[1][0]*np.ones(len(result[1][2])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            #print(errors)
            print(result[0],"\t",mean,"\t",lower,"\t",upper)

        print("Uniform, Full")

        for result in results:
            #print(statistics.mean(result[1][3]), " and ", result[1][0])
            #print("(",result[0],",",format(abs(statistics.mean(result[1][3])-result[1][0]), '.6f'),") +- (",format(np.quantile(result[1][3],0.05)-result[1][0], '.6f'),",",format(np.quantile(result[1][3],0.95)-result[1][0], '.6f'),")")
            errors = np.absolute(result[1][3]-result[1][0]*np.ones(len(result[1][3])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print(result[0],"\t",format(abs(statistics.mean(result[1][3])-result[1][0]), '.6f'),"\t",format(np.quantile(result[1][3],0.05)-result[1][0], '.6f'),"\t", format(np.quantile(result[1][3],0.95)-result[1][0], '.6f'))

        print("Normal, GKExp")

        for result in results:
            errors = np.absolute(result[2][2]-result[2][0]*np.ones(len(result[2][2])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print("(",result[0],",",format(abs(statistics.mean(result[2][2])-result[2][0]), '.6f'),") +- (",format(np.quantile(result[2][2],0.05) - result[2][0], '.6f'),",",format(np.quantile(result[2][2],0.95) - result[2][0], '.6f'),")")

        print("Normal, Full")

        for result in results:
            errors = np.absolute(result[2][3]-result[2][0]*np.ones(len(result[2][3])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print("(",result[0],",",format(abs(statistics.mean(result[2][3])-result[2][0]), '.6f'),") +- (",format(np.quantile(result[2][3],0.05) - result[2][0], '.6f'),",",format(np.quantile(result[2][3],0.95) - result[2][0], '.6f'),")")

        print("Uniform distribution sizes")

        for size in sizes:
            #print("(",size[1],",",size[0],")")
            print(size[1],'\t',size[0])

        for size in sizes:
            #print("(",size[1],",",size[1],")")
            print(size[1],'\t',size[1])

        print("Normal distribution sizes")

        for size in sizes:
            #print("(",size[3],",",size[2],")")
            print(size[3],'\t',size[2])

        for size in sizes:
            #print("(",size[3],",",size[3],")")
            print(size[3],'\t',size[3])

        print("Sizes [len(np_gk_unif.S), len(full_np_gk_unif.S),len(np_gk_normal.S), len(full_np_gk_normal.S)] ", sizes)

    elif check==1:
        print("Uniform, GKExp")

        for result in results:
            #print(np.quantile(result[1][2],0.05))
            #print("(",result[0],",",format(abs(statistics.mean(result[1][2])-result[1][0]), '.6f'),") +- (",format(np.quantile(result[1][2],0.05)-result[1][0], '.6f'),",",format(np.quantile(result[1][2],0.95)-result[1][0], '.6f'),")")
            errors = np.absolute(result[1][2]-result[1][0]*np.ones(len(result[1][2])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            #print(errors)
            print(result[0],"\t",mean,"\t",lower,"\t",upper)

        print("Uniform, Full")

        for result in results:
            #print(statistics.mean(result[1][3]), " and ", result[1][0])
            #print("(",result[0],",",format(abs(statistics.mean(result[1][3])-result[1][0]), '.6f'),") +- (",format(np.quantile(result[1][3],0.05)-result[1][0], '.6f'),",",format(np.quantile(result[1][3],0.95)-result[1][0], '.6f'),")")
            errors = np.absolute(result[1][3]-result[1][0]*np.ones(len(result[1][3])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print(result[0],"\t",format(abs(statistics.mean(result[1][3])-result[1][0]), '.6f'),"\t",format(np.quantile(result[1][3],0.05)-result[1][0], '.6f'),"\t", format(np.quantile(result[1][3],0.95)-result[1][0], '.6f'))

        print("Normal, GKExp")

        for result in results:
            errors = np.absolute(result[2][2]-result[2][0]*np.ones(len(result[2][2])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print("(",result[0],",",format(abs(statistics.mean(result[2][2])-result[2][0]), '.6f'),") +- (",format(np.quantile(result[2][2],0.05) - result[2][0], '.6f'),",",format(np.quantile(result[2][2],0.95) - result[2][0], '.6f'),")")

        print("Normal, Full")

        for result in results:
            errors = np.absolute(result[2][3]-result[2][0]*np.ones(len(result[2][3])))
            mean = statistics.mean(errors)
            lower = format(mean-np.quantile(errors,0.05), '.6f')
            upper = format(np.quantile(errors,0.95)-mean, '.6f')
            mean = format(mean, '.6f')
            print(result[0],"\t",mean,"\t",lower,"\t",upper)
            #print("(",result[0],",",format(abs(statistics.mean(result[2][3])-result[2][0]), '.6f'),") +- (",format(np.quantile(result[2][3],0.05) - result[2][0], '.6f'),",",format(np.quantile(result[2][3],0.95) - result[2][0], '.6f'),")")
