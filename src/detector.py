# Find the maximum error in a given input domain
#The implementation of DEMC algorithm
import basic_func as bf
import time,signal
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
import warnings
import math
import numpy as np
import os
from mpmath import *
import itertools
from pygsl.testing import sf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root
from scipy.optimize import minimize

np.set_printoptions(precision=18)

class TimeoutError (RuntimeError):
    pass

def handler (signum, frame):
    raise TimeoutError()

signal.signal (signal.SIGALRM, handler)


def produce_interval(inp_l,bound_l):
    temp_bound = []
    for i,j in zip(inp_l,bound_l):
        a = bf.getulp(i)*1e10
        temp_bound.append([np.max([i-a,j[0]]),np.min([i+a,j[1]])])
    return temp_bound


def produce_interval1(x,k):
    a = bf.getulp(x)*1e12
    return [np.max([x-a,k[0]]),np.min([x+a,k[1]])]

def reduce_x(bound_l,xl):
    xl = list(xl)
    new_x = []
    for i,j in zip(bound_l,xl):
        j = float(j)
        new_x.append((i[0]+i[1]+(i[1]-i[0])*math.sin(j))/2.0)
    return tuple(new_x)

def generate_x(bound_l,xl):
    xl = list(xl)
    new_x = []
    for i,j in zip(bound_l,xl):
        temp = (j - i[0]+(j - i[1])) / (i[1] - i[0])
        if temp<0:
            temp = np.max([-1,temp])
        else:
            temp = np.min([1,temp])
        new_x.append(float(asin(temp)))
    return tuple(new_x)

def reduce_x1(a,b,x):
    x = float(x)
    return (a+b+(b-a)*math.sin(x))/2.0

def DEMC(rf,pf,inpdm,fnm,limit_n,limit_time):
    st = time.time()
    file_name = "../experiments/detecting_results/DEMC/" + fnm
    count = 0
    final_max = 0.0
    final_x = 0.0
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fdistribution_partition(inpdm[0], inpdm[1])
    glob_fitness_con = np.frompyfunc(lambda x: bf.fitness_fun1(rf, pf, x), 1, 1)
    glob_fitness_real = np.frompyfunc(lambda x: bf.fitness_fun(rf, pf, x), 1, 1)
    try:
        print "Detecting possible maximum error by DEMC algorithm"
        signal.alarm(limit_time)
        while(count<limit_n):
            temp_st=time.time()
            count1 = 0
            count2 = 0
            rand_seed = bf.rd_seed[count]
            np.random.seed(rand_seed)
            res_l = []
            for k in dom_l:
                temp_max = 0.0
                temp_x = 0.0
                res = differential_evolution(glob_fitness_con, popsize=15, bounds=[k], polish=False, strategy='best1bin')
                x = res.x[0]
                count2 = count2+res.nfev
                err = 1.0/glob_fitness_real(float(x))
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                temp = [temp_max, temp_x, k]
                res_l.append(temp)
            t1 = time.time() - temp_st
            # print t1
            res_l = sorted(res_l, reverse=True)
            temp_max = res_l[0][0]
            temp_x = res_l[0][1]
            bound = res_l[0][2]
            res_lr = []
            s_len = np.min([len(res_l), 10])
            # print res_l[0:s_len]
            # glob_fitness_real_temp = lambda x: x*x
            distan_two_search_x = 1.0
            minimizer_kwargs = {"method":"Nelder-Mead"}
            for j in res_l[0:s_len]:
                gen_l = produce_interval1(j[1], j[2])
                glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, reduce_x1(gen_l[0],gen_l[1],x))
                # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
                x = math.asin((2*j[1]-gen_l[0]-gen_l[1])/(gen_l[1]-gen_l[0]))
                # x = j[1]
                res = basinhopping(glob_fitness_real_temp,x,stepsize=bf.getulp(x)*1e10,minimizer_kwargs=minimizer_kwargs,niter_success=10,niter=200)
                count1 = count1 + res.nfev
                # x = res.x[0]
                x = reduce_x1(gen_l[0],gen_l[1],res.x[0])
                err = 1.0/res.fun
                temp = [err, x, gen_l]
                res_lr.append(temp)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                    bound = j[2]
                    distan_two_search_x = bf.getUlpError(temp_x, res_l[0][1])
            t2 = time.time() - temp_st
            temp_l = [temp_max,temp_x,bound,t2,count1,count2,rand_seed,count,t2-t1,distan_two_search_x]
            # print temp_l
            # print distan_two_search_x
            final_count1 = final_count1+count1
            final_count2 = final_count2+count2
            record_res_l.append(temp_l)
            count = count + 1
            distan_two_search_x_final = 1.0
            if temp_max>final_max:
                final_max=temp_max
                final_x = temp_x
                final_bound = bound
                distan_two_search_x_final = distan_two_search_x
                # distan_two_search_x = bf.getUlpError(final_x, res_l[0][1])
        final_time = time.time()-st
        bf.output_err(record_res_l, file_name, fnm)
        print "%.20e" % final_x
        print "%.20e" % final_max
        return [final_max, final_x, final_bound, final_time,count,final_count1,final_count2,distan_two_search_x_final]
    except TimeoutError:
        final_time = time.time() - st
        bf.output_err(record_res_l,file_name,fnm)
        return [final_max, final_x, final_bound, final_time,count,final_count1,final_count2,distan_two_search_x_final]




def DEMC_pure(rf,pf,inpdm,limit_n,limit_time):
    st = time.time()
    count = 0
    final_max = 0.0
    final_x = 0.0
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fdistribution_partition(inpdm[0], inpdm[1])
    glob_fitness_con = np.frompyfunc(lambda x: bf.fitness_fun1(rf, pf, x), 1, 1)
    glob_fitness_real = np.frompyfunc(lambda x: bf.fitness_fun(rf, pf, x), 1, 1)
    try:
        # print "Detecting possible maximum error by DEMC algorithm"
        signal.alarm(limit_time)
        while(count<limit_n):
            temp_st=time.time()
            count1 = 0
            count2 = 0
            # rand_seed = bf.rd_seed[count]
            rand_seed = np.random.randint(1,1000000)
            np.random.seed(rand_seed)
            res_l = []
            for k in dom_l:
                temp_max = 0.0
                temp_x = 0.0
                res = differential_evolution(glob_fitness_con, popsize=15, bounds=[k], polish=True, strategy='best1bin')
                x = res.x[0]
                count2 = count2+res.nfev
                err = 1.0/glob_fitness_real(float(x))
                if err >= temp_max:
                    temp_max = err
                    temp_x = x
                temp = [temp_max, temp_x, k]
                res_l.append(temp)
            t1 = time.time() - temp_st
            # print t1
            res_l = sorted(res_l, reverse=True)
            temp_max = res_l[0][0]
            temp_x = res_l[0][1]
            bound = res_l[0][2]
            res_lr = []
            s_len = np.min([len(res_l), 10])
            # print res_l[0:s_len]
            # glob_fitness_real_temp = lambda x: x*x
            distan_two_search_x = 1.0
            minimizer_kwargs = {"method":"Nelder-Mead"}
            for j in res_l[0:s_len]:
                # gen_l = j[2]
                gen_l = produce_interval1(j[1], j[2])
                glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, reduce_x1(gen_l[0],gen_l[1],x))
                # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
                x = math.asin((j[1]-gen_l[0]+(j[1]-gen_l[1]))/(gen_l[1]-gen_l[0]))
                # x = j[1]
                res = basinhopping(glob_fitness_real_temp,x,stepsize=bf.getulp(x)*1e10,minimizer_kwargs=minimizer_kwargs,niter_success=10,niter=200)
                count1 = count1 + res.nfev
                # x = res.x[0]
                x = reduce_x1(gen_l[0],gen_l[1],res.x[0])
                err = 1.0/res.fun
                temp = [err, x, gen_l]
                res_lr.append(temp)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                    bound = j[2]
                    distan_two_search_x = bf.getUlpError(temp_x, res_l[0][1])
            t2 = time.time() - temp_st
            temp_l = [temp_max,temp_x,bound,t2,count1,count2,rand_seed,count,t2-t1,distan_two_search_x]
            # print temp_l
            # print distan_two_search_x
            final_count1 = final_count1+count1
            final_count2 = final_count2+count2
            record_res_l.append(temp_l)
            count = count + 1
            distan_two_search_x_final = 1.0
            if temp_max>final_max:
                final_max=temp_max
                final_x = temp_x
                final_bound = bound
                distan_two_search_x_final = distan_two_search_x
                # distan_two_search_x = bf.getUlpError(final_x, res_l[0][1])
        final_time = time.time()-st
        return [final_max, final_x, final_bound, final_time,count,final_count1,final_count2,distan_two_search_x_final]
    except TimeoutError:
        final_time = time.time() - st
        return [final_max, final_x, final_bound, final_time,count,final_count1,final_count2,distan_two_search_x_final]




def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound

def fake_rf(rf,inp):
    return math.fabs(float(rf(*inp)))

def root_find_rf(rf,pf,point,step):
    try:
        glob_fitness_con = lambda x: fake_rf(rf, x)
        new_bound = generate_bound(point, step / 100.0)
        # print new_bound
        res = differential_evolution(glob_fitness_con, popsize=15, bounds=new_bound, polish=True, strategy='best1bin')
        # print res
        # print pf(*(res.x))
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point


def get_point_distance(p1,p2):
    dis_l = []
    for i,j in zip(p1,p2):
        dis_l.append(bf.getUlpError(i,j))
    return dis_l
def DDEMC(rf,pf,inpdm,fnm,limit_n,limit_time,id):
    st = time.time()
    file_name = "../experiments/detecting_results/DDEMC"+str(id)+ "v/" + fnm
    count = 0
    final_max = 0.0
    final_x = []
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fpartition(inpdm)
    glob_fitness_con = lambda x: bf.mfitness_fun1(rf, pf, x)
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    try:
        print "Detecting possible maximum error by DDEMC algorithm"
        signal.alarm(limit_time)
        while(count<limit_n):
            temp_st=time.time()
            count1 = 0
            count2 = 0
            # rand_seed = np.random.randint(1,1000000)
            # rand_seed = bf.rd_seed[count+5]
            rand_seed = bf.rd_seed[count]
            np.random.seed(rand_seed)
            res_l = []
            pec_count = 0
            len_dom = len(dom_l)
            # print len_dom
            for k in dom_l:
                temp_max = 0.0
                temp_x = []
                res = differential_evolution(glob_fitness_con, popsize=15, bounds=k, polish=False, strategy='best1bin')
                x = res.x
                count2 = count2 + res.nfev
                err = 1.0 / glob_fitness_real(x)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                temp = [temp_max, list(temp_x), k]
                res_l.append(temp)
                pec_count = pec_count + 1
                pec_dom = int(pec_count*100.0/float(len_dom))
                # if (pec_dom >= 10) & (math.fmod(pec_dom,10) == 0):
                    # print repr(pec_dom)+"%"
            t1 = time.time() - temp_st
            # print t1
            res_l.sort()
            res_l.reverse()
            temp_max = res_l[0][0]
            temp_x = res_l[0][1]
            bound = res_l[0][2]
            res_lr = []
            s_len = np.min([len(res_l), 10])
            # print res_l[0:s_len]
            # glob_fitness_real_temp = lambda x: x*x
            distan_two_search_x = [1.0]*len(temp_x)
            minimizer_kwargs = {"method": "Nelder-Mead"}
            for j in res_l[0:s_len]:
                # x = j[1]
                gen_l = produce_interval(j[1],j[2])
                glob_fitness_real_temp = lambda z: bf.mfitness_fun(rf, pf, reduce_x(gen_l, z))
                # # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
                x = generate_x(gen_l,j[1])
                res = basinhopping(glob_fitness_real_temp,x,minimizer_kwargs=minimizer_kwargs,niter_success=10, niter=200)
                count1 = count1 + res.nfev
                x = reduce_x(gen_l, res.x)
                err = 1.0 / res.fun
                temp = [err, x]
                res_lr.append(temp)
                temp_distan_two_search_x = get_point_distance(x, j[1])
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                    bound = j[2]
                    distan_two_search_x = temp_distan_two_search_x
            t2 = time.time() - temp_st
            # print distan_two_search_x
            temp_l = [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t2-t1,distan_two_search_x]
            # print temp_l
            final_count1 = final_count1 + count1
            final_count2 = final_count2 + count2
            record_res_l.append(temp_l)
            count = count + 1
            distan_two_search_x_final = [1.0]*len(temp_x)
            if temp_max > final_max:
                final_max = temp_max
                final_x = temp_x
                final_bound = bound
                distan_two_search_x_final = distan_two_search_x
                # print distan_two_search_x_final
        final_time = time.time() - st
        bf.output_err(record_res_l, file_name, fnm)
        # print distan_two_search_x_final
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2,distan_two_search_x_final]
    except TimeoutError:
        final_time = time.time() - st
        bf.output_err(record_res_l, file_name, fnm)
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2,distan_two_search_x_final]



def DDEMC_para(rf,pf,count,inpdm,fnm,limit_time,id):
    st = time.time()
    file_name = "../experiments/detecting_results/DDEMC"+str(id)+ "v/" + fnm
    final_max = 0.0
    final_x = []
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fpartition(inpdm)
    glob_fitness_con = lambda x: bf.mfitness_fun1(rf, pf, x)
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    try:
        print "Detecting possible maximum error by DDEMC algorithm"
        signal.alarm(limit_time)
        temp_st=time.time()
        count1 = 0
        count2 = 0
        # rand_seed = np.random.randint(1,1000000)
        rand_seed = bf.rd_seed[count]
        np.random.seed(rand_seed)
        res_l = []
        pec_count = 0
        len_dom = len(dom_l)
        # print len_dom
        for k in dom_l:
            temp_max = 0.0
            temp_x = []
            res = differential_evolution(glob_fitness_con, popsize=15, bounds=k, polish=False, strategy='best1bin')
            x = res.x
            count2 = count2 + res.nfev
            err = 1.0 / glob_fitness_real(x)
            if err > temp_max:
                temp_max = err
                temp_x = x
            temp = [temp_max, list(temp_x), k]
            res_l.append(temp)
            pec_count = pec_count + 1
            pec_dom = int(pec_count*100.0/float(len_dom))
            # if (pec_dom >= 10) & (math.fmod(pec_dom,10) == 0):
                # print repr(pec_dom)+"%"
        t1 = time.time() - temp_st
        # print t1
        res_l.sort()
        res_l.reverse()
        temp_max = res_l[0][0]
        temp_x = res_l[0][1]
        bound = res_l[0][2]
        res_lr = []
        s_len = np.min([len(res_l), 10])
        # print res_l[0:s_len]
        # glob_fitness_real_temp = lambda x: x*x
        distan_two_search_x = [1.0]*len(temp_x)
        minimizer_kwargs = {"method": "Nelder-Mead"}
        for j in res_l[0:s_len]:
            # x = j[1]
            gen_l = produce_interval(j[1],j[2])
            glob_fitness_real_temp = lambda z: bf.mfitness_fun(rf, pf, reduce_x(gen_l, z))
            # # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
            x = generate_x(gen_l,j[1])
            res = basinhopping(glob_fitness_real_temp,x,minimizer_kwargs=minimizer_kwargs,niter_success=10, niter=200)
            count1 = count1 + res.nfev
            x = reduce_x(gen_l, res.x)
            err = 1.0 / res.fun
            temp = [err, x]
            res_lr.append(temp)
            temp_distan_two_search_x = get_point_distance(x, j[1])
            if err > temp_max:
                temp_max = err
                temp_x = x
                bound = j[2]
                distan_two_search_x = temp_distan_two_search_x
        t2 = time.time() - temp_st
        # print distan_two_search_x
        # temp_l = [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t2-t1,distan_two_search_x]
        # print temp_l
        # final_count1 = final_count1 + count1
        # final_count2 = final_count2 + count2
        # record_res_l.append(temp_l)
        # count = count + 1
        # distan_two_search_x_final = [1.0]*len(temp_x)
        # if temp_max > final_max:
        #     final_max = temp_max
        #     final_x = temp_x
        #     final_bound = bound
        #     distan_two_search_x_final = distan_two_search_x
        #         # print distan_two_search_x_final
        # final_time = time.time() - st
        # bf.output_err(record_res_l, file_name, fnm)
        # print distan_two_search_x_final
        # [final_max, final_x, final_bound, final_time, count, final_count1, final_count2, distan_two_search_x_final]
        return [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t2-t1,distan_two_search_x]
    except TimeoutError:
        final_time = time.time() - st
        # bf.output_err(record_res_l, file_name, fnm)
        return [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t2-t1,distan_two_search_x]


def get_mid(k):
    temp_x = []
    for i in k:
        temp_x.append(np.random.uniform(i[0], i[1], 1))
    return temp_x

def fake_pf(pf):
    return lambda x: pf(*x)

def DDEMC_root(rf,pf,inpdm,fnm,limit_n,limit_time):
    st = time.time()
    file_name = "../experiments/detecting_results/DDEMC4v_1/" + fnm
    if not os.path.exists("../experiments/detecting_results/DDEMC4v_1/"):
        os.makedirs("../experiments/detecting_results/DDEMC4v_1/")
    count = 0
    final_max = 0.0
    final_x = []
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fpartition(inpdm)
    glob_fitness_con = lambda x: bf.mfitness_fun1(rf, pf, x)
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    try:
        print "Detecting possible maximum error by DDEMC algorithm"
        signal.alarm(limit_time)
        while(count<limit_n):
            temp_st=time.time()
            count1 = 0
            count2 = 0
            rand_seed = bf.rd_seed[count]
            np.random.seed(rand_seed)
            res_l = []
            pec_count = 0
            len_dom = len(dom_l)
            # print len_dom
            for k in dom_l:
                temp_max = 0.0
                temp_x = []
                # res = differential_evolution(glob_fitness_con, popsize=15, bounds=k, polish=False, strategy='best1bin')
                mid_p = get_mid(k)
                fake_pf = lambda x: pf(*x)
                res = minimize(fake_pf,mid_p,bounds = k)
                print res
                x = res.x
                count2 = count2 + res.nfev
                err = 1.0 / glob_fitness_real(x)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                temp = [temp_max, list(temp_x), k]
                res_l.append(temp)
                pec_count = pec_count + 1
                # pec_dom = int(pec_count*100.0/float(len_dom))
                # if (pec_dom >= 10) & (math.fmod(pec_dom,10) == 0):
                    # print repr(pec_dom)+"%"
            t1 = time.time() - temp_st
            # print t1
            res_l.sort()
            res_l.reverse()
            temp_max = res_l[0][0]
            temp_x = res_l[0][1]
            bound = res_l[0][2]
            res_lr = []
            s_len = np.min([len(res_l), 10])
            # print res_l[0:s_len]
            # glob_fitness_real_temp = lambda x: x*x
            minimizer_kwargs = {"method": "Nelder-Mead"}
            for j in res_l[0:s_len]:
                # x = j[1]
                gen_l = produce_interval(j[1],j[2])
                glob_fitness_real_temp = lambda z: bf.mfitness_fun(rf, pf, reduce_x(gen_l, z))
                # # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
                x = generate_x(gen_l,j[1])
                res = basinhopping(glob_fitness_real_temp,x, minimizer_kwargs=minimizer_kwargs,
                                   niter_success=10, niter=200)
                count1 = count1 + res.nfev
                x = reduce_x(gen_l, res.x)
                err = 1.0 / res.fun
                temp = [err, x]
                res_lr.append(temp)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                    bound = j[2]
            t2 = time.time() - temp_st
            temp_l = [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t1]
            # print temp_l
            final_count1 = final_count1 + count1
            final_count2 = final_count2 + count2
            record_res_l.append(temp_l)
            count = count + 1
            if temp_max > final_max:
                final_max = temp_max
                final_x = temp_x
                final_bound = bound
        final_time = time.time() - st
        bf.output_err(record_res_l, file_name, fnm)
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2]
    except TimeoutError:
        final_time = time.time() - st
        bf.output_err(record_res_l, file_name, fnm)
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2]

def DDEMC_pure(rf,pf,inpdm,limit_n,limit_time):
    st = time.time()
    count = 0
    final_max = 0.0
    final_x = []
    final_count1 = 0
    final_count2 = 0
    final_bound = []
    record_res_l = []
    dom_l = bf.fpartition(inpdm)
    glob_fitness_con = lambda x: bf.mfitness_fun1(rf, pf, x)
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    try:
        signal.alarm(limit_time)
        while(count<limit_n):
            temp_st=time.time()
            count1 = 0
            count2 = 0
            rand_seed = np.random.randint(1,1000000)
            np.random.seed(rand_seed)
            res_l = []
            temp_max = 0.0
            for k in dom_l:
                temp_x = []
                res = differential_evolution(glob_fitness_con, popsize=15, bounds=k, polish=False, strategy='best1bin')
                x = res.x
                count2 = count2 + res.nfev
                err = 1.0 / glob_fitness_real(x)
                if err > temp_max:
                    temp_max = err
                    temp_x = x
                temp = [temp_max, list(temp_x), k]
                res_l.append(temp)
            t1 = time.time() - temp_st
            # print t1
            res_l.sort()
            res_l.reverse()
            temp_max = res_l[0][0]
            temp_x = res_l[0][1]
            bound = res_l[0][2]
            res_lr = []
            s_len = np.min([len(res_l), 20])
            # print res_l[0:s_len]
            # glob_fitness_real_temp = lambda x: x*x
            minimizer_kwargs = {"method": "Nelder-Mead"}
            for j in res_l[0:s_len]:
                # x = j[1]
                gen_l = produce_interval(j[1],j[2])
                glob_fitness_real_temp = lambda z: bf.mfitness_fun(rf, pf, reduce_x(gen_l, z))
                # # glob_fitness_real_temp = lambda x: bf.fitness_fun(rf, pf, x)
                x = generate_x(gen_l,j[1])
                res = basinhopping(glob_fitness_real_temp,x,minimizer_kwargs=minimizer_kwargs,
                                   niter_success=10, niter=200)
                count1 = count1 + res.nfev
                x = res.x
                err = 1.0 / res.fun
                temp = [err, x]
                res_lr.append(temp)
                if err > temp_max:
                    temp_max = err
                    temp_x = reduce_x(gen_l,x)
                    bound = j[2]
            t2 = time.time() - temp_st
            temp_l = [temp_max, temp_x, bound, t2, count1, count2, rand_seed, count, t1]
            # print temp_l
            final_count1 = final_count1 + count1
            final_count2 = final_count2 + count2
            record_res_l.append(temp_l)
            count = count + 1
            if temp_max > final_max:
                final_max = temp_max
                final_x = temp_x
                final_bound = bound
        final_time = time.time() - st
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2]
    except TimeoutError:
        final_time = time.time() - st
        return [final_max, final_x, final_bound, final_time, count, final_count1, final_count2]
