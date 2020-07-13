# basic functions for many methods in the tool
import math
import numpy as np
from mpmath import *
import struct
import itertools
import xlwt
from scipy.misc import derivative
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from pygsl.testing import sf
from scipy.optimize import differential_evolution
import pickle
import time
from xlutils.copy import copy
import xlrd

# the random seed pool for people can repeating our experiments on detecting high floating-ponit errors,
# users can change it by generating new random number.
rd_seed = [82547955,18805512,51059660,67951510,96673401,92529168,43798981,\
           77041498,99700547,46432894,47637490,44611437,39774397,41271573,\
           4645333,25792865,3175680,69902962,60120588,56215621,86667354,\
           74905104,94207956,38027412,8741397,12937909,1370902,43545965,\
           47452337,66102720,86237691,61455401,14149645,39284815,92388247,\
           55354625,59213294,89102079,21502948,94527829,91610400,26056364,\
           41300704,79553483,78203397,20052848,70074407,21862765,17505322,\
           49703457,51989781,63982162,54105705,73199553,27712144,14028450,\
           57895331,88862329,99534636,50330848,14753501,65359048,62069927,\
           73549214,16226155,56551595,14029581,12154538,38929924,19960712,\
           85095147,72225765,25708618,28371123,55480794,21371248,7507139,\
           80070951,61317037,83546642,41962927,83218340,4355823,6686600,\
           18774345,84066402,41611436,22633123,45560493,11142569,37733241,\
           67382830,56461630,59719238,65235752,6412769,69435498,94266224,2120562,14276357]

warnings.filterwarnings("ignore")
def getulp(x):
    x = float(x)
    k = frexp(x)[1]-1
    if x == 0.0:
        return pow(2, -1074)
    if (k<1023)&(k>-1022):
        return pow(2,k-52)
    else:
        return pow(2,-1074)


def plot_3D_error(X,Y,Z1,Z2,U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(111, projection='3d')
    # ax.plot(X, Y, Z,'.', label='parametric curve')
    # ax2.plot(Xl3, Yl3, U2,'.', label='parametric curve')
    ax.plot(X, Y, Z1, '.')
    ax.plot(X, Y, Z2, '.')
    # Zeo = []
    # for i in U:
    #     Zeo.append(0)
    # ax.plot(X, Y, U, '.')
    # ax.plot(X, Y, Zeo, '.')
    # ax.plot(X, Yl, label='line approximation')
    # ax.plot(X, Y, label='line approximation')
    # ax2.plot(Xl2, Yl, label='line approximation')
    ax.legend()
    plt.show()

# partition the input domain according to floating-point distribution
def fdistribution_partition(in_min, in_max):
    tmp_l = []
    a = np.frexp(in_min)
    b = np.frexp(in_max)
    tmp_j = 0
    if (in_min < 0)&(in_max > 0):
        if in_min >= -1.0:
            tmp_l.append([in_min, 0])
        else:
            for i in range(1, a[1]+1):
                tmp_i = np.ldexp(-0.5, i)
                tmp_l.append([tmp_i, tmp_j])
                tmp_j = tmp_i
            if in_min != tmp_j:
                tmp_l.append([in_min, tmp_j])
        tmp_j = 0
        if in_max <= 1.0:
            tmp_l.append([0, in_max])
        else:
            for i in range(1, b[1]+1):
                tmp_i = np.ldexp(0.5, i)
                tmp_l.append([tmp_j, tmp_i])
                tmp_j = tmp_i
            if in_max != tmp_j:
                tmp_l.append([tmp_j, in_max])
    if (in_min < 0) & (0 >= in_max):
        if in_min >= -1:
            tmp_l.append([in_min, in_max])
            return tmp_l
        else:
            if in_max > -1:
                tmp_l.append([-1, in_max])
                tmp_j = -1.0
                for i in range(2, a[1] + 1):
                    tmp_i = np.ldexp(-0.5, i)
                    tmp_l.append([tmp_i, tmp_j])
                    tmp_j = tmp_i
                if in_min != tmp_j:
                    tmp_l.append([in_min, tmp_j])
            else:
                if a[1] == b[1]:
                    tmp_l.append([in_min, in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(-0.5, b[1]+1)
                    tmp_l.append([tmp_j, in_max])
                    if tmp_j != in_min:
                        for i in range(b[1]+2, a[1]+1):
                            tmp_i = np.ldexp(-0.5, i)
                            tmp_l.append([tmp_i, tmp_j])
                            tmp_j = tmp_i
                        if in_min != tmp_j:
                            tmp_l.append([in_min, tmp_j])
    if (in_min >= 0) & (in_max > 0):
        if in_max <= 1:
            tmp_l.append([in_min, in_max])
            return tmp_l
        else:
            if in_min < 1:
                tmp_l.append([in_min, 1])
                tmp_j = 1.0
                for i in range(2, b[1] + 1):
                    tmp_i = np.ldexp(0.5, i)
                    tmp_l.append([tmp_j, tmp_i])
                    tmp_j = tmp_i
                if in_max != tmp_j:
                    tmp_l.append([tmp_j, in_max])
            else:
                if a[1] == b[1]:
                    tmp_l.append([in_min, in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(0.5, a[1]+1)
                    tmp_l.append([in_min, tmp_j])
                    if tmp_j != in_max:
                        for i in range(a[1]+2, b[1]+1):
                            tmp_i = np.ldexp(0.5, i)
                            tmp_l.append([tmp_j, tmp_i])
                            tmp_j = tmp_i
                        if in_max != tmp_j:
                            tmp_l.append([tmp_j, in_max])
    return tmp_l
def fpartition(input_domain):
    l_var = []
    for i in input_domain:
        for j in i:
            tmp_l = fdistribution_partition(j[0], j[1])
            l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs


def bound_partition(bound):
    db = bound[0]
    ub = bound[1]
    bound_l = []
    tmp_b = bound[0]
    if getulp(db) != getulp(ub):
        sdb = np.sign(np.frexp(db)[0])
        edb = np.frexp(db)[1]
        sedb = np.sign(bound[0])
        print edb
        print sedb
        while (1):
            tmp_0 = 0.5 * sdb
            # print
            # print tmp_0
            # print edb
            if sedb > 0:
                edb = int(edb + sedb)
            up_b = np.ldexp(tmp_0, edb)
            if sedb < 0:
                edb = int(edb + sedb)
            print up_b
            if edb < -1022:
                print up_b
                edb = int(edb - sedb)
                print edb
                bound_l.append([tmp_b, 0])
                sdb = -1 * sdb
                sedb = -1 * sedb
            if up_b > bound[1]:
                if tmp_b != bound[1]:
                    bound_l.append([tmp_b, bound[1]])
                break
            if tmp_b != up_b:
                if getulp(tmp_b)==getulp(up_b):
                    bound_l.append([tmp_b, up_b])
                    ulp_b = getulp(up_b + getulp(up_b))
                    tmp_b = up_b + ulp_b
                else:
                    bound_l.append([tmp_b, up_b-getulp(tmp_b)])
                    tmp_b = up_b
    else:
        bound_l.append(bound)
    return bound_l

def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]

def longBitsToFloat(bits):
	return struct.unpack('d', struct.pack('Q', bits))[0]

def save_line_list(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)

def getMidDistance(a,b):
    ia = floatToRawLongBits(np.abs(a))
    ib = floatToRawLongBits(np.abs(b))
    zo = floatToRawLongBits(0)
    if sign(a)!=sign(b):
        res = abs(ib-zo)+abs(ia-zo)
    else:
        res = abs(ib-ia)
    if (sign(a)==sign(b))&(ib == ia):
        return 0
    return int(res)

def getUlpError(a,b):
    try:
        ia = floatToRawLongBits(np.abs(a))
        ib = floatToRawLongBits(np.abs(b))
        zo = floatToRawLongBits(0)
        if sign(a)!=sign(b):
            res = abs(ib-zo)+abs(ia-zo)
        else:
            res = abs(ib-ia)
        return int(res+1)
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError):
        return 1.0

def getPointBound(point,step):
    ini_bound = []
    for i in point:
        if i > 0:
            ini_bound.append([longBitsToFloat(floatToRawLongBits(i)-int(step)), longBitsToFloat(floatToRawLongBits(i)+int(step))])
        else:
            ini_bound.append([longBitsToFloat(floatToRawLongBits(i) + int(step)),longBitsToFloat(floatToRawLongBits(i) - int(step))])
    return ini_bound


def get_next_point(point,step,sign):
    if point>0:
        return longBitsToFloat(floatToRawLongBits(point) + sign*int(step))
    else:
        return longBitsToFloat(floatToRawLongBits(point) - sign * int(step))
# print get_next_point(-1,1e16,1)
def getFPNum(a,b):
    try:
        ia = floatToRawLongBits(np.abs(a))
        ib = floatToRawLongBits(np.abs(b))
        zo = floatToRawLongBits(0)
        if sign(a)!=sign(b):
            res = abs(ib-zo)+abs(ia-zo)
        else:
            res = abs(ib-ia)
        return int(res+1)
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError):
        return 1.0

def partial_derivative(func,b, var=0, point=[]):
    try:
        args = point[:]
        # diff = math.fabs(args[var] * 2.2204460492503131e-15)
        diff = getulp(args[var])*10
        def wraps(x):
            args[var] = x
            return func(*args)
        # return derivative(wraps, point[var],dx = 1e-8)
        # print (wraps(args[var] + diff) - b)
        # print diff
        # print point
        # print var
        return (wraps(args[var] + diff) - b) / diff
    except (ValueError, ZeroDivisionError, OverflowError, Warning, RuntimeWarning, TypeError):
        return 1.0

# estimate the condition number, do not use the diff in mpmath which is very slow.
def mcondition(a,b,f):
    try:
        tmp = 0
        for i in range(0,len(a)):
            tmp = tmp + getulp(a[i])*partial_derivative(f,b,i,a)
        ulp_b = getulp(b)
        cond_num = math.fabs(fdiv(tmp,ulp_b))
        if isinf(cond_num):
            return 1.7976931348623157e+308
        return cond_num
    except RuntimeWarning:
        return 1.0

#calculate the condition number: a is input; b is output; f is the numerical program
def condition(a,b,f):
    try:
        if math.fabs(b) < 2.2250738585072014e-308:
            j = 4.94065645841e-324
        else:
            j = b*2.2e-16
        i = a*2.220446049250313e-15
        ab = f(a + i) - b
        y = math.fabs(ab*1e-1/j)
        return y
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError) as e:
        y = 1.0
        return y

#return the 1/condition
def fitness_fun1(rf,pf,inp):
    try:
        b = pf(inp)
        res = condition(inp, b, pf)
        if (res == 0.0)|(np.isnan(res))|(res>np.finfo(np.double).max):
            res = 1.0
        else:
            res = 1.0/res
        return float(res)
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError) as e:
        return 1.0


# return the 1.0/FPNum
def fitness_fun(rf, pf, inp):
    try:
        inp = float(inp)
        r_val = float(rf(inp))
        p_val = float(pf(inp))
        if np.isnan(p_val):
            return 1.0
        if math.fabs(p_val) > np.finfo(np.double).max:
            return 1.0
        if math.fabs(r_val) > np.finfo(np.double).max:
            return 1.0
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return 1.0
    res = float(getUlpError(r_val, p_val))
    if (res == 0.0) | (np.isnan(res)):
        res = 1.0
    else:
        res = 1.0 / res
    if res == 0.0:
        res = 1.0
    return float(res)

def find_max(pf,rf,bound):
    bound_distance = getFPNum(bound[0],bound[1])
    popsize = 15+int(np.max([np.log2(bound_distance/1e9),1.0]))*2
    glob_fitness_fun = np.frompyfunc(lambda x: fitness_fun(rf, pf, x), 1, 1)
    ret = differential_evolution(glob_fitness_fun, popsize=popsize,maxiter=2000,bounds=[bound])
    return [1/ret.fun,ret.x]

def max_errorOnPoint(rf,pf,inp,step):
    ulp_inp = getulp(inp)
    glob_fitness_fun = np.frompyfunc(lambda x: fitness_fun(rf, pf, x), 1, 1)
    bn = np.max([step*1e-5,100])
    ret = differential_evolution(glob_fitness_fun,popsize=10,bounds=[[inp-bn*ulp_inp,inp+bn*ulp_inp]])
    return 1.0/ret.fun,ret.x[0]
#return the 1/condition
def mfitness_fun1(rf,pf,inp):
    try:
        b = pf(*inp)
        if np.isnan(b):
            return 1.0
        res = mcondition(inp, b, pf)
        if (res == 0.0)|(np.isnan(res))|(res>np.finfo(np.double).max):
            res = 1.0
        else:
            res = 1.0/res
        return float(res)
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError) as e:
        return 1.0
def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l

def mfitness_fun(rf, pf, inp):
    try:
        r_val = float(rf(*inp))
        p_val = float(pf(*inp))
        if np.isnan(p_val)|np.isnan(r_val):
            return 1.0
        if math.fabs(p_val) > np.finfo(np.double).max:
            return 1.0
        if math.fabs(r_val) > np.finfo(np.double).max:
            return 1.0
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return 1.0
    res = float(getUlpError(r_val, p_val))
    if (res == 0.0) | (np.isnan(res)):
        res = 1.0
    else:
        res = 1.0 / res
    if res == 0.0:
        res = 1.0
    return float(res)

#############################
# funcs for output results  #
#############################


def output_err(t_l,name,name2):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "max_error")
    sheet.write(0, 2, "input")
    sheet.write(0, 3, "interval")
    sheet.write(0, 4, "execute time")
    sheet.write(0, 5, "f1_n")
    sheet.write(0, 6, "f2_n")
    sheet.write(0, 7, "random_seed")
    sheet.write(0, 8, "count")
    sheet.write(0, 9, "MCMC_time")
    sheet.write(0, 10, "MCMC_jump")
    n = 1
    for t in t_l:
        sheet.write(n,0,name2)
        for k in range(0,len(t)):
            sheet.write(n,k+1,repr(t[k]))
        n = n+1
    book.save(name+".xls")



def test1vbound2excel(et,max_err,mean_err,max_x,exname,i,srate,lens):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    # table = old_excel.sheets()[0]
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(i, lens, repr(et))
    sheet.write(i, lens+1, repr(max_err))
    sheet.write(i, lens+2, repr(mean_err))
    sheet.write(i, lens+3, repr(max_x))
    sheet.write(i, lens+4, repr(srate))
    new_excel.save(exname)

def test2vbound2excel(et,max_err,mean_err,max_x,exname,i,srate,lens):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(i, lens, repr(et))
    sheet.write(i, lens+1, repr(max_err))
    sheet.write(i, lens+2, repr(mean_err))
    sheet.write(i, lens+3, str(max_x))
    sheet.write(i, lens+4, str(srate))
    new_excel.save(exname)

# def test1vbound2excel(et,max_err,mean_err,max_x,exname,i,lens):
#     book = xlrd.open_workbook(exname)
#     # sheet = old_excel.sheets()[0]
#     first_sheet = book.sheet_by_index(0)
#     # new_excel = copy(old_excel)
#     # sheet = new_excel.get_sheet(0)
#     first_sheet.write(i, lens, repr(et))
#     first_sheet.write(i, lens+1, repr(max_err))
#     first_sheet.write(i, lens+2, repr(mean_err))
#     first_sheet.write(i, lens+3, repr(max_x))
    # new_excel.save(exname)
# print getulp(-0.9999999999999999999)
# print gf17(1.3775982327465643, 0.17027936087729129)
# 1.702790989001757849e-01,
# 1.702793764559319412e-01,
# print rf17(1.3775982327465643, 0.17027936087729129)
# print getUlpError(1.702793764559319412e-01,0.17027936087729129)/1e10
# 91817.41136143089, 91819.74469405484
# 91823.91652828833, 91826.2498605461
# print getUlpError(91826.2498605461,91819.74469405484)
# print getUlpError( 3.667152670115475,3.667153136102705879)
# print getUlpError( 3.667152670115475,3.667153136102705879)
# print getUlpError( 19.07919866447409,1.907919870405034501e+01)/float(getUlpError(1.907916317691355701e+01,1.907919870405034501e+01))
# print getUlpError(0.9999999999999995212091059119913246975612,1.0)
# print getUlpError(9.999999999999992228e-01,0.9999999999999995212091059119913246975612)
# print getUlpError(9.999999999999992228e-01,1.0)
# print getUlpError(9.999999999999995559e-01,1.0)