from detector1v import detectHighErrs
import basic_func as bf
from mpmath import *
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from detector import DEMC_pure
import math
from eft import TwoSum
from eft import TwoPro
import matplotlib.pyplot as plt
from detector import DDEMC_pure
from localizer import PTB_MaxErr_tracing
import itertools
#line_search + error compensation
import sympy as sp
import pickle
import time
mp.dps = 40
#Build line with error compensation
def covetBoundTline(rf, bound, disbd):
    x0 = bound[0]
    x1 = bound[1]
    try:
        y0 = rf(bound[0])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y0 = rf(bound[0]+bf.getulp(bound[0]))
    try:
        y1 = rf(bound[1])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y1 = rf(bound[1]-bf.getulp(bound[1]))
    ulp_x0 = bf.getulp(x0)
    y1 = float(y1)
    y0 = float(y0)
    k = (y1 - y0) / (x1 - x0)
    delta_val = k * ulp_x0
    if bf.getulp(y1) < bf.getulp(y0):
        s_y = y1
        s_x = x1
        e_x = x0
        e_y = y0
    else:
        s_y = y0
        s_x = x0
        e_x = x1
        e_y = y1
    l_fun = lambda x: (x - s_x) * k + s_y
    if disbd == 2.0:
        return [(s_x, s_y), delta_val, bf.getMidDistance(x1, x0), 1.0, 1.0, x1, (x1, x0),(e_x, e_y)]
    glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(rf,l_fun,x), 1, 1)
    # ret = differential_evolution(glob_fitness_fun,popsize=15, bounds=[bound])
    mid_point = x0 + (x1 - x0) / 2.0
    ret = minimize(glob_fitness_fun,[mid_point], bounds=[bound])
    mid_b = ret.x[0]
    if 1.0/glob_fitness_fun(mid_point) -1.0/ret.fun>0:
        mid_b = mid_point
        print "replace"
    # mid_b = mid_point
    max_err = float(rf(mid_point)) - l_fun(mid_point)
    estimate_max_error = (mid_point - x0) * (mid_point - x1)
    curve_k = float(max_err / estimate_max_error)
    # if curve_k == 0.0:
    #     curve_k = 1.0
    return [(s_x, s_y), delta_val, bf.getMidDistance(x1, x0), 1.0,curve_k, mid_b, (x1, x0),
            (e_x, e_y)]

def taylor4_cof(rf,input,order):
    cof_l = []
    mp.dps = 50
    for i in range(order+1):
        for j in range(0,i+1):
            for p in range(0,j+1):
                for q in range(0,p+1):
                    temp_cof = fdiv(diff(rf,tuple(input),(i-j,j-p,p-q,q)),factorial(i-j)*factorial(j-p)*factorial(p-q)*factorial(q))
                    cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l
#produce the quadratic function
def lineAproFun(kb_val, x):
    i = kb_val[0]
    j = kb_val[7]
    if x == j[0]:
        return j[1]
    dv = kb_val[1]
    ulp_x = bf.getulp(x)
    if kb_val[4] == 0:
        compen = 0
    else:
        compen = (x-i[0])*(x-j[0]) * kb_val[4]
    rs = (x - i[0]) / ulp_x * dv + i[1] + compen
    return float(rs)

def lineFun(kb_val, x):
    i = kb_val[0]
    j = kb_val[7]
    if x == j[0]:
        return j[1]
    dv = kb_val[1]
    ulp_x = bf.getulp(x)
    rs = (x - i[0]) / ulp_x * dv + i[1]
    return float(rs)

# iterative refine algorithm
glob_point_l = []
iter_nums = 1
iter_count = 0
def iter_liner_build(th, bound, rf, n):
    mp.dps = 40
    global glob_point_l
    # global iter_nums
    # global iter_count
    disbd = bf.getFPNum(bound[0], bound[1])
    if disbd == 1:
        return 0
    kb_val = covetBoundTline(rf, bound, disbd)
    if disbd == 2:
        print kb_val
        glob_point_l.append(kb_val)
        return 0
    shadowFun = lambda x: lineAproFun(kb_val, x)
    glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(rf, shadowFun, x), 1, 1)
    ret = differential_evolution(glob_fitness_fun, popsize=15, bounds=[bound])
    # ret2 = DEMC_pure(rf,shadowFun,bound,1,1000)
    # if ret[0] == 0.0:
    #     print ret
    # if ret.fun == 0:
    #     print ret.fun
    #     print ret.x
    # max_err =
    # max_err = np.max([ret2[0],1.0 / glob_fitness_fun(ret.x[0])])
    # print "max_err"
    # print max_err
    # print ret2[0]
    # print 1.0 / glob_fitness_fun(ret.x[0])
    max_err = 1.0 / glob_fitness_fun(ret.x[0])
    mid_point = kb_val[5]
    if max_err <= np.floor(th):
        glob_point_l.append(kb_val)
        return 0
        # ret2 = DEMC_pure(rf, shadowFun, bound, 1, 1000)
        # if ret2[0] <= np.floor(th):
        #     iter_count = iter_count + 1
        #     # print "get here"
        #     # print float(iter_count)/iter_nums * 100
        #     # glob_point_l.append(kb_val)
        #     # return 0
        # else:
        #     print "inconsistant"
        #     return 0
    else:
        # iter_nums = iter_nums+2
        iter_liner_build(th, [bound[0], mid_point], rf, n)
        iter_liner_build(th, [mid_point, bound[1]], rf, n)


glob_point_l_tay = []

def taylor_exp(x,der_l,x0,n):
    temp = float(der_l[0])
    for i in range(1,n):
        temp = temp + math.pow((x-x0),i)*float(der_l[i])/math.factorial(i)
    return temp
def taylor_exp_eft(x,der_l_err,der_l,x0,n):
    temp_res = 0.0
    temp_err = 0.0
    k = 0
    for i in range(n):
        temp_cof = der_l[k]
        ta, tb = TwoPro(temp_cof, math.pow(x - x0, i))
        ea = der_l_err[k] * math.pow(x - x0, i)
        sa, sb = TwoSum(temp_res, ta)
        # sb_err, tb_err = TwoSum(float(temp_res),tb)
        temp_res = sa
        temp_err = temp_err + sb + tb + ea
        k = k + 1
    return temp_res + temp_err

def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)
def taylor1v_cof(f,inp,n):
    der_l = [fdiv(diff(f, inp, i),factorial(i)) for i in range(n+1)]
    der_l_err = [float(i-float(i)) for i in der_l]
    der_l_float = [float(i) for i in der_l]
    return der_l_err,der_l_float
#Build line with error compensation based on Taylor expersion
def covetBoundTlineTay(rf, bound, disbd,th):
    x0 = bound[0]
    x1 = bound[1]
    try:
        y0 = rf(bound[0])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y0 = rf(bound[0]+bf.getulp(bound[0]))
    try:
        y1 = rf(bound[1])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y1 = rf(bound[1]-bf.getulp(bound[1]))
    ulp_x0 = bf.getulp(x0)
    y1 = float(y1)
    y0 = float(y0)
    k = (y1 - y0) / (x1 - x0)
    delta_val = k * ulp_x0
    if bf.getulp(y1) < bf.getulp(y0):
        s_y = y1
        s_x = x1
        e_x = x0
        e_y = y0
    else:
        s_y = y0
        s_x = x0
        e_x = x1
        e_y = y1
    l_fun = lambda x: (x - s_x) * k + s_y
    if disbd == 2.0:
        return [(s_x, s_y),(e_x, e_y),bf.getMidDistance(x1, x0),delta_val,[],0,0,(x1, x0),1]
    #find the middle point
    ret = bf.find_max(rf,l_fun,bound)
    mid_point = ret[1][0]
    #Build the Error Function
    err_fun = lambda x: fsub(rf(x),l_fun(x))
    #Do Taylor Expansion around the Middle point
    #Note the Taylor Expansion is building on the Error function between the line and rf
    #Choose the level from 2-4 for the Taylor Expansion
    for i in range(2,5):
        der_l = []
        temp_n = i
        for j in range(temp_n):
            der_l.append(float(diff(err_fun,mid_point,j)))
        #get the taylor compensation function
        tayCom = lambda x: taylor_exp(x, der_l, mid_point, temp_n)
        #Construct the approximation
        appro_fun = lambda x: l_fun(x)+tayCom(x)
        res = bf.find_max_random(appro_fun, rf, bound)
        if res[0] < th:
            return [(s_x, s_y),(e_x, e_y),bf.getMidDistance(x1, x0),delta_val,der_l,temp_n,mid_point,(x1, x0),1]
    return [(s_x, s_y),(e_x, e_y),bf.getMidDistance(x1, x0),delta_val,der_l,temp_n,mid_point,(x1, x0),0]

# line iteration with Taylor expension
def iter_liner_build_LineTay(th, bound, rf, n):
    mp.dps = 40
    global glob_point_l_tay
    global iter_nums
    disbd = bf.getFPNum(bound[0], bound[1])
    if disbd == 1:
        return 0
    kb_val = covetBoundTlineTay(rf, bound, disbd,th)
    if disbd == 2:
        print kb_val
        glob_point_l_tay.append(kb_val)
        return 0
    mid_point = kb_val[6]
    if kb_val[-1]==1:
        glob_point_l_tay.append(kb_val)
        return 0
    else:
        iter_liner_build_LineTay(th, [bound[0], mid_point], rf, n)
        iter_liner_build_LineTay(th, [mid_point, bound[1]], rf, n)


def gener_tay_approxi(tay_lst,pf,x):
    for i in tay_lst:
        bound = i[3]
        if (x<=bound[1])&(x>=bound[0]):
            der_l_err = i[0]
            der_l_float = i[1]
            point = i[2]
            return taylor_exp_eft(x, der_l_err, der_l_float, point, len(der_l_err))
    return pf(x)

def plot_err_in_bound(rf,pf,err_bound,tay_lst):
    # err_bound = [tay_lst[0][3][0],tay_lst[-1][3][1]]
    tay_app = lambda x: gener_tay_approxi(tay_lst,pf,x)
    glob_fitness_real = lambda x: bf.fitness_fun(rf, tay_app, x)
    X = []
    Z = []
    input_l = np.random.uniform(err_bound[0], err_bound[1], 3000)
    for i in input_l:
        # temp_res = rf(i)
        temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        X.append(i)
        Z.append(float(temp_res))
        # Z.append(rf(i)-line_fun(i))
    print "max_Z"
    print glob_fitness_real(input_l[111])
    print tay_app(input_l[111])
    print rf(input_l[111])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Z, '.')
    ax.legend()
    plt.show()

def get_mid_lst(bound_l):
    mid_lst = []
    for i in bound_l:
        mid_lst.append(i[0]+(i[1]-i[0])/2.0)
    return mid_lst
def fake_rf(rf,inp):
    return math.fabs(float(rf(inp)))

def root_find_rf(rf,point,new_bound):
    try:
        glob_fitness_con = lambda x: fake_rf(rf, x)
        res = differential_evolution(glob_fitness_con, popsize=15, bounds=new_bound, polish=True, strategy='best1bin')
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point

def fake_rf2v(rf,inp):
    return math.fabs(float(rf(*inp)))

def root_find_rf2v(rf,point,new_bound):
    try:
        glob_fitness_con = lambda x: fake_rf2v(rf, x)
        res = differential_evolution(glob_fitness_con, popsize=15, bounds=new_bound, polish=True, strategy='best1bin')
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point
def horner_eft_1v(cof,x):
    pi_res = 0.0
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1,len(cof)):
        ta,tb = TwoPro(sa,x)
        sa,sb = TwoSum(ta,cof[i])
        pi_res = tb+sb + pi_res*x
        # sig_res = sig_res + sig_res*x
    return sa,pi_res
    # return sa+pi_res

def horner_eft_1vx(cof,x):
    pi_res = 0.0
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1, len(cof)):
        ta, tb = TwoPro(sa, x)
        sa, sb = TwoSum(ta, cof[i])
        pi_res = tb + sb + pi_res * x
        # sig_res = sig_res + sig_res*x
    # return [sa, pi_res]
    return sa+pi_res

def horner_1v(cof,x):
    res = cof[0]
    for i in range(1,len(cof)):
        res = res*x + cof[i]
    return res
def taylor1v_honer_eft(x,der_l_err,der_l,x0):
    temp_res = horner_eft_1vx(der_l,x-x0)
    temp_err = horner_1v(der_l_err,x-x0)
    return temp_res + temp_err

def iter_build_tay1v(th, bound,ori_bound, rf,pf,n,point,der_l_err, der_l_float,temp_n):
    mp.dps = 40
    global glob_point_l_tay
    global iter_nums
    disbd = bf.getFPNum(bound[0], bound[1])
    if disbd == 1:
        return 0
    # if len(glob_point_l_tay)>0:
    #     plot_err_in_bound(rf, pf, ori_bound, glob_point_l_tay)
    new_n = 2
    if der_l_err==[]:
        der_l_err, der_l_float = taylor1v_cof(rf, point, n)
    for i in range(temp_n,n+1):
        temp_cof=der_l_float[0:i+1]
        temp_cof_err=der_l_err[0:i+1]
        temp_cof.reverse()
        temp_cof_err.reverse()
        taylor1v_fun = lambda x: taylor1v_honer_eft(x,temp_cof_err ,temp_cof, point)
        # taylor1v_fun = lambda x: taylor_exp(x,der_l_float,point,i)
        glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(rf, taylor1v_fun, x), 1, 1)
        ret = differential_evolution(glob_fitness_fun, popsize=25, bounds=[bound])
        # ret = DEMC_pure(rf,taylor1v_fun,bound,1,1000)
        # if rf(ret.x)==0:
        #     point = ret.x
        # if ret.fun == 0:
        #     print ret.fun
        #     print ret.x
        max_err = 1.0/ret.fun
        # max_err = ret[0]
        # print max_err
        if max_err <= np.floor(th):
            glob_point_l_tay.append([temp_cof_err,temp_cof,point,bound,i])
            print "get here>>>>"
            return 0
        new_n = i-1
    # build new bound
    mid_p1 = point-(point-bound[0])/3.0
    mid_p2 = point+(bound[1]-point)/3.0
    new_bound_l = [[bound[0],mid_p1],[mid_p1,mid_p2],[mid_p2,bound[1]]]
    mid_lst = get_mid_lst(new_bound_l)
    iter_build_tay1v(th, new_bound_l[0],ori_bound, rf,pf, n, mid_lst[0],[],[],new_n)
    iter_build_tay1v(th, new_bound_l[1],ori_bound, rf,pf, n, point,der_l_err, der_l_float,new_n)
    iter_build_tay1v(th, new_bound_l[2],ori_bound, rf,pf, n, mid_lst[2],[],[],new_n)

def ini_tay1v_build(th, bound,ori_bound, rf,pf,n,point):
    try:
        # temp_p = root_find_rf(rf,point,[bound])
        temp_p = findroot(rf,point)
        if (temp_p<=bound[1])&(temp_p>=bound[0]):
            point = float(temp_p)
    except (ValueError, ZeroDivisionError, OverflowError, Warning,TypeError):
        point = point
        print "not in the point"
    der_l_err, der_l_float = taylor1v_cof(rf, point, n)
    temp_n = 1
    iter_build_tay1v(th, bound, ori_bound, rf, pf, n, point, der_l_err, der_l_float,temp_n)

def taylor2_cof(rfdd,input,order):
    cof_l = []
    for i in range(order+1):
        for j in range(0,i+1):
            temp_cof = fdiv(diff(rfdd,tuple(input),(i-j,j)),factorial(i-j)*factorial(j))
            cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l

def taylor2_fun_eft_ori_debug(cof,cof_err,x,y,input):
    order = int(cof[-1])
    k = 0
    temp_res = 0.0
    temp_err = 0.0
    for i in range(order+1):
        for j in range(0,i+1):
            temp_cof = cof[k]
            tam,tbm = TwoPro(math.pow(x - input[0], i-j),math.pow(y - input[1], j))
            ta, tb = TwoPro(temp_cof, tam)
            ea = cof_err[k]*tam
            ea2 = temp_cof*tbm
            sa, sb = TwoSum(temp_res,ta)
            # sb_err, tb_err = TwoSum(float(temp_res),tb)
            temp_res = sa
            # print "*****************"
            # print ta
            # print cof_real[k]*math.pow(x - input[0], i-j)*math.pow(y - input[1], j)
            # print cof_real[k]*pow(x - input[0], i-j)*pow(y - input[1], j)
            # print cof_real[k]*pow(x - input[0], i-j)*pow(y - input[1], j)-ta
            temp_err = temp_err + sb + tb + ea + ea2
            k = k + 1
    return temp_res+temp_err

def taylor2_fun_eft_ori(cof,cof_err,x,y,input):
    order = int(cof[-1])
    k = 0
    temp_res = 0.0
    temp_err = 0.0
    for i in range(order+1):
        for j in range(0,i+1):
            temp_cof = cof[k]
            temp_pow = math.pow(x - input[0], i-j)*math.pow(y - input[1], j)
            ta, tb = TwoPro(temp_cof, temp_pow)
            ea = cof_err[k]*temp_pow
            # ea2 = temp_cof*tbm
            sa, sb = TwoSum(temp_res,ta)
            # sb_err, tb_err = TwoSum(float(temp_res),tb)
            temp_res = sa
            # print "*****************"
            # print ta
            # print cof_real[k]*math.pow(x - input[0], i-j)*math.pow(y - input[1], j)
            # print cof_real[k]*pow(x - input[0], i-j)*pow(y - input[1], j)
            # print cof_real[k]*pow(x - input[0], i-j)*pow(y - input[1], j)-ta
            temp_err = temp_err + sb + tb + ea
            k = k + 1
    return temp_res+temp_err

def gen_mix_bound(dr,ini_bound,point):
    dbound = ini_bound[dr]
    dbound2 = ini_bound[1-dr]
    mid_p = dbound2[0]+(dbound2[1]-dbound2[0])/2.0
    dpoint = point[dr]
    mid_p1 = dpoint - (dpoint - dbound[0]) / 3.0
    mid_p2 = dpoint + (dbound[1] - dpoint) / 3.0
    new_bound_l = [[dbound[0], mid_p1], [mid_p1, mid_p2], [mid_p2, dbound[1]]]
    mid_lst = get_mid_lst(new_bound_l)
    mix_bound = []
    mix_points = []
    for i,j in zip(new_bound_l,mid_lst):
        temp_bound = [[], []]
        temp_point = [[], []]
        temp_bound[dr] = i
        temp_bound[1-dr] = dbound2
        temp_point[dr] = j
        temp_point[1-dr] = mid_p
        mix_bound.append(temp_bound)
        mix_points.append(temp_point)
    return mix_bound,mix_points
def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound

def point_in_bound(point,bound):
    flag = 1
    for i,j in zip(point,bound):
        if (i<=j[1])&(i>=j[0]):
            flag = 1*flag
        else:
            flag = 0*flag
    return flag


def get_jiaodian(poly_cof,bound,point):
    k = poly_cof[0]
    b = poly_cof[1]
    temp_points = []
    for j in itertools.product(*bound):
        temp_points.append(j)
    x0 = bound[0][0]-point[0]
    point1 = [bound[0][0],k*x0+b]
    x1 = bound[0][1]-point[0]
    point2 = [bound[0][1], k * x1 + b]
    if k!=0:
        y0 = bound[1][0]
        point3 = [(y0 - b) / k+point[0], y0]
        y1 = bound[1][1]
        point4 = [(y1 - b) / k+point[0], y1]
    else:
        point3 = []
        point4 = []
    points = [point1,point2,point3,point4]
    # print ">>>>>>>>>"
    # print points
    final_points = []
    for i in points:
        if i!=[]:
            flag = point_in_bound(i, bound)
            if flag == 1:
                final_points.append(i)
    return final_points




def get_direction(inte_points,ini_bound):
    if (inte_points[0][0]==ini_bound[0][0])&(inte_points[1][0]==ini_bound[0][1]):
        return 0
    else:
        return 1
def get_mid_lst(bound_l):
    mid_lst = []
    for i in bound_l:
        mid_lst.append(i[0]+(i[1]-i[0])/2.0)
    return mid_lst
def point_in_bound(point,bound):
    flag = 1
    for i,j in zip(point,bound):
        if (i<=j[1])&(i>=j[0]):
            flag = 1*flag
        else:
            flag = 0*flag
    return flag

def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l
def estimate_maxErr_inbound(rf,pf,bp):
    bound = generate_bound(bp,1e2)
    ulp_bp = bf.getulp(rf(*bp))
    ulp_bpr = bf.getulp(pf(*bp))
    input_l = produce_n_input(bound, 30)
    abs_err_lst = []
    ulp_err_lst = []
    for i in input_l:
        a = rf(*i)
        b = pf(*i)
        abs_err_lst.append(math.fabs(float(a-b)))
        ulp_err_lst.append(bf.getUlpError(a,b))
    max_abs_err = np.max(abs_err_lst)
    print "max_abs_err"
    print max_abs_err
    print rf(*bp)
    print rf(*bp)-pf(*bp)
    print np.max(ulp_err_lst)
    print ulp_bp
    print ulp_bpr
    print bf.getUlpError(rf(*bp),pf(*bp))
    print (rf(*bp)-pf(*bp))/ulp_bp
    estimate_err = max_abs_err/ulp_bp
    estimate_err2 = max_abs_err/ulp_bpr
    print estimate_err
    print estimate_err2
    return estimate_err

def test_err_on_2vbound_debug(rf,pf,bound,dr):
    mid_point = get_mid_lst(bound)
    bp1 = [0,0]
    bp2 = [0,0]
    bound_size = bf.getUlpError(bound[0][0], bound[0][1])
    step = int(bound_size) * 0.1
    bp1[dr] = bound[dr][0]
    bp1[1-dr] = mid_point[1-dr]
    bp2[dr] = bound[dr][1]
    bp2[1 - dr] = mid_point[1 - dr]
    bound1 = [[],[]]
    bound1[dr] = [bp1[dr],bp1[dr]+1e4*bf.getulp(bp1[dr])]
    bound1[1-dr] = bound[1-dr]
    bound2 = [[], []]
    bound2[dr] = [bp2[dr]- 1e4 * bf.getulp(bp2[dr]), bp2[dr] ]
    bound2[1 - dr] = bound[1 - dr]
    bp1 = root_find_rf2v(rf,bp1,bound1)
    es_err1 = estimate_maxErr_inbound(rf,pf,bp1)
    bp2 = root_find_rf2v(rf,bp2,bound2)
    es_err2 = estimate_maxErr_inbound(rf, pf, bp2)
    new_tes_bound = [[],[]]
    new_tes_bound[dr] = [bp1[dr],bp2[dr]]
    if bp1[1-dr]<bp2[1-dr]:
        new_tes_bound[1-dr] = [bp1[1-dr],bp2[1-dr]]
    else:
        new_tes_bound[1 - dr] = [bp2[1 - dr], bp1[1 - dr]]
    temp_res = DDEMC_pure(rf, pf, [new_tes_bound], 1, 20000)
    max_err = temp_res[0]
    new_tes_bound1 = generate_bound(bp1,step)
    new_tes_bound1[dr][0] = bp1[dr]
    new_tes_bound2 = generate_bound(bp2,step)
    new_tes_bound2[dr][1] = bp2[dr]
    temp_res = DDEMC_pure(rf, pf, [new_tes_bound1], 1, 20000)
    max_err1 = temp_res[0]
    temp_res = DDEMC_pure(rf, pf, [new_tes_bound2], 1, 20000)
    max_err2 = temp_res[0]
    print "<<<<<<<<<<"
    print max_err,max_err1,max_err2,es_err1,es_err2
    print bf.getUlpError(rf(*bp1),pf(*bp1))
    print bf.getUlpError(rf(*bp2),pf(*bp2))
    print rf(*bp1)
    print rf(*bp2)
    print point_in_bound(bp1,bound)
    print point_in_bound(bp2,bound)
    return np.max([max_err,max_err1,max_err2,es_err1,es_err2])

def test_err_on_2vbound_debug2(rf,pf,bound,dr):
    mid_point = get_mid_lst(bound)
    bp1 = [0,0]
    bp2 = [0,0]
    bound_size = bf.getUlpError(bound[0][0], bound[0][1])
    step = int(bound_size) * 0.1
    bp1[dr] = bound[dr][0]
    bp1[1-dr] = mid_point[1-dr]
    bp2[dr] = bound[dr][1]
    bp2[1 - dr] = mid_point[1 - dr]
    bound1 = [[],[]]
    bound1[dr] = [bp1[dr],bp1[dr]+1e4*bf.getulp(bp1[dr])]
    bound1[1-dr] = bound[1-dr]
    bound2 = [[], []]
    bound2[dr] = [bp2[dr]- 1e4 * bf.getulp(bp2[dr]), bp2[dr] ]
    bound2[1 - dr] = bound[1 - dr]
    bp1 = root_find_rf2v(rf,bp1,bound1)
    es_err1 = estimate_maxErr_inbound(rf,pf,bp1)
    bp2 = root_find_rf2v(rf,bp2,bound2)
    es_err2 = estimate_maxErr_inbound(rf, pf, bp2)
    print "<<<<<<<<<<"
    print es_err1,es_err2
    return np.max([es_err1,es_err2])
def test_err_on_2vbound(rf,pf,bound,dr):
    mid_point = get_mid_lst(bound)
    bp1 = [0,0]
    bp2 = [0,0]
    bound_size = bf.getUlpError(bound[0][0], bound[0][1])
    step = int(bound_size) * 0.1
    bp1[dr] = bound[dr][0]
    bp1[1-dr] = mid_point[1-dr]
    bp2[dr] = bound[dr][1]
    bp2[1 - dr] = mid_point[1 - dr]
    bound1 = [[],[]]
    bound1[dr] = [bp1[dr],bp1[dr]+step*bf.getulp(bp1[dr])]
    bound1[1-dr] = bound[1-dr]
    bound2 = [[], []]
    bound2[dr] = [bp2[dr]- step * bf.getulp(bp2[dr]), bp2[dr] ]
    bound2[1 - dr] = bound[1 - dr]
    res1 = DDEMC_pure(rf, pf, [bound1], 1, 20000)
    es_err1 = res1[0]
    res2 = DDEMC_pure(rf, pf, [bound2], 1, 20000)
    es_err2 = res2[0]
    print "<<<<<<<<<<"
    print es_err1,es_err2
    return np.max([es_err1,es_err2])
glob_tay2v_lst = []

def test_err_along_line2(kb_fun,bound,dr,rf,pf):
    dis_bound = 0.01*(bound[dr][1]-bound[dr][0])
    bound1 = [bound[dr][0],bound[dr][0]+dis_bound]
    bound2 = [bound[dr][1]-dis_bound,bound[dr][1]]
    st_time = time.time()
    X1 = np.random.uniform(bound1[0], bound1[1],1000)
    Y1 = [kb_fun(x1) for x1 in X1]
    X2 = np.random.uniform(bound2[0], bound2[1],1000)
    Y2 = [kb_fun(x2) for x2 in X2]
    print time.time()-st_time
    st_time = time.time()
    err_lst = []
    for i in zip(X1, Y1):
        temp_i = [0, 0]
        temp_i[dr] = i[0]
        temp_i[1-dr] = i[1]
        a = rf(*temp_i)
        b = pf(*temp_i)
        err_lst.append(bf.getUlpError(a,b))
    max_err1 = np.max(err_lst)
    err_lst = []
    for i in zip(X2, Y2):
        temp_i = [0, 0]
        temp_i[dr] = i[0]
        temp_i[1 - dr] = i[1]
        a = rf(*temp_i)
        b = pf(*temp_i)
        err_lst.append(bf.getUlpError(a, b))
    max_err2 = np.max(err_lst)
    print time.time() - st_time
    return np.max([max_err1,max_err2])

def gen_mix_bounds(bound,dr,kb_fun):
    dis_bound = 0.01 * (bound[dr][1] - bound[dr][0])
    bound1 = [bound[dr][0], bound[dr][0] + dis_bound]
    bound1dr = [kb_fun(i) for i in bound1]
    bound1dr.sort()
    bound2 = [bound[dr][1] - dis_bound, bound[dr][1]]
    bound2dr = [kb_fun(i) for i in bound2]
    bound2dr.sort()
    fbound1 = bound
    fbound1[dr] = bound1
    fbound1[1-dr] = bound1dr
    fbound2 = bound
    fbound2[dr] = bound2
    fbound2[1 - dr] = bound2dr
    return fbound1,fbound2
def test_err_along_line2(kb_fun,bound,dr,rf,pf):
    # if dr == 1:
    #     new_rf = lambda y: rf(kb_fun(y), y)
    #     new_pf = lambda y: pf(kb_fun(y), y)
    # else:
    #     new_rf = lambda x: rf(x, kb_fun(x))
    #     new_pf = lambda x: pf(x, kb_fun(x))
    # dis_bound = 0.01 * (bound[dr][1] - bound[dr][0])
    dis_bound = 0.005 * (bound[dr][1] - bound[dr][0])
    bp1 = [bound[dr][0], bound[dr][0] + dis_bound]
    bp2 = [bound[dr][1] - dis_bound, bound[dr][1]]
    st_time = time.time()
    X1 = np.random.uniform(bp1[0], bp1[1], 200)
    Y1 = [kb_fun(x1) for x1 in X1]
    X2 = np.random.uniform(bp2[0], bp2[1], 200)
    Y2 = [kb_fun(x2) for x2 in X2]
    print time.time() - st_time
    st_time = time.time()
    res_lst1 = []
    abs_err1 = []
    for i in zip(X1, Y1):
        temp_i = [0, 0]
        temp_i[dr] = i[0]
        temp_i[1 - dr] = i[1]
        a = rf(*temp_i)
        b = pf(*temp_i)
        abs_err1.append(fabs(a-b))
        res_lst1.append(fabs(b))
    max_res1 = np.min(res_lst1)
    res_lst2 = []
    abs_err2 = []
    for i in zip(X2, Y2):
        temp_i = [0, 0]
        temp_i[dr] = i[0]
        temp_i[1 - dr] = i[1]
        a = rf(*temp_i)
        b = pf(*temp_i)
        abs_err2.append(fabs(a - b))
        res_lst2.append(fabs(b))
    max_res2 = np.min(res_lst1)
    print time.time() - st_time
    # bound1,bound2 = gen_mix_bounds(bound,dr,kb_fun)
    # bound1 = [bound[dr][0], bound[dr][0] + dis_bound]
    # bound2 = [bound[dr][1] - dis_bound, bound[dr][1]]
    # glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(new_rf, new_pf, x), 1, 1)
    # ret1 = differential_evolution(glob_fitness_fun, popsize=15, bounds=[bound1])
    # ret2 = differential_evolution(glob_fitness_fun, popsize=15, bounds=[bound2])
    # DDEMC_pure(rf, pf, [bound1], 1, 20000)
    # input_l1 = produce_n_input(bound1,20)
    # input_l2 = produce_n_input(bound2,20)
    # abs_err1 = [fabs(rf(*i)-pf(*i)) for i in input_l1]
    # abs_err2 = [fabs(rf(*i)-pf(*i)) for i in input_l2]
    ulp_err1 = np.max(abs_err1)/bf.getulp(max_res1)
    ulp_err2 = np.max(abs_err2)/bf.getulp(max_res2)
    # res3 = DDEMC_pure(rf, pf, [bound1], 1, 20000)
    # res4 = DDEMC_pure(rf, pf, [bound2], 1, 20000)
    # print res3
    # print ret1.x[0]
    # print ret2.x[0]
    # err1 = 1.0/ret1.fun
    # err2 = 1.0/ret2.fun
    # print err1
    # print err2
    # err3 = res3[0]
    # err4 = res4[0]
    return np.max([ulp_err1,ulp_err2])
    # res1 = DEMC_pure(new_rf,new_pf,bound[dr],1,20000)
    # print res1
    # print new_pf(res1[1])
    # print kb_fun(res1[1])
    # print new_pf(res1[1])
    # dis_bound = 0.1*(bound[dr][1]-bound[dr][0])
    # bound1 = [bound[dr][0],bound[dr][0]+dis_bound]
    # bound2 = [bound[dr][1]-dis_bound,bound[dr][1]]
    # return res1[0]

def test_err_along_line(kb_fun,bound,dr,rf,pf):
    if dr == 1:
        new_rf = lambda y: rf(kb_fun(y), y)
        new_pf = lambda y: pf(kb_fun(y), y)
    else:
        new_rf = lambda x: rf(x, kb_fun(x))
        new_pf = lambda x: pf(x, kb_fun(x))
    dis_bound = 0.01 * (bound[dr][1] - bound[dr][0])
    bound1 = [bound[dr][0], bound[dr][0] + dis_bound]
    bound2 = [bound[dr][1] - dis_bound, bound[dr][1]]
    # glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(new_rf, new_pf, x), 1, 1)
    # ret1 = differential_evolution(glob_fitness_fun, popsize=25, bounds=[bound1])
    # ret2 = differential_evolution(glob_fitness_fun, popsize=25, bounds=[bound2])
    # print ret1.x[0]
    # print ret2.x[0]
    # err1 = 1.0 / ret1.fun
    # err2 = 1.0 / ret2.fun
    # print err1
    # print err2
    res3 = DEMC_pure(new_rf,new_pf,bound1,1,20000)
    res4 = DEMC_pure(new_rf,new_pf,bound2,1,20000)
    # print res3
    # print res4
    err3 = res3[0]
    err4 = res4[0]
    # return np.max([err1,err2])
    return np.max([err3,err4])
    # res1 = DEMC_pure(new_rf,new_pf,bound[dr],1,20000)
    # print res1
    # print new_pf(res1[1])
    # print kb_fun(res1[1])
    # print new_pf(res1[1])
    # dis_bound = 0.1*(bound[dr][1]-bound[dr][0])
    # bound1 = [bound[dr][0],bound[dr][0]+dis_bound]
    # bound2 = [bound[dr][1]-dis_bound,bound[dr][1]]
    # return res1[0]

def taylor2_horner_cof(rf,input,order):
    cof_l = []
    mp.dps = 40
    for i in range(order+1):
        lst = range(order-i+1)
        lst.reverse()
        for j in lst:
            temp_cof = fdiv(diff(rf, tuple(input), (i, j)), factorial(i) * factorial(j))
            cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l


def taylor32_horner_eft(cof,x,y,input,order):
    k = 0
    lst = range(order+1)
    lst.reverse()
    temp_cof_x = []
    temp_cof_x2 = []
    for i in lst:
        cof1,cof2 = horner_eft_1v(cof[k:k+i+1], y-input[1])
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        # temp_cof_x_err.append(horner_eft_1vx(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_res,temp_res1 = horner_eft_1v(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    # temp_err = horner_eft_1vx(temp_cof_x_err, x-input[0])
    return temp_res,temp_res2+temp_res1

def horner_2v(cof,x,y,order):
    k = 0
    lst = range(order + 1)
    lst.reverse()
    temp_cof_x = []
    for i in lst:
        cof1 = horner_1v(cof[k:k+i+1],y)
        temp_cof_x.append(cof1)
    temp_cof_x.reverse()
    res = horner_1v(temp_cof_x,x)
    return res

def taylor3_horner_eft_ori2(cof,cof_err,x,y,z,input):
    order = int(cof[-1])
    k = 0
    lst = range(order+1)
    temp_cof_x = []
    temp_cof_x2 = []
    temp_cof_x_err = []
    for i in lst:
        cof1,cof2 = taylor32_horner_eft(cof[i], y, z, input[1:],order-i)
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        temp_cof_x_err.append(horner_2v(cof_err[i],y-input[1],z-input[2],order-i))
        # temp_cof_x_err.append(horner_eft_1vx(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vx(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    temp_err = horner_1v(temp_cof_x_err, x-input[0])
    # temp_err = horner_eft_1vx(temp_cof_x_err, x-input[0])
    return temp_res+temp_res2+temp_err

def taylor2_horner_eft_ori2(cof,cof_err,x,y,input):
    order = int(cof[-1])
    k = 0
    lst = range(order+1)
    lst.reverse()
    temp_cof_x = []
    temp_cof_x2 = []
    temp_cof_x_err = []
    for i in lst:
        cof1,cof2 = horner_eft_1v(cof[k:k+i+1], y-input[1])
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        temp_cof_x_err.append(horner_1v(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vx(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    temp_err = horner_1v(temp_cof_x_err, x-input[0])
    return temp_res+temp_res2+temp_err

def iter_build_tay2v_debug(th,bound,ori_bound,rf,pf,n,point,dr,cof_err,cof_float,kb_fun,temp_n):
    global glob_tay2v_lst
    if cof_err == []:
        cof = taylor2_cof(rf, point, n)
        cof_err = [float(x - float(x)) for x in cof]
        cof_float = [float(x) for x in cof]
    mix_bound, mix_points = gen_mix_bound(dr, bound, point)
    bound_size = bf.getUlpError(bound[dr][0], bound[dr][1])
    new_n = 2
    for i in range(temp_n, n + 1):
        idx = 0
        for j in range(0, i + 1):
            idx = idx + j + 1
        temp_cof = cof_float[0:idx] + [i]
        print temp_cof
        temp_cof_err = cof_err[0:idx]
        print temp_cof_err
        tay_fun = lambda x, y: taylor2_fun_eft_ori(temp_cof, temp_cof_err, x, y, point)
        temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
        max_err = temp_res[0]
        print max_err
        # print rf(*temp_res[1])
        if max_err < th:
            st_time = time.time()
            temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
            print time.time()-st_time
            max_err = temp_res[0]
            max_err2 = test_err_along_line(kb_fun,bound,dr,rf,tay_fun)
            # temp_err_lst = []
            # for k in mix_bound:
            #     temp_res = DDEMC_pure(rf, tay_fun, [k], 1, 20000)
            #     temp_maxerr = temp_res[0]
            #     temp_err_lst.append(temp_maxerr)
                # print temp_maxerr3
                # temp_maxerr3 = test_err_on_2vbound(rf, tay_fun, bound, dr)
                # print ">>>>>>"
                # print temp_maxerr3
            # temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
            # max_err2 = temp_res[0]
            # max_err2 = np.max(temp_err_lst)
            # max_err2 = test_err_on_2vbound(rf,tay_fun,bound,dr)
            max_err2 = np.max([max_err2,max_err])
            print max_err2
            if max_err2 < th:
                print "get Here"
                glob_tay2v_lst.append([temp_cof_err,temp_cof,point,bound])
                return 0
        new_n = i
    # print "<<<<<<<<<<"
    # print mix_bound
    # print mix_points
    step = int(bound_size) * 0.01
    find_root_bound = []
    for i,j in zip(mix_bound,mix_points):
        temp_bound = generate_bound(j,step)
        temp_bound[1-dr]=i[1-dr]
        find_root_bound.append(temp_bound)
    roots_lst = []
    # print find_root_bound
    for i,j in zip(find_root_bound,mix_points):
        roots_lst.append(root_find_rf2v(rf,j,i))
    for i in roots_lst:
        print rf(*i)
        print pf(*i)
        print 1.0/bf.mfitness_fun(rf,pf,i)
    count = 0
    for i,j in zip(mix_bound,roots_lst):
        if count == 1:
            iter_build_tay2v(th, i, ori_bound, rf, pf, n, point, dr,cof_err,cof_float,kb_fun,new_n-1)
        else:
            iter_build_tay2v(th, i, ori_bound, rf, pf, n, j, dr,[],[],kb_fun,new_n-1)
        count = count+1
    # print "************"
    # print roots_lst
    # print mix_bound
    # print mix_points
    return 0

def get_item_cof(i,cof,n):
    len_lst = range(n+1)
    len_lst.reverse()
    temp_cof = []
    k=0
    for j in len_lst:
        if i>=0:
            temp_cof = temp_cof + cof[j-i+k:j+k+1]
        else:
            break
        i = i-1
        k = j+k+1
    return temp_cof



def iter_build_tay2v(th,bound,ori_bound,rf,pf,n,point,dr,cof_err,cof_float,kb_fun,temp_n):
    global glob_tay2v_lst
    if cof_err == []:
        cof = taylor2_horner_cof(rf, point, n)
        cof_err = [float(x - float(x)) for x in cof]
        cof_float = [float(x) for x in cof]
    mix_bound, mix_points = gen_mix_bound(dr, bound, point)
    bound_size = bf.getUlpError(bound[dr][0], bound[dr][1])
    new_n = 2
    print "temp_n"
    print temp_n
    for i in range(temp_n, n + 1):
        idx = 0
        # for j in range(0, i + 1):
        #     idx = idx + j + 1
        temp_cof = get_item_cof(i,cof_float,n)+[i]
        print temp_cof
        print cof_float
        temp_cof_err = get_item_cof(i,cof_err,n)
        print temp_cof_err
        tay_fun = lambda x, y: taylor2_horner_eft_ori2(temp_cof, temp_cof_err, x, y, point)
        temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
        max_err = temp_res[0]
        print max_err
        # print rf(*temp_res[1])
        if max_err < th:
            # st_time = time.time()
            # temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
            # print time.time() - st_time
            # max_err = temp_res[0]
            print max_err
            max_err2 = test_err_along_line(kb_fun, bound, dr, rf, tay_fun)
            # temp_err_lst = []
            # for k in mix_bound:
            #     temp_res = DDEMC_pure(rf, tay_fun, [k], 1, 20000)
            #     temp_maxerr = temp_res[0]
            #     temp_err_lst.append(temp_maxerr)
            # print temp_maxerr3
            # temp_maxerr3 = test_err_on_2vbound(rf, tay_fun, bound, dr)
            # print ">>>>>>"
            # print temp_maxerr3
            # temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
            # max_err2 = temp_res[0]
            # max_err2 = np.max(temp_err_lst)
            # max_err2 = test_err_on_2vbound(rf,tay_fun,bound,dr)
            # max_err2 = np.max([max_err2, max_err])
            print max_err2
            if max_err2 < th:
                print "get Here"
                print i
                print [temp_cof_err,temp_cof,point,bound]
                glob_tay2v_lst.append([temp_cof_err,temp_cof,point,bound])
                return 0
        new_n = i
    # print "<<<<<<<<<<"
    # print mix_bound
    # print mix_points
    # step = int(bound_size) * 0.01
    # find_root_bound = []
    # for i,j in zip(mix_bound,mix_points):
    #     temp_bound = generate_bound(j,step)
    #     temp_bound[1-dr]=i[1-dr]
    #     find_root_bound.append(temp_bound)
    roots_lst = []
    # print find_root_bound
    # for i,j in zip(find_root_bound,mix_points):
    for i in mix_points:
        temp_point = [0,0]
        temp_point[dr] = i[dr]
        temp_point[1-dr] = kb_fun(i[dr])
        # roots_lst.append(root_find_rf2v(rf,j,i))
        roots_lst.append(temp_point)
    for i in roots_lst:
        print rf(*i)
        print pf(*i)
        print 1.0/bf.mfitness_fun(rf,pf,i)
    count = 0
    for i,j in zip(mix_bound,roots_lst):
        if count == 1:
            iter_build_tay2v(th, i, ori_bound, rf, pf, n, point, dr,cof_err,cof_float,kb_fun,new_n)
        else:
            iter_build_tay2v(th, i, ori_bound, rf, pf, n, j, dr,[],[],kb_fun,new_n)
        count = count+1
    # print "************"
    # print roots_lst
    # print mix_bound
    # print mix_points
    return 0

def bound_rand_test(bound,rf,tay_fun,test_n):
    input_l = produce_n_input(bound, test_n)
    st = time.time()
    pf_res = [tay_fun(*i) for i in input_l]
    rf_res = [rf(*i) for i in input_l]
    err_lst = [bf.getUlpError(i, j) for i, j in zip(pf_res, rf_res)]
    print time.time() - st
    max_err = np.max(err_lst)
    return max_err
def get_bound_points(re_bound):
    points = []
    for i in itertools.product(*re_bound):
        points.append(i)
    return points
def checkIn_bound(kb_fun,re_bound,dr):
    points = get_bound_points(re_bound)
    flags = 0
    for i in points:
        temp_dr = kb_fun(i[dr])
        if temp_dr<i[1-dr]:
            flags=flags+1
        else:
            flags=flags+0
    if (flags==0)|(flags==4):
        return 0
    else:
        return 1

def iter_build_tay2v_debug(th,bound,ori_bound,rf,pf,n,point,dr,cof_err,cof_float,kb_fun,temp_n):
    global glob_tay2v_lst
    if cof_err == []:
        cof = taylor2_horner_cof(rf, point, n)
        cof_err = [float(x - float(x)) for x in cof]
        cof_float = [float(x) for x in cof]
    new_n = 2
    print "temp_n"
    print temp_n
    for i in range(temp_n, n + 1):
        idx = 0
        temp_cof = get_item_cof(i,cof_float,n)+[i]
        print temp_cof
        print cof_float
        temp_cof_err = get_item_cof(i,cof_err,n)
        print temp_cof_err
        tay_fun = lambda x, y: taylor2_horner_eft_ori2(temp_cof, temp_cof_err, x, y, point)
        temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
        max_err = temp_res[0]
        rnd_err = bound_rand_test(bound,rf,tay_fun,30)
        max_err = np.max([max_err,rnd_err])
        print max_err
        if max_err < th:
            if checkIn_bound(kb_fun,bound,dr):
                print max_err
                max_err2 = test_err_along_line(kb_fun, bound, dr, rf, tay_fun)
                print max_err2
                if max_err2 < th:
                    print "get Here"
                    print i
                    print [temp_cof_err,temp_cof,point,bound]
                    glob_tay2v_lst.append([temp_cof_err,temp_cof,point,bound])
                    return 0
            else:
                glob_tay2v_lst.append([temp_cof_err, temp_cof, point, bound])
                return 0
        new_n = i
    roots_lst = []
    new_bounds, temp_bound = bound_divide_3(bound)
    for i in new_bounds:
        temp_res = DDEMC_pure(rf, pf, [i], 1, 20000)
        max_err2 = temp_res[0]
        if max_err2 > th:
            temp_res2 = DDEMC_pure(rf, tay_fun, [i], 1, 20000)
            max_err2 = temp_res2[0]
            max_err = bound_rand_test(bound,rf,tay_fun,100)
            max_err2 = np.max([max_err, max_err2])
            print max_err2
            if checkIn_bound(kb_fun,bound,dr):
                print max_err
                max_err = test_err_along_line(kb_fun, bound, dr, rf, tay_fun)
                max_err2 = np.max([max_err, max_err2])
            if max_err2 > th:
                if i == temp_bound:
                    print "in point"
                    iter_build_tay2v_debug(th, i, ori_bound, rf, pf, n, point, dr,cof_err,cof_float,kb_fun,new_n)
                else:
                    iter_build_tay2v_debug(th, i, ori_bound, rf, pf, n, temp_res[1], dr,[],[],kb_fun,new_n)
            else:
                glob_tay2v_lst.append([cof_err, cof_float, point, i])
    return 0
def estimate_tay2v_bound(th,bound,ori_bound,rf,pf,n,point,dr,cof_err,cof_float,kb_fun,temp_n):
    if cof_err == []:
        cof = taylor2_horner_cof(rf, point, n)
        cof_err = [float(x - float(x)) for x in cof]
        cof_float = [float(x) for x in cof]
    print "temp_n"
    print temp_n
    mix_bound = []
    for j in range(0,100):
        for i in range(temp_n, n + 1):
            idx = 0
            # for j in range(0, i + 1):
            #     idx = idx + j + 1
            temp_cof = get_item_cof(i,cof_float,n)+[i]
            print temp_cof
            print cof_float
            temp_cof_err = get_item_cof(i,cof_err,n)
            print temp_cof_err
            tay_fun = lambda x, y: taylor2_horner_eft_ori2(temp_cof, temp_cof_err, x, y, point)
            temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
            max_err = temp_res[0]
            print max_err
            # print rf(*temp_res[1])
            if max_err < th:
                max_err2 = test_err_along_line(kb_fun,bound,dr,rf,tay_fun)
                # temp_err_lst = []
                # for k in mix_bound:
                #     temp_res = DDEMC_pure(rf, tay_fun, [k], 1, 20000)
                #     temp_maxerr = temp_res[0]
                #     temp_err_lst.append(temp_maxerr)
                    # print temp_maxerr3
                    # temp_maxerr3 = test_err_on_2vbound(rf, tay_fun, bound, dr)
                    # print ">>>>>>"
                    # print temp_maxerr3
                # temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
                # max_err2 = temp_res[0]
                # max_err2 = np.max(temp_err_lst)
                # max_err2 = test_err_on_2vbound(rf,tay_fun,bound,dr)
                print max_err2
                if max_err2 < th:
                    return bound
        temp_n = n
        roots_lst = []
        mix_bound, mix_points = gen_mix_bound(dr, bound, point)
        for i in mix_points:
            temp_point = [0,0]
            temp_point[dr] = i[dr]
            temp_point[1-dr] = kb_fun(i[dr])
            roots_lst.append(temp_point)
        for i in roots_lst:
            print rf(*i)
            print pf(*i)
            print 1.0/bf.mfitness_fun(rf,pf,i)
        bound = mix_bound[1]
    # print "************"
    # print roots_lst
    # print mix_bound
    # print mix_points
    return 0


def find_accuracy_Roots(rf,point,ini_bound,dr):
    new_step = (ini_bound[dr][1]-ini_bound[dr][0])/200
    # new_bound = generate_bound(point,1e3)
    # point = root_find_rf(rf,point,new_bound)
    p_val = rf(*point)
    # new_step = 1e3
    b0 = ini_bound[dr][0]
    if p_val == 0:
        p_val = 1e-18
    # ulp_b0 = bf.getulp(b0)
    temp_b = b0
    points = []
    for i in range(0,200):
        temp_bound = [[],[]]
        temp_bound[dr] = [temp_b,temp_b+new_step]
        temp_bound[1-dr] = ini_bound[1-dr]
        mid_p = get_mid_lst(temp_bound)
        # temp_point = root_find_rf(tay_fun,mid_p,temp_bound)
        if dr == 0:
            new_rf = lambda y: rf(mid_p[0], y)
            temp_point = [mid_p[0], findroot(new_rf, mid_p[1], tol=p_val * p_val*10)]
        else:
            new_rf = lambda x: rf(x,mid_p[1])
            temp_point = [findroot(new_rf, mid_p[0], tol=p_val*p_val*10),mid_p[1]]
        # res = DDEMC_pure(rf, pf, [temp_bound], 1, 200)
        # points.append(list(res[1]))
        points.append(list(temp_point))
        temp_b = temp_b+new_step
    A = []
    B = []
    p0 = points[0]
    for i in points[1:]:
        # print (i[1-dr]-p0[1-dr])/(i[dr]-p0[dr])
        ta = mpf(i[dr]-p0[dr])
        # ta = mpf(i[0])
        tb = mpf(i[1-dr]-p0[1-dr])
        # tb = mpf(i[1])
        # A.append([1,ta])
        # A.append([1.0,ta,pow(ta,2.0),pow(ta,3.0),pow(ta,4.0),pow(ta,5.0),pow(ta,6.0)])
        # A.append([1.0,ta,fmul(ta,ta),pow(ta,3.0)])
        A.append([1.0,ta,fmul(ta,ta),pow(ta,3.0),pow(ta,4.0)])
        # A.append([1.0,ta,fmul(ta,ta)])
        # A.append([1.0,ta])
        B.append(tb)
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    kb = list(A.cholesky_solve(B))
    # kb_fun = lambda x:float(kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    # kb_fun = lambda x:float(kb[3]*pow((x-p0[dr]),3.0)+kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    kb_fun = lambda x:float(kb[4]*pow((x-p0[dr]),4.0)+kb[3]*pow((x-p0[dr]),3.0)+kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    return kb_fun

def ls_fun(inp,dr,x,n,kb):
    temp_res = 0
    for i in range(0,n+1):
        temp_res = temp_res+ pow((x-inp[dr]),i)*kb[i]
    return temp_res+inp[1-dr]

def divid_bound(ini_bound,es_bound,dr,kb_fun):
    bound_size = bf.getUlpError(ini_bound[dr][0], ini_bound[dr][1])
    bound_size2 = bf.getUlpError(es_bound[dr][0], es_bound[dr][1])
    bounds = []
    points = []
    ids = int(bound_size/float(bound_size2))+1
    temp_point = ini_bound[dr][0]
    temp_bound = list(ini_bound)
    for i in range(ids):
        next_point = bf.get_next_point(temp_point,bound_size2,1)
        if next_point< ini_bound[dr][1]:
            temp_bound[dr]=[temp_point, next_point]
            bounds.append(temp_bound)
            mid_p = get_mid_lst(temp_bound)
            mid_p[1 - dr] = kb_fun(mid_p[dr])
            points.append(mid_p)
        else:
            temp_bound[dr] = [temp_point, ini_bound[dr][1]]
            bounds.append(temp_bound)
            mid_p = get_mid_lst(temp_bound)
            mid_p[1 - dr] = kb_fun(mid_p[dr])
            points.append(mid_p)
            break
        temp_point = next_point
    return bounds,points
def ini_tay2v_build(rf,pf,point,ini_bound,th,n,file_name):
    # get the max_err line
    kb_fun_file = file_name + "/kb_fun.txt"
    bounds_file = file_name + "/bounds.txt"
    points_file = file_name + "/points.txt"
    kb_fun_cof = pickle.load(open(kb_fun_file, "rb"))
    bounds = pickle.load(open(bounds_file, "rb"))
    points = pickle.load(open(points_file, "rb"))
    kb = kb_fun_cof[0]
    p0 = kb_fun_cof[1]
    dr = kb_fun_cof[2]
    kbn = kb_fun_cof[3]
    print "********"
    print len(kb)
    # ps_len = len(points)/2
    # temp_res = fabs(rf(*point))
    # for i in range(ps_len-10,ps_len+10):
    #     next_res = fabs(rf(*points[i]))
    #     if next_res<temp_res:
    #         point = points[i]
    kb_fun = lambda x: float(ls_fun(p0, dr, x, kbn, kb))
    final_bound = bounds[0]
    mid_p = get_mid_lst(final_bound)
    mid_p[1 - dr] = kb_fun(mid_p[dr])
    p_val = rf(*mid_p)
    if dr == 0:
        new_rf = lambda y: rf(mid_p[0], y)
        temp_point = [mid_p[0], float(findroot(new_rf, mid_p[1], tol=p_val * p_val))]
    else:
        new_rf = lambda x: rf(x, mid_p[1])
        temp_point = [float(findroot(new_rf, mid_p[0], tol=p_val * p_val)), mid_p[1]]
    point = list(temp_point)
    print rf(*mid_p)
    print rf(*point)
    # bound_size = bf.getUlpError(ini_bound[dr][0],ini_bound[dr][1])
    mid_p = point
    cof = taylor2_horner_cof(rf, point, n)
    cof_err = [float(x - float(x)) for x in cof]
    cof_float = [float(x) for x in cof]
    temp_n = 1
    iter_build_tay2v(th/2.0, ini_bound, ini_bound, rf, pf, n, point, dr, cof_err, cof_float,kb_fun,temp_n)
    return dr



def ini_tay2v_build_debug(rf,pf,point,ini_bound,th,n,file_name):
    # get the max_err line
    kb_fun_file = file_name + "/kb_fun.txt"
    bounds_file = file_name + "/bounds.txt"
    points_file = file_name + "/points.txt"
    kb_fun_cof = pickle.load(open(kb_fun_file, "rb"))
    bounds = pickle.load(open(bounds_file, "rb"))
    points = pickle.load(open(points_file, "rb"))
    kb = kb_fun_cof[0]
    p0 = kb_fun_cof[1]
    dr = kb_fun_cof[2]
    kbn = kb_fun_cof[3]
    print "********"
    print len(kb)
    # ps_len = len(points)/2
    # temp_res = fabs(rf(*point))
    # for i in range(ps_len-10,ps_len+10):
    #     next_res = fabs(rf(*points[i]))
    #     if next_res<temp_res:
    #         point = points[i]
    kb_fun = lambda x: float(ls_fun(p0, dr, x, kbn, kb))
    final_bound = bounds[0]
    mid_p = get_mid_lst(final_bound)
    mid_p[1 - dr] = kb_fun(mid_p[dr])
    p_val = rf(*mid_p)
    if dr == 0:
        new_rf = lambda y: rf(mid_p[0], y)
        temp_point = [mid_p[0], float(findroot(new_rf, mid_p[1], tol=p_val * p_val))]
    else:
        new_rf = lambda x: rf(x, mid_p[1])
        temp_point = [float(findroot(new_rf, mid_p[0], tol=p_val * p_val)), mid_p[1]]
    point = list(temp_point)
    print rf(*mid_p)
    print rf(*point)
    # bound_size = bf.getUlpError(ini_bound[dr][0],ini_bound[dr][1])
    mid_p = point
    cof = taylor2_horner_cof(rf, point, n)
    cof_err = [float(x - float(x)) for x in cof]
    cof_float = [float(x) for x in cof]
    temp_n = 1
    # st_time = time.time()
    # es_bound = estimate_tay2v_bound(th, ini_bound, ini_bound, rf, pf, n, point, dr, cof_err, cof_float, kb_fun, temp_n)
    # print "*******"
    # print time.time()-st_time
    # bounds,points = divid_bound(ini_bound, es_bound, dr,kb_fun)
    # print bounds
    # print points
    # print ini_bound
    # print len(bounds)
    # print len(bounds)*time.time()-st_time
    iter_build_tay2v(th/2.0, ini_bound, ini_bound, rf, pf, n, point, dr, cof_err, cof_float,kb_fun,temp_n)
    return dr
    # dbound = ini_bound[dr]
    # dpoint = point[dr]
    # mid_p1 = dpoint-(dpoint-dbound[0])/3.0
    # mid_p2 = dpoint+(dpoint-dbound[0])/3.0
    # new_bound_l = [[dbound[0], mid_p1], [mid_p1, mid_p2], [mid_p2, dbound[1]]]
    # mid_lst = get_mid_lst(new_bound_l)
    # print mid_lst


def step_back_3v(temp_step,step):
    return temp_step

# def ini_tay3v_build(rf,pf,point,ini_bound,th,n):
#     res_lst = []
#     cof = taylor2_horner_cof(rf, point, n)
#     for i in range(1, n + 1):
#         temp_step = 0
#         tay_fun = 0
#         for j in range(0,100):
#             step = 1e1
#             bound = bf.getPointBound(point, step)
#             temp_res = DDEMC_pure(rf, pf, [bound], 1, 20000)
#             max_err = temp_res[0]
#             if max_err < th:
#                 step = step * 2
#             else:
#                 step = step_back_3v(temp_step,step )
#                 break
#             temp_step = step



def ini_tay2v_build_debug(rf,pf,point,ini_bound,th,n):
    # get the max_err line
    poly_cof = PTB_MaxErr_tracing(point, rf, pf, ini_bound)
    # poly_cof = [1.251135654028887, 2.6362344781912626]
    # poly_cof = [1.1405225286110896, 94.56261414433114]
    # print poly_cof
    inte_points = get_jiaodian(poly_cof, ini_bound, point)
    bound_size = bf.getUlpError(ini_bound[0][0],ini_bound[0][1])
    step = int(bound_size)/200
    # print "inte_points"
    # print point
    # print inte_points
    # print ini_bound
    mid_p = point
    dr = get_direction(inte_points, ini_bound)
    rf_res = rf(*point)
    if fabs(rf_res) < 1:
        temp_bound = generate_bound(point, step)
        temp_bound[1 - dr] = ini_bound[1 - dr]
        point = root_find_rf2v(rf, point, temp_bound)
        # p_val = rf(*point)
    #     if p_val == 0:
    #         p_val = 1e-18
    #     if dr == 0:
    #         new_rf = lambda y: rf(mid_p[0], y)
    #         temp_point = [mid_p[0], float(findroot(new_rf, mid_p[1], tol=p_val * p_val*10))]
    #     else:
    #         new_rf = lambda x: rf(x,mid_p[1])
    #         temp_point = [float(findroot(new_rf, mid_p[0], tol=p_val*p_val*10)),mid_p[1]]
    # if fabs(rf(*point))<fabs(rf(*temp_point)):
    #     point = temp_point
    print "*********"
    print step
    print dr
    print rf_res
    print rf(*point)
    kb_fun = find_accuracy_Roots(rf,point,ini_bound,dr)
    cof = taylor2_horner_cof(rf, point, n)
    cof_err = [float(x - float(x)) for x in cof]
    cof_float = [float(x) for x in cof]
    temp_n = 1
    iter_build_tay2v(th/2.0, ini_bound, ini_bound, rf, pf, n, point, dr, cof_err, cof_float,kb_fun,temp_n)
    return dr
    # dbound = ini_bound[dr]
    # dpoint = point[dr]
    # mid_p1 = dpoint-(dpoint-dbound[0])/3.0
    # mid_p2 = dpoint+(dpoint-dbound[0])/3.0
    # new_bound_l = [[dbound[0], mid_p1], [mid_p1, mid_p2], [mid_p2, dbound[1]]]
    # mid_lst = get_mid_lst(new_bound_l)
    # print mid_lst
# file_name = '../experiments/Localizing_results13/'

# tam = pow(x-pointx,i-j);
#             tbm = pow(y-pointy,j);
#             twoPro(&tam,&tbm);

# void eft_tay2v(double cof_float[],double cof_err[],double pointx,double pointy,double x,double y,double *resf,int size_cof){
#     double ta=0.0;
#     double tam=0.0;
#     double tb=0.0;
#     double tbm=0.0;
#     double ea=0.0;
#     double ea2=0.0;
#     double sa=0.0;
#     double sb=0.0;
#     double res=0.0;
#     double err=0.0;
#     double temp_pow = 0.0;
#     int k = 0;
#     int i,j = 0;
#     for(i=0; i<=size_cof; i++){
#         for(j=0;j<=i;j++){
#             ta = cof_float[k];
#             temp_pow = pow(x-pointx,i-j)*pow(y-pointy,j);
#             tb = temp_pow;
#             twoPro(&ta,&tb);
#             ea = cof_err[k]*temp_pow;
#             ea2 = cof_float[k]*tbm;
#             sa = res;
#             sb = ta;
#             twoSum(&sa,&sb);
#             res = sa;
#             err = err + sb + tb + ea + ea2;
#             k=k+1;
#         }
#     }
#     *resf = res+err;
# }

def bound_divide_3(ini_bound):
    temp_bound = []
    for i in ini_bound:
        temp_size = (i[1]-i[0])/3.0
        temp_bound.append([i[0]+temp_size,i[1]-temp_size])
    input_lst = []
    for i, j in zip(ini_bound, temp_bound):
        input_lst.append([[i[0], j[0]], [j[0], j[1]], [j[1], i[1]]])
    new_bound_l = []
    for element in itertools.product(*input_lst):
        new_bound_l.append(list(element))
    # new_bound_l.remove(temp_bound)
    return new_bound_l,temp_bound



def taylor32_horner_cof(rf,input,order,horder):
    cof_l = []
    mp.dps = 40
    # mp.prec = 106
    for i in range(order+1):
        lst = range(order-i+1)
        lst.reverse()
        for j in lst:
            temp_cof = fdiv(diff(rf, tuple(input), (horder,i, j)), factorial(i) * factorial(j)*factorial(horder))
            cof_l.append(temp_cof)
    return cof_l

def taylor3_horner_cof(rf,input,order):
    cof_l = []
    mp.dps = 40
    # mp.prec = 106
    for i in range(order+1):
        temp_cof = taylor32_horner_cof(rf,input,order-i,i)
        cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l
def get_3v_cof_float(cof,order):
    cof_l = []
    cof_err_l = []
    for i in range(order+1):
        temp_cof_l = []
        temp_cof_err_l = []
        for j in cof[i]:
            temp_cof_l.append(float(j))
            temp_cof_err_l.append(float(j-float(j)))
        cof_l.append(temp_cof_l)
        cof_err_l.append(temp_cof_err_l)
    cof_l.append(order)
    return cof_l,cof_err_l

def taylor3_cof(rf,input,order):
    cof_l = []
    mp.dps = 60
    for i in range(order+1):
        for j in range(0,i+1):
            for p in range(0,j+1):
                temp_cof = fdiv(diff(rf,tuple(input),(i-j,j-p,p)),factorial(i-j)*factorial(j-p)*factorial(p))
                cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l
def taylor3_fun_eft(cof,cof_err,x,y,z,input):
    order = int(cof[-1])
    k = 0
    temp_res = 0.0
    temp_err = 0.0
    for i in range(order+1):
        for j in range(0,i+1):
            for p in range(0,j+1):
                temp_cof = cof[k]
                ta, tb = TwoPro(float(temp_cof), math.pow(x - input[0], i-j)*math.pow(y - input[1], j-p)*math.pow(z-input[2],p))
                ea = cof_err[k]*(math.pow(x - input[0], i-j)*math.pow(y - input[1], j-p)*math.pow(z-input[2],p))
                sa, sb = TwoSum(float(temp_res),ta)
                temp_res = sa
                temp_err = temp_err + sb + tb + ea
                k = k + 1
    return temp_res+temp_err
glob_tay3v_lst = []
def iter_build_tay3v(th,bound,rf,pf,n,cof_float, cof_err,point):
    global glob_tay3v_lst
    if cof_float == []:
        cof = taylor3_horner_cof(rf, point, n)
        cof_float, cof_err = get_3v_cof_float(cof, n)
    tay_fun = lambda x, y, z: taylor3_horner_eft_ori2(cof_float, cof_err, x, y, z, point)
    # cof = taylor3_cof(rf, point, n)
    # print cof
    # print len(cof)
    # cof_err = [float(x - float(x)) for x in cof]
    # cof_float = [float(x) for x in cof]
    # tay_fun = lambda x, y, z: taylor3_fun_eft(cof_float, cof_err, x, y, z, point)
    temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
    max_err2 = temp_res[0]
    input_l = produce_n_input(bound, 10)
    st = time.time()
    pf_res = [tay_fun(*i) for i in input_l]
    rf_res = [rf(*i) for i in input_l]
    err_lst = [bf.getUlpError(i, j) for i, j in zip(pf_res, rf_res)]
    print time.time()-st
    max_err = np.max(err_lst)
    max_err = np.max([max_err, max_err2])
    print max_err
    # print rf(*temp_res[1])
    if max_err < th:
        # st_time = time.time()
        # temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
        # max_err2 = temp_res[0]
        # print max_err2
        if max_err2 < th:
            print "get Here"
            glob_tay3v_lst.append([cof_err, cof_float, point, bound])
            return 0
    new_bounds,temp_bound = bound_divide_3(bound)
    for i in new_bounds:
        temp_res = DDEMC_pure(rf, tay_fun, [i], 1, 20000)
        max_err2 = temp_res[0]
        input_l = produce_n_input(i, 10)
        pf_res = [tay_fun(*d) for d in input_l]
        rf_res = [rf(*d) for d in input_l]
        err_lst = [bf.getUlpError(d, j) for d, j in zip(pf_res, rf_res)]
        max_err = np.max(err_lst)
        max_err2 = np.max([max_err,max_err2])
        print max_err2
        if max_err2 > th:
            if i == temp_bound:
                print "in point"
                iter_build_tay3v(th, i, rf, pf, n, cof_float, cof_err, point)
            else:
                temp_res = DDEMC_pure(rf, pf, [i], 1, 20000)
                # max_err2 = temp_res[0]
                # new_point = get_mid_lst(i)
                new_point = temp_res[1]
                iter_build_tay3v(th, i, rf, pf, n, [],[],new_point)
        else:
            glob_tay3v_lst.append([cof_err, cof_float, point, i])
    return 0

def taylor42_horner_cof(rf,input,order,horder,hhorder):
    cof_l = []
    mp.dps = 40
    # mp.prec = 106
    for i in range(order+1):
        lst = range(order-i+1)
        lst.reverse()
        for j in lst:
            temp_cof = fdiv(diff(rf, tuple(input), (hhorder,horder,i, j)), factorial(i) * factorial(j)*factorial(horder)*factorial(horder))
            cof_l.append(temp_cof)
    return cof_l

def taylor43_horner_cof(rf,input,order,horder):
    cof_l = []
    mp.dps = 40
    # mp.prec = 106
    for i in range(order+1):
        lst = range(order-i+1)
        lst.reverse()
        # for j in lst:
        temp_cof = taylor42_horner_cof(rf,input,order-i,i,horder)
        cof_l.append(temp_cof)
    return cof_l

def taylor4_horner_cof(rf,input,order):
    cof_l = []
    mp.dps = 40
    # mp.prec = 106
    for i in range(order+1):
        lst = range(order-i+1)
        lst.reverse()
        # for j in lst:
        temp_cof = taylor43_horner_cof(rf,input,order-i,i)
        cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l


def get_4v_cof_float(cof,order):
    fcof_l = []
    fcof_err_l = []
    for i in range(order+1):
        cof_l = []
        cof_err_l = []
        for t in cof[i]:
            temp_cof_l = []
            temp_cof_err_l = []
            for j in t:
                temp_cof_l.append(float(j))
                temp_cof_err_l.append(float(j - float(j)))
            cof_l.append(temp_cof_l)
            cof_err_l.append(temp_cof_err_l)
        fcof_l.append(cof_l)
        fcof_err_l.append(cof_err_l)
    fcof_l.append(order)
    return fcof_l,fcof_err_l

def taylor43_horner_eft(cof,x,y,z,input,order):
    k = 0
    lst = range(order+1)
    temp_cof_x = []
    temp_cof_x2 = []
    for i in lst:
        cof1,cof2 = taylor32_horner_eft(cof[i], y, z, input[1:],order-i)
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_res,temp_res1 = horner_eft_1v(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    # temp_err = horner_eft_1vx(temp_cof_x_err, x-input[0])
    return temp_res,temp_res2+temp_res1

def horner_3v(cof,x,y,z,order):
    lst = range(order + 1)
    temp_cof_x = []
    for i in lst:
        cof1 = horner_2v(cof[i],y,z,order-i)
        temp_cof_x.append(cof1)
    res = horner_1v(temp_cof_x,x)
    return res
def taylor4_horner_eft_ori2(cof,cof_err,x,y,z,p,input):
    order = int(cof[-1])
    k = 0
    lst = range(order+1)
    temp_cof_x = []
    temp_cof_x2 = []
    temp_cof_x_err = []
    for i in lst:
        cof1,cof2 = taylor43_horner_eft(cof[i], y,z,p, input[1:],order-i)
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        # err_cof1,err_cof2 = taylor43_horner_eft(cof_err[i], y, z,p, input[1:],order-i)
        # temp_cof_x_err.append(err_cof1+err_cof2)
        temp_cof_x_err.append(horner_3v(cof_err[i],y-input[1],z-input[2],p-input[3],order-i))
        # temp_cof_x_err.append(horner_eft_1vx(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vx(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    temp_err = horner_1v(temp_cof_x_err, x-input[0])
    # temp_err = horner_eft_1vx(temp_cof_x_err, x-input[0])
    return temp_res+temp_res2+temp_err



glob_tay4v_lst = []
def iter_build_tay4v(th,bound,rf,pf,n,point):
    global glob_tay4v_lst
    cof = taylor4_horner_cof(rf, point, n)
    cof_float, cof_err = get_4v_cof_float(cof, n)
    tay_fun = lambda x, y, z,p: taylor4_horner_eft_ori2(cof_float, cof_err, x, y, z,p, point)
    temp_res = DDEMC_pure(rf, tay_fun, [bound], 1, 20000)
    max_err = temp_res[0]
    print max_err
    print rf(*temp_res[1])
    print tay_fun(*temp_res[1])
    if max_err < th:
        # st_time = time.time()
        temp_res = DDEMC_pure(rf, tay_fun, [bound], 3, 20000)
        max_err2 = temp_res[0]
        print max_err2
        if max_err2 < th:
            print "get Here"
            glob_tay4v_lst.append([cof_err, cof_float, point, bound])
            return 0
    new_bounds,temp_bound = bound_divide_3(bound)
    for i in new_bounds:
        temp_res = DDEMC_pure(rf, tay_fun, [i], 1, 20000)
        max_err2 = temp_res[0]
        if max_err2 > th:
            new_point = list(temp_res[1])
            iter_build_tay4v(th, i, rf, pf, n, new_point)
        else:
            glob_tay4v_lst.append([cof_err, cof_float, point, i])
    return 0
