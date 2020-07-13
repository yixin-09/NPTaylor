import numpy as np
import basic_func as bf
from mpmath import *
from localizer import PTB_MaxErr_tracing
import itertools
import sympy as sp
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from detector import DEMC_pure
import os

def pickle_fun(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)

def point_in_bound(point,bound):
    flag = 1
    for i,j in zip(point,bound):
        if (i<=j[1])&(i>=j[0]):
            flag = 1*flag
        else:
            flag = 0*flag
    return flag

def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound

def check_bound_over_inpdm(new_bound,inpdm):
    stop_flag = 0
    for i in range(0,len(new_bound)):
        if new_bound[i][0] <= inpdm[i][0]:
            new_bound[i][0] = inpdm[i][0]
            stop_flag = stop_flag + 1
        if new_bound[i][1] >= inpdm[i][1]:
            new_bound[i][1] = inpdm[i][1]
            stop_flag = stop_flag + 1
    if stop_flag == 2:
        return new_bound,1
    else:
        return new_bound,0
def generate_bound4less1(inp):
    a = np.frexp(inp)
    if inp==0:
        return [-pow(2 ,-1022) ,pow(2 ,-1022)]
    if inp < 0:
        tmp_i = np.ldexp(-0.5, a[1])
        tmp_j = np.ldexp(-0.5, a[1] + 1)
        return [tmp_j, tmp_i]
    else:
        tmp_i = np.ldexp(0.5, a[1])
        tmp_j = np.ldexp(0.5, a[1] + 1)
        return [tmp_i, tmp_j]

# print generate_bound4less1(0)
# print bf.getUlpError(-2.2250738585072014e-308, 2.2250738585072014e-308)
# print np.log2(bf.getUlpError(-2.2250738585072014e-308, 2.2250738585072014e-308))
def get_bound_size(bound):
    bound_size = []
    for i in bound:
        bound_size.append(bf.getUlpError(i[0] ,i[1]))
    return bound_size



def get_repair_bound(inpdm ,inp):
    new_bound = []
    for i, j in zip(inpdm, inp):
        if fabs(j) <= 1.0:
            new_bound.append(generate_bound4less1(j))
        else:
            new_bound.append(i)
    return new_bound
# print get_repair_bound([(524288.0, 1048576.0), (-1.0, 0)],[600652.1227978469, -0.0009123744870894734])
# print get_bound_size([(524288.0, 1048576.0), (-1.0, 0)],600652.1227978469, -0.0009123744870894734)
# for i in get_bound_size([(524288.0, 1048576.0), (-1.0, 0)],600652.1227978469, -0.0009123744870894734):
#     print np.log2(i)
def get_bound_point_size(bound ,point):
    bp_size =[]
    for i, j in zip(bound, point):
        bp_size.append(bf.getUlpError(i[0], j))
        bp_size.append(bf.getUlpError(i[1], j))
    return bp_size


def get_ini_bound(re_bound, inp):
    bound_size = get_bound_size(re_bound)
    print bound_size
    min_size = np.min(bound_size)
    print min_size
    ini_bound = generate_bound(inp, min_size / 20000.0)
    ini_bound, flag = check_bound_over_inpdm(ini_bound, re_bound)
    print ini_bound
    bp_size = get_bound_point_size(ini_bound, inp)
    print bp_size
    min_size = np.min(bp_size)
    print min_size
    ini_bound = generate_bound(inp, min_size / 2.0)
    return ini_bound



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
    # count = 0
    # for i,j in zip(inte_points,ini_bound):
    #     if (i[count]=j[count])
    if (inte_points[0][0]==ini_bound[0][0])&(inte_points[1][0]==ini_bound[0][1]):
        return 0
    else:
        return 1
def get_mid_lst(bound_l):
    mid_lst = []
    for i in bound_l:
        mid_lst.append(i[0]+(i[1]-i[0])/2.0)
    return mid_lst

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
    print kb
    kb_fun = lambda x:float(kb[4]*pow((x-p0[dr]),4.0)+kb[3]*pow((x-p0[dr]),3.0)+kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    return kb_fun

def fake_rf2v(rf,inp):
    return math.fabs(float(rf(*inp)))

def root_find_rf2v(rf,point,new_bound):
    try:
        glob_fitness_con = lambda x: fake_rf2v(rf, x)
        res = differential_evolution(glob_fitness_con, popsize=15, bounds=new_bound, polish=True, strategy='best1bin')
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point

def err_tracing_In_domain2(rf,pf,inpdm,point):
    re_bound = get_repair_bound(inpdm, point)
    bound_size = get_bound_size(re_bound)
    min_size = np.min(bound_size)
    ini_bound = generate_bound(point, min_size / 20000.0)
    poly_cof = PTB_MaxErr_tracing(point, rf, pf, ini_bound)
    # poly_cof = [1.251135654028887, 2.6362344781912626]
    # poly_cof = [1.1405225286110896, 94.56261414433114]
    # print poly_cof
    inte_points = get_jiaodian(poly_cof, ini_bound, point)
    bound_size = bf.getUlpError(ini_bound[0][0], ini_bound[0][1])
    step = int(bound_size) / 200
    mid_p = point
    dr = get_direction(inte_points, ini_bound)
    kb_fun = find_accuracy_Roots(rf, point, ini_bound, dr)
    # rf_res = rf(*point)
    # if fabs(rf_res) < 1:
    #     temp_bound = generate_bound(point, step)
    #     temp_bound[1 - dr] = ini_bound[1 - dr]
    #     point = root_find_rf2v(rf, point, temp_bound)

def generate_around_points(point,step):
    mix_points = []
    for i in point:
        temp_ulp = bf.getulp(i)
        mix_points.append([i-step*temp_ulp,i,i+step*temp_ulp])
    final_points = []
    for i in itertools.product(*mix_points):
        final_points.append(list(i))
    final_points.remove(point)
    return final_points

def ls_fun(inp,dr,x,n,kb):
    temp_res = 0
    for i in range(0,n+1):
        temp_res = temp_res+ pow((x-inp[dr]),i)*kb[i]
    return temp_res+inp[1-dr]

def points_approx(points,dr,n):
    A = []
    B = []
    p0 = points[0]
    mp.dps = 60
    for i in points[1:]:
        # print (i[1-dr]-p0[1-dr])/(i[dr]-p0[dr])
        ta = mpf(i[dr] - p0[dr])
        # ta = mpf(i[0])
        tb = mpf(i[1 - dr] - p0[1 - dr])
        # tb = mpf(i[1])
        # A.append([1,ta])
        # A.append([1.0,ta,pow(ta,2.0),pow(ta,3.0),pow(ta,4.0),pow(ta,5.0),pow(ta,6.0)])
        # A.append([1.0,ta,fmul(ta,ta),pow(ta,3.0)])
        temp_l = []
        for j in range(0,n+1):
            temp_l.append(pow(ta,j))
        A.append(temp_l)
        # A.append([1.0,ta,fmul(ta,ta)])
        # A.append([1.0,ta])
        B.append(tb)
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    kb = list(A.cholesky_solve(B))
    # kb_fun = lambda x:float(kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    # kb_fun = lambda x:float(kb[3]*pow((x-p0[dr]),3.0)+kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    print kb
    kb_fun = lambda x: float(ls_fun(p0,dr,x,n,kb))
    return kb_fun

def points_approx_kb(points,dr,n):
    A = []
    B = []
    p0 = points[0]
    mp.dps = 60
    for i in points[1:]:
        # print (i[1-dr]-p0[1-dr])/(i[dr]-p0[dr])
        ta = mpf(i[dr] - p0[dr])
        # ta = mpf(i[0])
        tb = mpf(i[1 - dr] - p0[1 - dr])
        # tb = mpf(i[1])
        # A.append([1,ta])
        # A.append([1.0,ta,pow(ta,2.0),pow(ta,3.0),pow(ta,4.0),pow(ta,5.0),pow(ta,6.0)])
        # A.append([1.0,ta,fmul(ta,ta),pow(ta,3.0)])
        temp_l = []
        for j in range(0,n+1):
            temp_l.append(pow(ta,j))
        A.append(temp_l)
        # A.append([1.0,ta,fmul(ta,ta)])
        # A.append([1.0,ta])
        B.append(tb)
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    kb = list(A.cholesky_solve(B))
    # kb_fun = lambda x:float(kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    # kb_fun = lambda x:float(kb[3]*pow((x-p0[dr]),3.0)+kb[2]*(x-p0[dr])*(x-p0[dr])+kb[1]*(x-p0[dr])+kb[0]+p0[1-dr])
    print kb
    kb_fun = lambda x: float(ls_fun(p0,dr,x,n,kb))
    return kb_fun,[kb,p0,dr,n]

def fake_rf(rf,inp):
    return fabs(float(rf(*inp)))

def root_find_rf(rf,point,new_bound):
    try:
        glob_fitness_con = lambda x: fake_rf(rf, x)
        res = differential_evolution(glob_fitness_con, popsize=15, bounds=new_bound, polish=True, strategy='best1bin')
        # res = minimize(glob_fitness_con, point, bounds=new_bound)
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point
def get_point_direction(points):
    p0 = points[0]
    p1 = points[-1]
    count = 0
    dr = 0
    temp_dr = 0
    for i,j in zip(p0,p1):
        temp_dis = bf.getUlpError(i,j)
        if temp_dis > temp_dr:
            temp_dr = temp_dis
            dr = count
        count = count + 1
    return dr


def err_tracing_In_domain2(rf,pf,inpdm,point):
    re_bound = get_repair_bound(inpdm, point)
    print "**********"
    # point = [-1.999999998603031e+00, point[1]]
    # print re_bound
    # print rf(*point)
    # print pf(*point)
    # p_val = rf(*point)
    # print bf.getUlpError(rf(*point),pf(*point))
    point = list(point)
    new_rf = lambda y: rf(point[0], y)
    temp_point = [point[0], float(findroot(new_rf, point[1]))]
    print point
    print temp_point
    # print bf.getUlpError(point[1],temp_point[1])
    print rf(*temp_point)
    print rf(*point)
    print pf(*temp_point)
    point = temp_point
    # print bf.getUlpError(rf(*temp_point), pf(*temp_point))
    temp_point = list(point)+[]
    gl_points = []
    step = 10
    app_points = []
    app_points.append(temp_point)
    for ct in range(100):
        ar_ps =  generate_around_points(temp_point,1)
        temp_res = []
        for i in ar_ps:
            if i not in gl_points:
                temp_res.append([fabs(rf(*i)),i,rf(*i),np.log2(bf.getUlpError(rf(*i),pf(*i)))])
                gl_points.append(i)
                gl_points.append(temp_point)
        temp_res.sort()
        # print temp_res[0]
        app_points.append(temp_res[0][1])
        # print gl_points
        # print len(gl_points)
        temp_point = temp_res[0][1]
    dr = get_point_direction(app_points)
    kb_fun = points_approx(app_points, dr,1)
    new_point = point
    app_points = []
    app_points.append(point)
    bound_size = get_bound_size(re_bound)
    min_size = np.min(bound_size)
    p0=point
    step = 1e8
    ulp_p0 = [bf.getulp(i) for i in p0]
    print ulp_p0
    print point
    p_val = rf(*point)
    count = 0
    final_points = []
    final_points.append(point)
    for i in range(0,1000):
        print i
        new_point[dr] = new_point[dr]-ulp_p0[dr]*step
        new_point[1-dr] = kb_fun(new_point[dr])
        print rf(*new_point)
        print new_point
        if dr == 0:
            new_rf = lambda y: rf(new_point[0], y)
            new_point = [new_point[0], findroot(new_rf, new_point[1], tol=p_val * p_val*10)]
        else:
            new_rf = lambda x: rf(x,new_point[1])
            new_point = [findroot(new_rf, new_point[0], tol=p_val*p_val*10),new_point[1]]
        print rf(*new_point)
        print float(rf(float(new_point[0]),float(new_point[1])))
        p_val = float(rf(float(new_point[0]),float(new_point[1])))
        print float(pf(float(new_point[0]),float(new_point[1])))
        # print np.log2(bf.getUlpError(rf(float(new_point[0]),float(new_point[1])),pf(float(new_point[0]),float(new_point[1]))))
        print new_point
        app_points.append(new_point)
        if count == 50:
            dr = get_point_direction(app_points)
            print dr
            kb_fun = points_approx(app_points, dr, 1)
            app_points = []
            # app_points.append(point)
            # app_points.append(new_point)
            count = 0
            step = step*2
        count = count+1
    print point
    print re_bound
    # print kb_fun
    # a = [-1.9999999996485036, -32768.0]
    # new_bound = generate_bound(a,1e11)
    # a = root_find_rf(rf,a,new_bound)
    # kb_fun = points_approx(app_points, 1)
    # print "%.18e" % kb_fun(a[1])
    # print a
    # print rf(*a)
    # print pf(*a)
    # print np.log2(bf.getUlpError(rf(*a), pf(*a)))

def fake_rf_min(rf,x):
    print x
    print rf(*x)
    return float(rf(*x))

def get_point_bound_dis(point,bound):
    pb_dis = []
    for i,j in zip(bound,point):
        pb_dis.append([bf.getUlpError(i[0],j),bf.getUlpError(i[1],j)])
    return pb_dis

def get_points_dis(p1,p2):
    pdis = []
    for i,j in zip(p1,p2):
        pdis.append(bf.getUlpError(i,j))
    return pdis
def build_line_fun(p1,p2,dr):
    k = (p2[1-dr]-p1[1-dr])/(p2[dr]-p1[dr])
    kb_fun = lambda x: float(k*(x-p1[dr]) + p1[1-dr])
    return kb_fun

def gen_new_point(new_point,dr,rf,kb_fun,ulp_p0,step,p_val):
    # print ">>>>>>>"
    # print new_point
    # new_point[dr] = new_point[dr] + ulp_p0[dr] * step
    new_point[dr] = bf.get_next_point(new_point[dr],step,1)
    new_point[1 - dr] = kb_fun(new_point[dr])
    # be_res = float(rf(*new_point))
    # print float(rf(*new_point))
    # print new_point
    # temp_point = []
    # temp_point = new_point
    if dr == 0:
        new_rf = lambda y: rf(new_point[0], y)
        new_point = [new_point[0], findroot(new_rf, new_point[1], tol=p_val * p_val * 10)]
    else:
        new_rf = lambda x: rf(x, new_point[1])
        new_point = [findroot(new_rf, new_point[0], tol=p_val * p_val * 10), new_point[1]]
    return new_point
def gen_final_point(new_point,dr,rf,p_val):
    try:
        print "final_point"
        print new_point
        print rf(*new_point)
        if dr == 0:
            new_rf = lambda y: rf(new_point[0], y)
            new_point = [new_point[0], re(findroot(new_rf, new_point[1], tol=p_val * p_val * 10))]
        else:
            new_rf = lambda x: rf(x, new_point[1])
            new_point = [re(findroot(new_rf, new_point[0], tol=p_val * p_val * 10)), new_point[1]]
        print new_point
        print rf(*new_point)
        return new_point
    except ValueError:
        return new_point

def reach_bound_check(re_bound,point):
    flg = 0
    new_point = point
    count = 0
    idx =0
    for i, j in zip(re_bound, point):
        if (float(j)<=i[0]):
            flg = 1
            new_point[count] = i[0]
            idx = count
        if (float(j)>=i[1]):
            flg = 1
            new_point[count] = i[1]
            idx =count
        count = count+1
    return new_point,flg,idx

def find_bound_point(point,re_bound,dr,p_val,kb_fun,step,rf):
    ulp_p0 = [bf.getulp(i) for i in point]
    temp_point = list(point)
    # try to approximate the max error line in a larger region
    brk_flag = 0
    for ir in range(0, 5):
        app_points = []
        new_point = list(point)
        app_points.append(point)
        for i in range(0, 100):
            new_point = gen_new_point(new_point, dr, rf, kb_fun, ulp_p0, step, p_val)
            # dr = get_point_direction(app_points)
            kb_fun = build_line_fun(temp_point, new_point, dr)
            app_points.append(list(new_point))
            temp_point = list(new_point)
            new_point, brk_flag, idx = reach_bound_check(re_bound, list(new_point))
            if brk_flag == 1:
                break
        if brk_flag == 1:
            break
        kb_fun = points_approx(app_points, dr, 2 + ir)
        step = step * 10
    app_points2 = []
    step = step/10
    new_point = list(point)
    app_points2.append(point)
    count = 200
    for i in range(0, 1003):
        new_point = list(gen_new_point(new_point, dr, rf, kb_fun, ulp_p0, step, p_val))
        # dr = get_point_direction(app_points)
        # print new_point
        fl_point, brk_flag, idx = reach_bound_check(re_bound, list(new_point))
        if brk_flag == 1:
            # print "here"
            # print dr
            # print new_point
            # print temp_point
            # print fl_point
            # print i
            if idx == dr:
                fl_point[1-dr] = mpf(kb_fun(fl_point[dr]))
                # fl_point[1-dr] = mpf(temp_point[1-dr])
            else:
                kb_fun = points_approx(app_points2, 1-dr, 10 + i / 100)
                fl_point[dr] = kb_fun(fl_point[1-dr])
                # fl_point[dr] = mpf(temp_point[dr])
            break
        if len(app_points2)>=200:
            if len(app_points2)==count:
                kb_fun = points_approx(app_points2, dr, 6+i/200)
                count = count + 200
            # if len(app_points2)==400:
            #     kb_fun = points_approx(app_points2, dr, 12)
            # if len(app_points2)==600:
            #     kb_fun = points_approx(app_points2, dr, 12)
            # if len(app_points2)==800:
            #     kb_fun = points_approx(app_points2, dr, 12)
        else:
            kb_fun = build_line_fun(temp_point, new_point, dr)
        app_points2.append(list(new_point))
        temp_point = list(new_point)
    # kb_fun = points_approx(app_points, dr, 6)
    fl_point = gen_final_point(fl_point, idx, rf, p_val)
    app_points2.append(list(fl_point))
    return fl_point,app_points2


# def gen_error_bound(re_bound,kb_fun,kb_fun2,dr):
def gen_error_bound(re_bound,fl_pointl,fl_pointr):
    final_bound = []
    for i,j in zip(fl_pointl,fl_pointr):
        i = float(i)
        j = float(j)
        if i<=j:
            final_bound.append([i,j])
        else:
            final_bound.append([j,i])
    return final_bound

def random_test_err(rf,pf,bound):
    input_l = np.random.uniform(bound[0],bound[1],1000)
    err_lst = []
    for i in input_l:
        temp_err = bf.getUlpError(rf(i), pf(i))
        err_lst.append(temp_err)
    return np.max(err_lst)


def step_back_2v(th,point,step,ori_step,rf,pf,sign,kb_fun,dr,ob):
    print "step_back"
    back_step = step-ori_step
    temp_step = step
    jump_step = back_step/2.0
    while(step-temp_step<back_step):
        temp_dis = bf.get_next_point(point[1-dr],temp_step-jump_step, sign) - point[1-dr]
        temp_kb_fun = lambda x: kb_fun(x)+temp_dis
        if dr == 1:
            new_rf = lambda y: rf(temp_kb_fun(y), y)
            new_pf = lambda y: pf(temp_kb_fun(y), y)
        else:
            new_rf = lambda x: rf(x, temp_kb_fun(x))
            new_pf = lambda x: pf(x, temp_kb_fun(x))
        res = DEMC_pure(new_rf, new_pf, ob, 1, 20000)
        max_err = res[0]
        max_err2 = random_test_err(new_rf, new_pf,ob)
        max_err = np.max([max_err,max_err2])
        print "******"
        print max_err
        print th
        print jump_step
        print temp_step
        print step
        if jump_step<1.0:
            return step
        if (max_err < th):
            if (max_err > th-th/50.0):
                return temp_step
            temp_step=temp_step-jump_step
            jump_step = jump_step/2.0
            if fabs(jump_step)<np.max([temp_step/1e8,10000]):
                return temp_step
        else:
            jump_step = -jump_step/2.0
            # if fabs(jump_step)<np.max([temp_step/1e8,10000]):
            #     return temp_step
    return ori_step

# def step_back_2v2(th, point, step,ori_step, rf, pf, sign,temp_max,kb_fun):
#     step =step
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


def bound_err_Under_th(rf,pf,re_bound,point,th,kb_fun,dr,file_name):
    ib = re_bound[dr]
    ob = re_bound[1-dr]
    ulp_o = bf.getulp(point[1-dr])
    limit_size = bf.getUlpError(ob[0],ob[1])
    step = 1e2
    sign = 1
    up_bound = 0
    final_step = 0
    temp_step = 0
    print limit_size
    for i in range(0,400):
        temp_dis = bf.get_next_point(point[1-dr], step, sign) - point[1-dr]
        # print temp_dis
        # print point
        print step
        print limit_size
        temp_kb_fun = lambda x: kb_fun(x) + temp_dis
        if dr == 1:
            new_rf = lambda y: rf(temp_kb_fun(y), y)
            new_pf = lambda y: pf(temp_kb_fun(y), y)
        else:
            new_rf = lambda x: rf(x, temp_kb_fun(x))
            new_pf = lambda x: pf(x, temp_kb_fun(x))
        res = DEMC_pure(new_rf, new_pf, ib, 1, 20000)
        max_err = res[0]
        max_err2 = random_test_err(new_rf, new_pf, ib)
        # print max_err
        # print th
        # print jump_step
        # print temp_step
        max_err = np.max([max_err, max_err2])
        print max_err
        print th
        try:
            times = np.max([np.log10(max_err / th), 2.0])
        except AttributeError:
            times = 1.5
        if (max_err < th-th/100.0):
            final_step = step_back_2v(th, point, step, temp_step, rf, pf, -sign, kb_fun, dr, ib)
            # if (temp_max<max_err):
            #     final_step = step_back_2v2(th, point, step,temp_step, rf, pf, -sign,temp_max,kb_fun,dr,ob)
            # else:
            #
            break
        temp_step = step
        flag = checkIn_bound(temp_kb_fun,re_bound,dr)
        if flag == 0:
            print "touch flag"
            return []
        if step > limit_size:
            break
        step = int(step * times)
        temp_max = max_err
    up_bound = final_step
    step = 1e2
    sign = -1
    down_bound = 0
    temp_step = 0
    final_step = 0
    for i in range(0, 400):
        # print "right"
        # print step
        temp_dis = bf.get_next_point(point[1-dr], step, sign) - point[1-dr]
        # print temp_dis
        # print point
        temp_kb_fun = lambda x: kb_fun(x) + temp_dis
        if dr == 1:
            new_rf = lambda y: rf(temp_kb_fun(y), y)
            new_pf = lambda y: pf(temp_kb_fun(y), y)
        else:
            new_rf = lambda x: rf(x, temp_kb_fun(x))
            new_pf = lambda x: pf(x, temp_kb_fun(x))
        res = DEMC_pure(new_rf, new_pf, ib, 1, 20000)
        max_err = res[0]
        # print max_err
        # print th
        max_err2 = random_test_err(new_rf, new_pf, ib)
        # print max_err
        # print th
        # print jump_step
        # print temp_step
        max_err = np.max([max_err, max_err2])
        print max_err
        print th
        try:
            times = np.max([np.log10(max_err / th), 2.0])
        except AttributeError:
            times = 1.0
        if (max_err < th-th/100.0):
            final_step = step_back_2v(th, point, step, temp_step, rf, pf, -sign, kb_fun, dr, ib)
            # if (temp_max<max_err):
            #     final_step = step_back_2v2(th, point, step,temp_step, rf, pf, -sign,temp_max,kb_fun,dr,ob)
            # else:
            #
            break
        flag = checkIn_bound(temp_kb_fun, re_bound, dr)
        if flag == 0:
            print "touch flag"
            return []
        if step > limit_size:
            return []
        temp_step = step
        step = int(step * times)
        temp_max = max_err
    down_bound = final_step
    up_dis = bf.get_next_point(point[1-dr], up_bound, 1) - point[1-dr]
    print up_dis
    down_dis = bf.get_next_point(point[1-dr], down_bound, -1) - point[1-dr]
    print down_dis
    # if fabs(up_dis)>fabs(down_dis):
    #     down_dis = -up_dis
    # else:
    #     up_dis = -down_dis
    up_kb_fun = lambda x: kb_fun(x) + up_dis
    down_kb_fun = lambda x: kb_fun(x) + down_dis
    Xi = np.random.uniform(re_bound[dr][0], re_bound[dr][1], 2000)
    Yi = [up_kb_fun(xi) for xi in Xi]
    Yi2 = [down_kb_fun(xi) for xi in Xi]
    # print Yi[0:10]
    # print Yi2[0:10]
    up_points = []
    down_points = []
    for i, j, j2 in zip(Xi, Yi, Yi2):
        temp_i = [0, 0]
        temp_i2 = [0, 0]
        temp_i[dr] = float(i)
        temp_i[1 - dr] = float(j)
        temp_i2[dr] = float(i)
        temp_i2[1 - dr] = float(j2)
        up_points.append(temp_i)
        down_points.append(temp_i2)
    kb_fun_up, kb_up_lst = points_approx_kb(up_points, dr, 1)
    kb_up = kb_up_lst[0]
    kb_fun_down, kb_down_lst = points_approx_kb(down_points, dr, 1)
    kb_down = kb_down_lst[0]
    # print kb_up_lst
    # print kb_down_lst
    # print up_points[0:10]
    # print down_points[0:10]
    add_dis_up = 0
    add_dis_up_lst = []
    for i in up_points:
        temp_dis = i[1-dr]-kb_fun_up(i[dr])
        add_dis_up_lst.append(temp_dis)
    add_dis_up = np.max(add_dis_up_lst)
    # print add_dis_up
    add_dis_down = 0
    add_dis_down_lst = []
    for i in down_points:
        temp_dis = i[1 - dr] - kb_fun_down(i[dr])
        add_dis_down_lst.append(temp_dis)
    add_dis_down = np.max(add_dis_down_lst)
    # print add_dis_down
    kb_up[0]= kb_up[0]+add_dis_up
    kb_down[0]= kb_down[0]+add_dis_down
    store_bound_lst = [up_points[0],kb_up,down_points[0],kb_down]
    kb_up_fun = lambda x:up_points[0][1-dr]+float(kb_up[0])+(x-up_points[0][dr])*float(kb_up[1])
    kb_down_fun = lambda x:down_points[0][1-dr]+float(kb_down[0])+(x-down_points[0][dr])*float(kb_down[1])
    X = []
    X1 = []
    X2 = []
    Y = []
    Y1 = []
    Y2 = []
    Z = []
    Z2 = []
    # input_l = np.random.uniform(new_bound[0], new_bound[1],3000)
    input_l = bf.produce_n_input(re_bound, 100)
    for i in input_l:
        utemp_i = list(i)
        utemp_i[1-dr] = kb_up_fun(i[dr])
        X1.append(utemp_i[0])
        Y1.append(utemp_i[1])
        dtemp_i = list(i)
        dtemp_i[1 - dr] = kb_down_fun(i[dr])
        X2.append(dtemp_i[0])
        Y2.append(dtemp_i[1])
        # temp_res = rf(i)
        # temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        temp_res = np.log2(bf.getUlpError(rf(*i), pf(*i)))
        X.append(i[0])
        Y.append(i[1])
        Z.append(float(temp_res))
        # Z.append(rf(*i))
        Z2.append(12)
        # Z.append(rf(i)-line_fun(i))
    print "max_Z"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, '.')
    ax.plot(X1, Y1, Z2, '.')
    ax.plot(X2, Y2, Z2, '.')
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot(X, Y, Z2, '.')
    # ax = fig.add_subplot(111)
    # ax.plot(X, Z, '.')
    ax.legend()
    fig_name = file_name + "/err_bound.eps"
    plt.savefig(fig_name, format="eps")
    # plt.show()
    return store_bound_lst



def err_tracing_In_bound(rf,pf,re_bound,point,file_name,fun_name,th):
    print "**********"
    print fun_name
    filename = file_name + fun_name
    # generate shell scripts to apply patches
    if not os.path.exists(filename):
        # os.remove(filename + "/patch/patch_cmd.sh")
        os.makedirs(filename)
    point = list(point)
    new_rf = lambda y: rf(point[0], y)
    temp_point = [point[0], float(findroot(new_rf, point[1]))]
    point = list(temp_point)
    temp_point = list(point)
    ulp_p0 = [bf.getulp(i) for i in point]
    gl_points = []
    app_points = []
    app_points.append(temp_point)
    for ct in range(100):
        ar_ps = generate_around_points(temp_point, 1e1)
        temp_res = []
        for i in ar_ps:
            if i not in gl_points:
                temp_res.append([fabs(rf(*i)), i, rf(*i), np.log2(bf.getUlpError(rf(*i), pf(*i)))])
                gl_points.append(i)
                gl_points.append(temp_point)
        temp_res.sort()
        app_points.append(temp_res[0][1])
        temp_point = temp_res[0][1]
    dr = get_point_direction(app_points)
    kb_fun = points_approx(app_points, dr, 1)
    print dr
    p_val = rf(*point)
    final_pointsl = []
    final_pointsr = []
    fl_pointl = []
    fl_pointr = []
    pb_dis = get_point_bound_dis(point, re_bound)
    # print pb_dis
    dr_disr = pb_dis[dr][1]
    dr_disl = pb_dis[dr][0]
    step = np.max([(dr_disr / 1e3) / 1e4, 1000])
    # print step
    # print re_bound
    # print temp_point
    fl_pointl, final_pointsl = find_bound_point(point, re_bound, dr, p_val, kb_fun, step, rf)
    step = -np.max([(dr_disl / 1e3) / 1e4, 1000])
    fl_pointr, final_pointsr = find_bound_point(point, re_bound, dr, p_val, kb_fun, step, rf)
    # print fl_pointl
    # print fl_pointr
    # print rf(*fl_pointl)
    # print rf(*fl_pointr)
    final_pointsl.reverse()
    fl_points = final_pointsl + final_pointsr
    fl_points_name = filename + "/points.txt"
    pickle_fun(fl_points_name, fl_points)
    # final_bound = gen_error_bound(re_bound,fl_pointl,fl_pointr)
    X = []
    Y = []
    Y1 = []
    fresh_points = []
    for i in fl_points:
        fi = [float(i[0]), float(i[1])]
        X.append(float(i[0]))
        # print fi
        temp_err = np.log2(bf.getUlpError(float(rf(*fi)), pf(*fi)))
        if temp_err > 60:
            # print rf(*fi)
            # print pf(*fi)
            # print fi
            fresh_points.append(i)
        # Y.append()
        Y.append(float(i[1]))
        Y1.append(temp_err)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(X, Y, '.')
    ax2 = fig.add_subplot(122)
    ax2.plot(X, Y1, '.')
    ax.legend()
    fig_name = filename + "/max_err_line.eps"
    plt.savefig(fig_name, format="eps")
    plt.close()
    # plt.show()
    print re_bound
    print len(fresh_points)
    kb_fun, kb = points_approx_kb(fresh_points, dr,10)
    bound_th_res = bound_err_Under_th(rf, pf, re_bound, point, th, kb_fun, dr,filename)
    bound_th_name = filename + "/bound_th.txt"
    pickle_fun(bound_th_name, bound_th_res)
    kb_fun_name = filename + "/kb_fun.txt"
    pickle_fun(kb_fun_name, kb)
    # kb_fun2 = points_approx(fresh_points, 1-dr, 17)
    final_bound = gen_error_bound(re_bound, fl_pointl, fl_pointr)
    print final_bound
    print re_bound
    bounds_c = [final_bound, re_bound]
    bounds_name = filename + "/bounds.txt"
    pickle_fun(bounds_name, bounds_c)
    Xi = np.random.uniform(final_bound[dr][0], final_bound[dr][1], 2000)
    Yi = [kb_fun(xi) for xi in Xi]
    Yi2 = [kb_fun(xi) + 1e12 * ulp_p0[1 - dr] for xi in Xi]
    X = []
    Y = []
    Y1 = []
    Y2 = []
    for i, j, j2 in zip(Xi, Yi, Yi2):
        temp_i = [0, 0]
        temp_i2 = [0, 0]
        temp_i[dr] = float(i)
        temp_i[1 - dr] = float(j)
        temp_i2[dr] = float(i)
        temp_i2[1 - dr] = float(j2)
        a = rf(*temp_i)
        b = pf(*temp_i)
        c = rf(*temp_i2)
        d = pf(*temp_i2)
        X.append(temp_i[0])
        Y.append(temp_i[1])
        # Y.append(a)
        # print temp_i
        # print temp_i2
        # print a
        # print b
        # print np.log2(float(bf.getUlpError(a, b)))
        Y1.append(np.log2(float(bf.getUlpError(a, b))))
        Y2.append(np.log2(float(bf.getUlpError(c, d))))
    # print np.max(Y)
    # print np.min(Y)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(X, Y1, '.')
    ax2 = fig.add_subplot(132)
    ax2.plot(X, Y2, '.')
    ax3 = fig.add_subplot(133)
    ax3.plot(X, Y, '.')
    ax.legend()
    # plt.show()
    fig_name = filename + "/app_max_err_line.eps"
    plt.savefig(fig_name, format="eps")
    return 0

def err_tracing_In_domain(rf,pf,inpdm,point,file_name,fun_name):
    re_bound = get_repair_bound(inpdm, point)
    print "**********"
    print "fun_name"
    filename = file_name + fun_name
    # generate shell scripts to apply patches
    if not os.path.exists(filename):
        # os.remove(filename + "/patch/patch_cmd.sh")
        os.makedirs(filename)
    point = list(point)
    new_rf = lambda y: rf(point[0], y)
    temp_point = [point[0], float(findroot(new_rf, point[1]))]
    point = list(temp_point)
    temp_point = list(point)
    ulp_p0 = [bf.getulp(i) for i in point]
    gl_points = []
    app_points = []
    app_points.append(temp_point)
    for ct in range(100):
        ar_ps =  generate_around_points(temp_point,1e1)
        temp_res = []
        for i in ar_ps:
            if i not in gl_points:
                temp_res.append([fabs(rf(*i)),i,rf(*i),np.log2(bf.getUlpError(rf(*i),pf(*i)))])
                gl_points.append(i)
                gl_points.append(temp_point)
        temp_res.sort()
        app_points.append(temp_res[0][1])
        temp_point = temp_res[0][1]
    dr = get_point_direction(app_points)
    kb_fun = points_approx(app_points, dr,1)
    print dr
    p_val = rf(*point)
    final_pointsl = []
    final_pointsr = []
    fl_pointl = []
    fl_pointr = []
    pb_dis = get_point_bound_dis(point,re_bound)
    # print pb_dis
    dr_disr = pb_dis[dr][1]
    dr_disl = pb_dis[dr][0]
    step = np.max([(dr_disr/1e3)/1e4,1000])
    # print step
    # print re_bound
    # print temp_point
    fl_pointl,final_pointsl = find_bound_point(point, re_bound, dr, p_val, kb_fun, step,rf)
    step = -np.max([(dr_disl/1e3)/1e4,1000])
    fl_pointr, final_pointsr = find_bound_point(point, re_bound, dr, p_val, kb_fun, step,rf)
    # print fl_pointl
    # print fl_pointr
    # print rf(*fl_pointl)
    # print rf(*fl_pointr)
    final_pointsl.reverse()
    fl_points = final_pointsl+final_pointsr
    fl_points_name = filename + "/points.txt"
    pickle_fun(fl_points_name, fl_points)
    # final_bound = gen_error_bound(re_bound,fl_pointl,fl_pointr)
    X = []
    Y = []
    Y1 = []
    fresh_points = []
    for i in fl_points:
        fi = [float(i[0]),float(i[1])]
        X.append(float(i[0]))
        # print fi
        temp_err = np.log2(bf.getUlpError(float(rf(*fi)),pf(*fi)))
        if temp_err>60:
            # print rf(*fi)
            # print pf(*fi)
            # print fi
            fresh_points.append(i)
        # Y.append()
        Y.append(float(i[1]))
        Y1.append(temp_err)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(X, Y, '.')
    ax2 = fig.add_subplot(122)
    ax2.plot(X, Y1, '.')
    ax.legend()
    fig_name = filename+"/max_err_line.eps"
    plt.savefig(fig_name, format="eps")
    # plt.show()
    print re_bound
    print len(fresh_points)
    kb_fun,kb = points_approx_kb(fresh_points, dr, 24)
    kb_fun_name = filename + "/kb_fun.txt"
    pickle_fun(kb_fun_name, kb)
    # kb_fun2 = points_approx(fresh_points, 1-dr, 17)
    final_bound = gen_error_bound(re_bound,fl_pointl,fl_pointr)
    print final_bound
    print re_bound
    bounds_c = [final_bound,re_bound]
    bounds_name = filename + "/bounds.txt"
    pickle_fun(bounds_name, bounds_c)
    Xi = np.random.uniform(final_bound[dr][0], final_bound[dr][1],2000)
    Yi = [kb_fun(xi) for xi in Xi]
    Yi2 = [kb_fun(xi)+1e12*ulp_p0[1-dr] for xi in Xi]
    X = []
    Y = []
    Y1 = []
    Y2 = []
    for i,j,j2 in zip(Xi,Yi,Yi2):
        temp_i = [0, 0]
        temp_i2 = [0, 0]
        temp_i[dr] = float(i)
        temp_i[1 - dr] = float(j)
        temp_i2[dr] = float(i)
        temp_i2[1 - dr] = float(j2)
        a = rf(*temp_i)
        b = pf(*temp_i)
        c = rf(*temp_i2)
        d = pf(*temp_i2)
        X.append(temp_i[0])
        Y.append(temp_i[1])
        # Y.append(a)
        # print temp_i
        # print temp_i2
        # print a
        # print b
        # print np.log2(float(bf.getUlpError(a, b)))
        Y1.append(np.log2(float(bf.getUlpError(a, b))))
        Y2.append(np.log2(float(bf.getUlpError(c, d))))
    # print np.max(Y)
    # print np.min(Y)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(X, Y1, '.')
    ax2 = fig.add_subplot(132)
    ax2.plot(X, Y2, '.')
    ax3 = fig.add_subplot(133)
    ax3.plot(X, Y, '.')
    ax.legend()
    # plt.show()
    fig_name = filename + "/app_max_err_line.eps"
    plt.savefig(fig_name, format="eps")


def load_test(rf,pf,kb_fun_name,final_bound,ulp_p0):
    kb_fun_cof = pickle.load(open(kb_fun_name, "rb"))
    kb = kb_fun_cof[0]
    p0 = kb_fun_cof[1]
    dr = kb_fun_cof[2]
    n = kb_fun_cof[3]
    kb_fun3 = lambda x: float(ls_fun(p0, dr, x, n, kb))
    Xi = np.random.uniform(final_bound[dr][0], final_bound[dr][1], 2000)
    Yi = [kb_fun3(xi) for xi in Xi]
    Yi2 = [kb_fun3(xi) + 1e12 * ulp_p0[1 - dr] for xi in Xi]
    X = []
    Y = []
    Y1 = []
    Y2 = []
    for i, j, j2 in zip(Xi, Yi, Yi2):
        temp_i = [0, 0]
        temp_i2 = [0, 0]
        temp_i[dr] = float(i)
        temp_i[1 - dr] = float(j)
        temp_i2[dr] = float(i)
        temp_i2[1 - dr] = float(j2)
        a = rf(*temp_i)
        b = pf(*temp_i)
        c = rf(*temp_i2)
        d = pf(*temp_i2)
        X.append(temp_i[0])
        Y.append(temp_i[1])
        # Y.append(a)
        # print temp_i
        # print temp_i2
        # print a
        # print b
        # print np.log2(float(bf.getUlpError(a, b)))
        Y1.append(np.log2(float(bf.getUlpError(a, b))))
        Y2.append(np.log2(float(bf.getUlpError(c, d))))
    # print np.max(Y)
    # print np.min(Y)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(X, Y1, '.')
    ax2 = fig.add_subplot(132)
    ax2.plot(X, Y2, '.')
    ax3 = fig.add_subplot(133)
    ax3.plot(X, Y, '.')
    ax.legend()
    plt.show()
#gsl_sf_conicalP_half,gsl_sf_conicalP_0

