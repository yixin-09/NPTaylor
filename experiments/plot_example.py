import numpy as np
import matplotlib.pyplot as plt
import xlrd
import matplotlib
import os
from matplotlib.patches import Wedge
import math
import xlwt
import ast
import importlib
import bench1v
import bench2v
import bench3v
import bench4v
import pickle
import src.basic_func as bf
import itertools
from mpmath import *


# matplotlib.rcParams['text.usetex']=True
# matplotlib.rcParams['text.latex.unicode']=True
def read_res_from_file(exname,idx,num):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    detecting_res = []
    #get the name
    detecting_res.append(str(table.row_values(idx)[0]))
    #get the max_err
    detecting_res.append(np.max([float(table.row_values(idx)[2]), 1.0]))
    #get the input
    detecting_res.append(ast.literal_eval(table.row_values(idx)[3]))
    #get the input interval
    detecting_res.append(ast.literal_eval(table.row_values(idx)[4]))
    #get the execu time
    detecting_res.append(float(table.row_values(idx)[5]))
    #get the number of repeat
    detecting_res.append(int(table.row_values(idx)[6]))
    # get the number of fit1 (conditon)
    detecting_res.append(int(table.row_values(idx)[7]))
    # get the number of fit2 (err_bits)
    detecting_res.append(int(table.row_values(idx)[8]))
    # detecting_res.append(ast.literal_eval(table.row_values(idx)[10+num-1]))
    return detecting_res

def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l

def point_in_bound(point,bound):
    flag = 1
    for i,j in zip(point,bound):
        if (i<=j[1])&(i>=j[0]):
            flag = 1*flag
        else:
            flag = 0*flag
    return flag
def ls_fun(inp,dr,x,n,kb):
    temp_res = 0
    for i in range(0,n+1):
        temp_res = temp_res+ pow((x-inp[dr]),i)*kb[i]
    return temp_res+inp[1-dr]

def plot_err_inp_debug2D(n_var, id, rd_seed):
    print "Random seed is :" + str(rd_seed)
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, id,1)
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    # refresh()
    pf = ld_module.gfl[id - 1]
    rf = ld_module.rfl[id - 1]
    fnm = detecting_res[0]
    inp = detecting_res[2]
    # new_bound = generate_bound([inp],3e12)[0]
    new_bound = bf.getPointBound(inp, 6e12)
    X = []
    Z = []
    Y = np.random.uniform(new_bound[1][0], new_bound[1][1], 5000)
    for i in Y:
        temp_i = [inp[0],i]
        Z.append(np.log2(float(bf.getUlpError(rf(*temp_i), pf(*temp_i)))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y, Z, '.', color='black')
    ax.set_xlabel('Inputs', fontsize=16)
    ax.plot([new_bound[1][0], new_bound[1][1]], [12, 12], 'r')
    plt.ylim(0, 30)
    plt.xlim(new_bound[1][0], new_bound[1][1])
    ax.annotate(r'$\varepsilon=12$',
                 xy=(new_bound[1][0], 12), xycoords='data',
                 xytext=(2, 2), textcoords='offset points', fontsize=20)
    # plt.ylabel('Repair time ratios',fontsize=20)
    ax.set_ylabel('ErrBits', fontsize=16)
    plt.legend()
    # plt.savefig("graph/example_before2D.pdf", format="pdf")
    plt.savefig("graph/example_after2D.pdf", format="pdf")
    plt.show()

def plot_err_inp_debug(n_var, id, rd_seed):
    print "Random seed is :" + str(rd_seed)
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, id,1)
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    # refresh()
    pf = ld_module.gfl[id - 1]
    rf = ld_module.rfl[id - 1]
    fnm = detecting_res[0]
    inp = detecting_res[2]
    # new_bound = generate_bound([inp],3e12)[0]
    new_bound = bf.getPointBound(inp, 5e12)
    # new_bound = [-226.19467105846508, -226.19464054088897]
    print inp
    print new_bound
    print detecting_res
    kb_fun_file = "/home/yixin/PycharmProjects/NPTaylor/experiments/Localizing_results12/" + fnm + "/kb_fun.txt"
    kb_fun_cof = pickle.load(open(kb_fun_file, "rb"))
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
    print fnm
    fnm.strip()
    bounds_file = "/home/yixin/PycharmProjects/NPTaylor/experiments/Localizing_results12/"+fnm+"/bound_th.txt"
    bounds_th = pickle.load(open(bounds_file, "rb"))
    up_p0 = bounds_th[0]
    kb_up = bounds_th[1]
    down_p0 = bounds_th[2]
    kb_down = bounds_th[3]
    dis = np.fabs(float(kb_up[0]))-np.fabs(float(kb_down[0]))
    kb_up_fun = lambda x: up_p0[1 - dr] + float(kb_up[0]) + (x - up_p0[dr]) * float(kb_up[1])
    # kb_down_fun = lambda x: -up_p0[1 - dr] + float(kb_up[0]) + (x - up_p0[dr]) * float(kb_up[1])
    kb_down_fun = lambda x: down_p0[1 - dr] + float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
    # kb_up_fun = lambda x: -down_p0[1 - dr] - float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
    # kb_down_fun = lambda x: down_p0[1 - dr] + float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
    # kb_down_fun = lambda x: down_p0[1 - dr] + float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
    # line_fun = get_line_fun(rf,new_bound)
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    X = []
    X1 = []
    X2 = []
    Y = []
    Y1 = []
    Y2 = []
    Z = []
    Z2 = []
    Z3 = []
    Z4 = []
    Z5 = []
    iX = np.random.uniform(new_bound[dr][0], new_bound[dr][1],600)
    iY = [kb_fun(i) for i in iX]
    Xk = []
    Yk = []
    Zk = []
    for i,j in zip(iX,iY):
        temp_i = [0,0]
        temp_i[dr]=i
        temp_i[1-dr]=j
        temp_res = np.log2(float(bf.getUlpError(rf(*temp_i), pf(*temp_i))))
        # if temp_res > 60:
        Xk.append(temp_i[0])
        Yk.append(temp_i[1])
        Zk.append(temp_res/2.4)
    input_l = produce_n_input(new_bound,50)
    for i in input_l:
        # temp_res = rf(i)
        temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        # temp_res = np.log2(float(bf.getUlpError(rf(*i),pf(*i))))
        X.append(i[0])
        Y.append(i[1])
        Z.append(float(temp_res))
        # Z.append(rf(*i))
        Z2.append(pf(*i))
        utemp_i = list(i)
        utemp_i[1 - dr] = kb_up_fun(i[dr])
        if point_in_bound(utemp_i,new_bound):
            X1.append(utemp_i[0])
            Y1.append(utemp_i[1])
            Z3.append(12)
        dtemp_i = list(i)
        dtemp_i[1 - dr] = kb_down_fun(i[dr])
        if point_in_bound(dtemp_i,new_bound):
            X2.append(dtemp_i[0])
            Y2.append(dtemp_i[1])
            Z4.append(12)
        Z5.append(12)
        # Z.append(rf(i)-line_fun(i))
    print "max_Z"
    print np.max(Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, '.',color='black')
    ax.plot(X1, Y1, Z3,color='blue',linewidth=5)
    ax.plot(X2, Y2, Z4,color='blue',linewidth=5)
    ax.plot(Xk, Yk, Zk, '.',color='r')
    ax.set_xlabel('Inputs', fontsize=16)
    # plt.ylabel('Repair time ratios',fontsize=20)
    ax.set_zlabel('ErrBits', fontsize=16)
    # ax.scatter(X, Y, Z)
    # ax.scatter(X1, Y1, Z3)
    # ax.scatter(X2, Y2, Z4)
    # ax.scatter(Xk, Yk, Zk)
    # X = np.arange(new_bound[0][0], new_bound[0][1], (new_bound[0][1]-new_bound[0][0])/20.0)
    # Y = np.arange(new_bound[1][0], new_bound[1][1], (new_bound[1][1]-new_bound[1][0])/20.0)
    # X, Y = np.meshgrid(X, Y)
    # # R = np.sqrt(X ** 2 + Y ** 2)
    # Z5 = X*0+Y*0+12
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # Z5 = np.asarray(Z5)
    # ax.plot_surface(X,Y,Z5)
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot(X, Y, Z2, '.')
    # ax = fig.add_subplot(111)
    # ax.plot(X, Z, '.')
    # plt.savefig("graph/example_after.pdf", format="pdf")
    # plt.savefig("graph/example_before.pdf", format="pdf")
    ax.legend()
    plt.show()


def plot_err_inp_debug2(n_var, id, rd_seed):
    print "Random seed is :" + str(rd_seed)
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, id,1)
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    # refresh()
    pf = ld_module.gfl[id - 1]
    rf = ld_module.rfl[id - 1]
    fnm = detecting_res[0]
    inp = detecting_res[2]
    # new_bound = generate_bound([inp],3e12)[0]
    new_bound = bf.getPointBound(inp, 5e12)
    # new_bound[1][1] = 1.0
    # print new_bound
    input_l = produce_n_input(new_bound, 30)
    X = []
    Y = []
    P = []
    Q = []
    Z = []
    for i in input_l:
        # Z.append(1.0/bf.mfitness_fun(rf,pf,i))
        try:
            res = float(pf(*i))
            X.append(i[1])
            Y.append(i[0])
            # P.append(i[2])
            # Q.append(i[3])
            Z.append(res)
        except TypeError:
            continue
    # new_bound = [-226.19467105846508, -226.19464054088897]
    fig = plt.figure()
    print Z
    print len(Z)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z,'.', color='black')
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(Y, P, Z, '.', color='black')
    # ax = fig.add_subplot(143, projection='3d')
    # ax.plot(P, Q, Z, '.', color='black')
    # ax = fig.add_subplot(144, projection='3d')
    # ax.plot(Y, Q, Z, '.', color='black')
    ax.set_xlabel('Inputs', fontsize=16,linespacing=2.1)
    ax.set_zlabel('Outputs', fontsize=16)
    plt.savefig("graph/fail_graph2.pdf", format="pdf")
    ax.legend()
    plt.show()



# for i in [1]:
#     plot_err_inp_debug(2,i, 222)

for i in [1]:
    plot_err_inp_debug(2,i, 222)
    # plot_err_inp_debug2D(2,i, 222)