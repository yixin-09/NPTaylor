import xlrd
import numpy as np
import ast
import time
import importlib
import bench1v
import bench2v
import bench3v
import bench4v
import os
import sys
import xlwt
from xlutils.copy import copy
from src.main import main2v
from src.main import detectHighErrs
import src.basic_func as bf
import matplotlib.pyplot as plt
import itertools
import matplotlib
import shutil
from multiprocessing import Pool,Lock,Manager
import pickle
from mpl_toolkits.mplot3d import Axes3D
from src.Localize_err_bound import err_tracing_In_bound

def sudo_cmd(cmd):
    sudoPassword = 'hello'
    os.system('echo %s|sudo -S %s' % (sudoPassword, cmd))
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

def refresh():
    command = "rm -rf ../benchmarks/GSL_function/specfunc4patch/"
    command2 = "cp -R ../benchmarks/GSL_function/specfunc/. ../benchmarks/GSL_function/specfunc4patch/"
    os.system(command)
    os.system(command2)
def create_excel(table_name):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "id")
    sheet.write(0, 1, "Programs")
    sheet.write(0, 2, "Threshold")
    sheet.write(0, 3, "Bound")
    sheet.write(0, 4, "Bound_distance")
    sheet.write(0, 5, "RandomSeed")
    # sheet.write(0, 5, "PTB")
    sheet.write(0, 6, "Repair")
    # sheet.write(0, 7, "Total")
    sheet.write(0, 7, "Patch Size")
    sheet.write(0, 8, "Line number")
    book.save(table_name)


def pt_res_excel(res,table_name):
    old_excel = xlrd.open_workbook(table_name)
    table = old_excel.sheets()[0]
    lens = table.nrows
    i = lens
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(i, 0, str(res[0]))
    sheet.write(i, 1, res[1])
    sheet.write(i, 2, repr(res[2]))
    sheet.write(i, 3, str(res[4]))
    sheet.write(i, 4, repr(res[5]))
    # sheet.write(i, 5, res[2])
    sheet.write(i, 5, repr(res[8]))
    # sheet.write(i, 7, res[6])
    sheet.write(i, 6, repr(res[7]))
    sheet.write(i, 7, str(res[9]))
    sheet.write(i, 8, str(res[10]))
    new_excel.save(table_name)
# res = [2, 'gsl_sf_airy_Bi', 1048576.0, 0.0004048347473144531, [-422.09747923595603, -422.09595167060695], 26873213817, 1.7894492149353027, 1.7901890277862549, 27023943, 1910, 3]
# pt_res_excel(res,'experiment_results/table_results/experiment_results_total2.xls')
def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound


def get_line_fun(rf,bound):
    x0 = bound[0]
    x1 = bound[1]
    try:
        y0 = rf(bound[0])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y0 = rf(bound[0] + bf.getulp(bound[0]))
    try:
        y1 = rf(bound[1])
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        y1 = rf(bound[1] - bf.getulp(bound[1]))
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
    return lambda x: (x - s_x) * k + s_y

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
    dr = 0
    print fnm
    fnm.strip()
    bounds_file = "/home/yixin/PycharmProjects/NPTaylor/experiments/Localizing_results12/"+fnm+"/bound_th.txt"
    bounds_th = pickle.load(open(bounds_file, "rb"))
    up_p0 = bounds_th[0]
    kb_up = bounds_th[1]
    down_p0 = bounds_th[2]
    kb_down = bounds_th[3]
    kb_up_fun = lambda x: up_p0[1 - dr] + float(kb_up[0]) + (x - up_p0[dr]) * float(kb_up[1])
    kb_down_fun = lambda x: down_p0[1 - dr] + float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
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
    # input_l = np.random.uniform(new_bound[0], new_bound[1],3000)
    input_l = produce_n_input(new_bound,100)
    for i in input_l:
        # temp_res = rf(i)
        # temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        temp_res = np.log2(bf.getUlpError(rf(*i),pf(*i)))
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
    ax.plot(X, Y, Z, '.')
    ax.plot(X1, Y1, Z3, '.')
    ax.plot(X2, Y2, Z4, '.')
    # X = np.arange(new_bound[0][0], new_bound[0][1], (new_bound[0][1]-new_bound[0][0])/20.0)
    # Y = np.arange(new_bound[1][0], new_bound[1][1], (new_bound[1][1]-new_bound[1][0])/20.0)
    # plt.zlim(0, 64)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X ** 2 + Y ** 2)
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
    ax.legend()
    plt.show()

def plot_err_inp_debug2d(n_var, id, rd_seed):
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
    dr = 0
    print fnm
    fnm.strip()
    bounds_file = "/home/yixin/PycharmProjects/NPTaylor/experiments/Localizing_results12/"+fnm+"/bound_th.txt"
    bounds_th = pickle.load(open(bounds_file, "rb"))
    up_p0 = bounds_th[0]
    kb_up = bounds_th[1]
    down_p0 = bounds_th[2]
    kb_down = bounds_th[3]
    kb_up_fun = lambda x: up_p0[1 - dr] + float(kb_up[0]) + (x - up_p0[dr]) * float(kb_up[1])
    kb_down_fun = lambda x: down_p0[1 - dr] + float(kb_down[0]) + (x - down_p0[dr]) * float(kb_down[1])
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
    # input_l = np.random.uniform(new_bound[0], new_bound[1],3000)
    input_l = produce_n_input(new_bound,100)
    for i in input_l:
        # temp_res = rf(i)
        # temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        temp_res = np.log2(bf.getUlpError(rf(*i),pf(*i)))
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
    ax.plot(X, Y, Z, '.')
    ax.plot(X1, Y1, Z3, '.')
    ax.plot(X2, Y2, Z4, '.')
    X = np.arange(new_bound[0][0], new_bound[0][1], (new_bound[0][1]-new_bound[0][0])/20.0)
    Y = np.arange(new_bound[1][0], new_bound[1][1], (new_bound[1][1]-new_bound[1][0])/20.0)
    X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X ** 2 + Y ** 2)
    Z5 = X*0+Y*0+12
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # Z5 = np.asarray(Z5)
    ax.plot_surface(X,Y,Z5)
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot(X, Y, Z2, '.')
    # ax = fig.add_subplot(111)
    # ax.plot(X, Z, '.')
    ax.legend()
    plt.show()
def plot_test2(f1,f2,a,name,th,name2,dis2,inp,a1):
    n = 10000
    if n>dis2:
        n = dis2
    X = sorted(np.random.uniform(a[0],a[1],n))
    print "%.18e" % a[0]
    print "%.18e" % a[1]
    glob_fitness_fun = np.frompyfunc(lambda x: bf.fitness_fun(f1, f2, x), 1, 1)
    temp_res = glob_fitness_fun(X)
    l = []
    #re_err_1 = [math.log(distan_cal(op_r[0],op_f[0])+1,2) for op_f, op_r in zip(res_d, res_mp)]
    re_err_1 = [np.log2(np.max([1.0/r,1.0])) for r in temp_res]
    # re_err_1 = [op_r[0]-op_f[0] for op_f, op_r in zip(res_d, res_mp)]
    # re_err_1 = [op_r[0] for op_f, op_r in zip(res_d, res_mp)]
    #re_err_1 = [bf.getRelError(op_r[0],op_f[0]) for op_f, op_r in zip(res_d, res_mp)]
    plt.figure()
    plt.scatter(X, re_err_1,s=2,c='k')
    varep = np.max(re_err_1)
    print min(re_err_1)
    print max(re_err_1)
    print varep
    varep = th
    plt.plot([a[0],a[1]],[varep,varep],'r')
    plt.plot([inp,inp],[0,25],'g--')
    plt.plot([a1[0],a1[0]],[0,64],'b-')
    plt.plot([a1[1],a1[1]],[0,64],'b-')
    th = "%.1f" % th
    plt.annotate(r'$\varepsilon=$'+str(th),
                 xy=(X[n/8],varep), xycoords='data',
                 xytext=(0,10), textcoords='offset points', fontsize=20)
    plt.annotate(r'$x_0=$' + str(inp),
                 xy=(inp-(inp-a1[0])/4.0, 20), xycoords='data',
                 xytext=(0, 5), textcoords='offset points', fontsize=15)
    plt.annotate(s='', xy=(a1[0], 40), xytext=(a1[1], 40), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(r'$I_{err}$',
                 xy=(inp, 40), xycoords='data',
                 xytext=(0, 5), textcoords='offset points', fontsize=20)
    plt.annotate(str(a1),
                 xy=(a1[0]+(inp-a1[0])/6.0, 36), xycoords='data',
                 xytext=(0, 5), textcoords='offset points', fontsize=11)
    matplotlib.rc('xtick', labelsize = 10)
    matplotlib.rc('ytick', labelsize = 20)
    plt.rc('ytick', labelsize=10)
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_fontsize(10)
    plt.ylabel(r"$ErrBits$",fontsize=12)
    plt.xlabel("Input",fontsize=12)
    plt.xlim(a[0],a[1])
    plt.ylim(0,64)
    # plt.ylim(np.min(re_err_1),np.max(re_err_1))
    #plt.yticks([0, 2e-7, 4e-7, 6e-7, 8e-7],['0','2e-7','4e-7','6e-7','8e-7'])
    #plt.yticks([0, 3e-14, 6e-14, 9e-14, 1e-13], ['0', '3e-14', '6e-14', '9e-14', '1e-13'])
    plt.savefig("graph/example/"+name2+name+".png", format="png")
    plt.show()
def plot_err_inp(n_var, id, rd_seed):
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
    dr = 0
    print fnm
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
    # input_l = np.random.uniform(new_bound[0], new_bound[1],3000)
    input_l = produce_n_input(new_bound,100)
    for i in input_l:
        # temp_res = rf(i)
        # temp_res = np.log2(float(1.0 / glob_fitness_real(i)))
        temp_res = np.log2(bf.getUlpError(rf(*i),pf(*i)))
        X.append(i[0])
        Y.append(i[1])
        Z.append(float(temp_res))
        # Z.append(rf(*i))
        Z2.append(pf(*i))
        # Z.append(rf(i)-line_fun(i))
    print "max_Z"
    print np.max(Z)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(X, Y, Z, '.')
    # ax = fig.add_subplot(111)
    # ax.plot(X, Z, '.')
    ax.legend()
    plt.show()
# [1,2,16,27,32,24]
# [6,17,22,25,28,29,30,31,34,35]
for i in [1]:
    plot_err_inp_debug(2,i, 222)

def add_bound_back(detecting_res_file,bound,num,idx):
    old_excel = xlrd.open_workbook(detecting_res_file, formatting_info=True)
    lens = 10
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(idx, lens+num-1, str(bound))
    new_excel.save(detecting_res_file)

def generate_bound_3level(n_var,idx,num):
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, idx)
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    pf = ld_module.gfl[idx - 1]
    rf = ld_module.rfl[idx - 1]
    th_lst = [24,20,16]
    th = np.power(2.0,th_lst[num-1])
    inp = detecting_res[2]
    max_ret = [detecting_res[1], inp]
    max_x, bound, bound_l = detectHighErrs(max_ret, th, rf, pf)
    add_bound_back(detecting_res_file,bound,num,idx)
# Test_id_fun(1, i, rd_seed, 1, 0.1, limit_time, np.power(2.0, 24),"ddd")
def Test_id_fun(n_var,idx,rd_seed,num,level,limit_time,th,table_name,lock):
    print "Random seed is :" + str(rd_seed)
    pwd = os.getcwd()
    detecting_res_file = pwd+"/final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, idx,num)
    print "hello"
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    # refresh()
    pf = ld_module.gfl[idx - 1]
    rf = ld_module.rfl[idx - 1]
    fnm = detecting_res[0]
    inp = detecting_res[2]
    print inp
    print detecting_res
    filename = '../experiments/Localizing_results12/'
    max_ret = [detecting_res[1],inp]
    # bound = detecting_res[-1]
    bound = bf.getPointBound(inp, 5e12)
    # if np.log2(detecting_res[1])>30:
    # res = err_tracing_In_bound(rf,pf,bound,inp,filename,fnm,th)
    file_name = filename+fnm
    res = main2v(rf, pf, level, th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock,file_name)
    return res
        # ln = [0, 3, 2, 1]
        # k = ln[int(level * 10)]
        # table_name = "experiment_results/table_results/experiment_results_total" + str(num) + ".xls"
        # if not os.path.exists(table_name):
        #     create_excel(table_name)
        # old_excel = xlrd.open_workbook(table_name, formatting_info=True)
        # new_excel = copy(old_excel)
        # sheet = new_excel.get_sheet(0)
        # sheet.write(i + k, 0, res[0])
        # sheet.write(i + k, 1, res[1])
        # sheet.write(i + k, 2, str(res[3]))
        # sheet.write(i + k, 3, res[4])
        # sheet.write(i + k, 4, rd_seed)
        # sheet.write(i + k, 5, "After")
        # sheet.write(i + k, 6, res[2])
        # sheet.write(i + k, 7, res[5])
        # sheet.write(i + k, 8, res[6])
        # sheet.write(i + k, 9, res[8])
        # sheet.write(i + k, 10, res[9])
        # new_excel.save(table_name)

def Test_id(n_var,idx):
    detecting_res_file = "/home/yixin/PycharmProjects/AutoEFT/experiments/final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, idx)
    if np.log2(detecting_res[1])>30:
        return idx
    else:
        return 0
# id_lst = []
# for i in range(1,105):
#     res = Test_id(1,i)
#     if res!=0:
#         id_lst.append(res)
# print id_lst

def test_f(i):
    global glob_res
    st_time = time.time()
    rd_seed = np.random.randint(0, 1e8, 1)[0]
    np.random.seed(rd_seed)
    repair_enable = 1
    num = 1
    # for i in range(1, 3):
    limit_time = 3600 * 3
    manager = Manager()
    lock = manager.Lock()
    print "*****************************************************"
    Test_id_fun(1, i, rd_seed, 3, 0.3, limit_time, np.power(2.0, 16),"experiment_results/table_results/experiment_results_total3.xls",lock)
    # pt_res_excel(glob_res, 111)
    print "Total time is :" + repr(time.time() - st_time)

# test_f(4)
# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103,
#           104]
# for i in id_lst:
#     test_f(i)
# def para_f(i):


def store_bound():
    for k in range(1,4):
        id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100,102, 103, 104]
        def para_f(i):
            generate_bound_3level(1,i,k)
        p = Pool(np.min([24, len(id_lst)]))
        p.map(para_f, id_lst)
        p.close()
        p.join()


# complete three levels repair
# if __name__ == "__main__":
#     for i in range(1,3):
#         st_time = time.time()
#         # os.system("./unpatch_cmd.sh")
#         # sudo_cmd("./make_install.sh > tmp_log")
#         rd_seed = np.random.randint(0, 1e8, 1)[0]
#         np.random.seed(rd_seed)
#         th_lst = [16,12]
#         num = i
#         # limit_time = 3600 * 6
#         limit_time = 3
#         th = th_lst[i-1]
#         # id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
#         id_lst = [1, 2, 6, 16, 17, 22, 24, 25, 26,27, 28, 29, 30, 31, 32,33, 34, 35]
#         # id_lst = [35]
#         # id_lst = [ 29, 30, 31, 32, 34, 35]
#         # id_lst = [102]
#         # id_lst = [24]
#         index_lst = range(0, len(id_lst))
#         level = 0.1*i
#         print time.time() - st_time
#         filename = "experiment_results/repair_results" + str(num) + "/test" + repr(int(level * 10))
#         # generate shell scripts to apply patches
#         if not os.path.exists(filename + "/patch"):
#             # os.remove(filename + "/patch/patch_cmd.sh")
#             os.makedirs(filename + "/patch")
#         else:
#             shutil.rmtree(filename + "/patch")
#             os.makedirs(filename + "/patch")
#         line_filename = "../experiments/experiment_results/repair_results" + str(num) + "/lines" + repr(int(level * 10))
#         if not os.path.exists(line_filename):
#             os.makedirs(line_filename)
#         f = open(filename + "/patch/patch_cmd.sh", "a")
#         f.write("#!/usr/bin/env bash\n")
#         f.close()
#         filename = "experiment_results/table_results/"
#         if not os.path.exists(filename):
#             os.makedirs(filename)
#         table_name = "experiment_results/table_results/experiment_results_total" + str(num) + ".xls"
#         pwd = os.getcwd()
#         print pwd
#         if not os.path.exists(table_name):
#             create_excel(table_name)
#         manager = Manager()
#         lock = manager.Lock()
#         def para_f(d):
#             return Test_id_fun(2, d, rd_seed, num, level, limit_time, np.power(2.0, th), table_name,lock)
#         print np.min([20, len(id_lst)])
#         # for ids in id_lst:
#         # for ids in [2]:
#         #     para_f(ids)
#         # for result in p.imap(para_f, id_lst):
#         #     pt_res_excel(result,table_name)
#         # para_f(1, lock)
#         result_lst = []
#         p = Pool(np.min([16, len(id_lst)]))
#         # for result in p.imap(para_f, id_lst):
#         #     result_lst.append(result)
#         for result in p.imap(para_f, id_lst):
#             pt_res_excel(result,table_name)
#         p.close()
#         p.join()
        # print len(id_lst)
        # print time.time() - st_time



# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# print len(id_lst)
# gsl_sf_bessel_y2
# gsl_sf_bessel_j2