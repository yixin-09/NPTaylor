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
from src.main import main3v
from src.main import detectHighErrs
import src.basic_func as bf
import matplotlib.pyplot as plt
import itertools
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



def add_bound_back(detecting_res_file,bound,num,idx):
    old_excel = xlrd.open_workbook(detecting_res_file, formatting_info=True)
    lens = 10
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(idx, lens+num-1, str(bound))
    new_excel.save(detecting_res_file)


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
    filename = '../experiments/Localizing_results14/'
    max_ret = [detecting_res[1],inp]
    # bound = detecting_res[-1]
    bound = bf.getPointBound(inp, 5e12)
    # if np.log2(detecting_res[1])>30:
    # res = err_tracing_In_bound(rf,pf,bound,inp,filename,fnm,th)
    # file_name = filename+fnm
    res = main3v(rf, pf, level, th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock)
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



# complete three levels repair
if __name__ == "__main__":
    for i in range(1,3):
        st_time = time.time()
        # os.system("./unpatch_cmd.sh")
        # sudo_cmd("./make_install.sh > tmp_log")
        rd_seed = np.random.randint(0, 1e8, 1)[0]
        np.random.seed(rd_seed)
        th_lst = [16,12]
        num = i
        limit_time = 3600 * 9
        th = th_lst[i-1]
        # id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
        id_lst = [1,4,5,6,7]
        # id_lst = [ 29, 30, 31, 32, 34, 35]
        # id_lst = [102]
        # id_lst = [24]
        index_lst = range(0, len(id_lst))
        level = 0.1*i
        print time.time() - st_time
        filename = "experiment_results3v/repair_results" + str(num) + "/test" + repr(int(level * 10))
        # generate shell scripts to apply patches
        if not os.path.exists(filename + "/patch"):
            # os.remove(filename + "/patch/patch_cmd.sh")
            os.makedirs(filename + "/patch")
        else:
            shutil.rmtree(filename + "/patch")
            os.makedirs(filename + "/patch")
        line_filename = "../experiments/experiment_results3v/repair_results" + str(num) + "/lines" + repr(int(level * 10))
        if not os.path.exists(line_filename):
            os.makedirs(line_filename)
        f = open(filename + "/patch/patch_cmd.sh", "a")
        f.write("#!/usr/bin/env bash\n")
        f.close()
        filename = "experiment_results3v/table_results/"
        if not os.path.exists(filename):
            os.makedirs(filename)
        table_name = "experiment_results3v/table_results/experiment_results_total" + str(num) + ".xls"
        pwd = os.getcwd()
        print pwd
        if not os.path.exists(table_name):
            create_excel(table_name)
        manager = Manager()
        lock = manager.Lock()
        def para_f(d):
            return Test_id_fun(3, d, rd_seed, num, level, limit_time, np.power(2.0, th), table_name,lock)
        print np.min([20, len(id_lst)])
        # for ids in id_lst:
        # for ids in [5]:
        #     para_f(ids)
        # for result in p.imap(para_f, id_lst):
        #     pt_res_excel(result,table_name)
        # para_f(1, lock)
        # result_lst = []
        p = Pool(np.min([5, len(id_lst)]))
        # for result in p.imap(para_f, id_lst):
        #     result_lst.append(result)
        for result in p.imap(para_f, id_lst):
            pt_res_excel(result,table_name)
        p.close()
        p.join()
        # print len(id_lst)
        # print time.time() - st_time



# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# print len(id_lst)
# gsl_sf_bessel_y2
# gsl_sf_bessel_j2