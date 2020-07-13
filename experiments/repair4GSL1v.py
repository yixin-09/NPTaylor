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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import src.basic_func as bf
from src.detector1v import detectHighErrs
import shutil
from multiprocessing import Pool,Lock,Manager
from src.main import main1v
from src.main import main1v_tay


def sudo_cmd(cmd):
    sudoPassword = 'hello'
    os.system('echo %s|sudo -S %s' % (sudoPassword, cmd))

# read results from final_results_1v.xls
def read_res_from_file1(exname,idx):
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
    detecting_res.append(ast.literal_eval(table.row_values(idx)[10+num-1]))
    return detecting_res
# get the id of High floating-point errors
def Test_id(n_var,idx):
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, idx)
    if np.log2(detecting_res[1])>32:
        return idx
    else:
        return 0


def add_bound_back(detecting_res_file,bound,num,idx):
    old_excel = xlrd.open_workbook(detecting_res_file, formatting_info=True)
    lens = 10
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    sheet.write(idx, lens+num-1, str(bound))
    new_excel.save(detecting_res_file)
def generate_bound_3level(n_var,idx,num):
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file1(detecting_res_file, idx)
    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    # change it to youself system password
    password = "hello"
    pf = ld_module.gfl[idx - 1]
    rf = ld_module.rfl[idx - 1]
    th_lst = [16,12]
    th = np.power(2.0,th_lst[num-1])
    inp = detecting_res[2]
    max_ret = [detecting_res[1], inp]
    print th
    max_x, bound, bound_l = detectHighErrs(max_ret, th, rf, pf)
    print bf.getUlpError(bound[0],bound[1])
    add_bound_back(detecting_res_file,bound,num,idx)
    return 0
# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# def para_f(d):
#     generate_bound_3level(1, d, 2)
#
# for i in [100, 102, 103, 104]:
#     print i
#     para_f(i)
def read_bound(n_var,idx,num):
    detecting_res_file = "final_results_" + str(n_var) + "v.xls"
    detecting_res = read_res_from_file(detecting_res_file, idx,num)
    bound = detecting_res[-1]
    print "%.15e" % float(bf.getUlpError(bound[0],bound[1]))


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
def store_bound():
    for k in range(1,4):
        id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100,102, 103, 104]
        def para_f(i):
            generate_bound_3level(1,i,k)
        p = Pool(np.min([24, len(id_lst)]))
        p.map(para_f, id_lst)
        p.close()
        p.join()

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
def Test_id_fun(n_var,idx,rd_seed,num,level,limit_time,th,table_name,lock):
    print "Random seed is :" + str(rd_seed)
    pwd = os.getcwd()
    detecting_res_file = pwd+"/final_results_" + str(n_var) + "v.xls"
    print detecting_res_file
    detecting_res = read_res_from_file(detecting_res_file, idx,num)
    print "hello"
    bound = detecting_res[-1]
    print "%.15e" % float(bf.getUlpError(bound[0], bound[1]))
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
    max_ret = [detecting_res[1],inp]
    bound = detecting_res[-1]
    if np.log2(detecting_res[1])>32:
        res = main1v_tay(rf, pf, level, th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock)
        # res = main1v(rf, pf, level, th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock)
        return res

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

# complete two levels repair
if __name__ == "__main__":
    for i in range(1,3):
        st_time = time.time()
        rd_seed = np.random.randint(0, 1e8, 1)[0]
        np.random.seed(rd_seed)
        th_lst = [16,12]
        num = i
        limit_time = 3600 * 8
        th = th_lst[i-1]
        id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
        # id_lst = [1, 2, 3, 4]
        # id_lst = [100]
        # id_lst = [2]
        # id_lst = [1, 2]
        index_lst = range(0, len(id_lst))
        level = 0.1*i
        print time.time() - st_time
        filename = "experiment_results/repair_results" + str(num) + "/cpatch" + repr(int(level * 10))
        if not os.path.exists(filename):
            os.makedirs(filename)
        line_filename = "experiment_results/repair_results" + str(num) + "/lines" + repr(int(level * 10))
        if not os.path.exists(line_filename):
            os.makedirs(line_filename)
        filename = "experiment_results/table_results/"
        if not os.path.exists(filename):
            os.makedirs(filename)
        table_name = "experiment_results/table_results/experiment_results_total" + str(num) + ".xls"
        pwd = os.getcwd()
        if not os.path.exists(table_name):
            create_excel(table_name)
        manager = Manager()
        lock = manager.Lock()
        def para_f(i):
            return Test_id_fun(1, i, rd_seed, num, level, limit_time, np.power(2.0, th), table_name,lock)
        # print np.min([24, len(id_lst)])
        # para_f(1)
        p = Pool(np.min([20, len(id_lst)]))
        res_lst = []
        for result in p.imap(para_f, id_lst):
            res_lst.append(result)
        res_lst.sort()
        for rs in res_lst:
            pt_res_excel(rs, table_name)
        p.close()
        p.join()
        # glob_res = []
        print len(id_lst)
        print time.time() - st_time


# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# for i in id_lst:
#     read_bound(1,i,2)
# def store_bound():
# for k in range(1,3):
#     # id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
#     id_lst = [1]
#     def para_f(i):
#         generate_bound_3level(1,i,k)
#     p = Pool(np.min([24, len(id_lst)]))
#     p.map(para_f, id_lst)
#     p.close()
#     p.join()
# store_bound()
# [1, 2, 6, 16, 17, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
# [1, 2, 6, 16, 17, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35]
# print np.log2(3633225146256)
# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# id_lst_2v = [1, 2, 6, 10, 16, 17, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
# id_lst_3v = [1, 2, 6, 10, 16, 17, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
# id_lst_3v = final_results_3v.xls
# id_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
# print len(id_lst)
# gsl_sf_bessel_y2
# gsl_sf_bessel_j2