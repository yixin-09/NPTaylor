import os
import xlrd
import sys
import ast
import numpy as np
from multiprocessing import Pool,Lock,Manager
import itertools
import bench3v
import bench4v
import bench2v
import bench1v
import src.basic_func as bf
from xlutils.copy import copy
import xlwt
import pickle
def pickle_fun(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)
def get_exe_1vfiles():
    os.chdir('..')
    cwd = os.getcwd()
    print cwd
    path = cwd + "/benchmarks/driver_functions/1v_functions"
    os.chdir(path)
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".out"):
            file_list.append(file)
    return file_list

def get_exe_2vfiles():
    os.chdir('..')
    cwd = os.getcwd()
    print cwd
    path = cwd + "/benchmarks/driver_functions/2v_functions"
    os.chdir(path)
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".out"):
            file_list.append(file)
    return file_list

def get_exe_3vfiles():
    os.chdir('..')
    cwd = os.getcwd()
    print cwd
    path = cwd+"/benchmarks/driver_functions/3v_functions"
    os.chdir(path)
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".out"):
            file_list.append(file)
    return file_list
def get_exe_4vfiles():
    os.chdir('..')
    cwd = os.getcwd()
    print cwd
    path = cwd+"/benchmarks/driver_functions/4v_functions"
    os.chdir(path)
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".out"):
            file_list.append(file)
    return file_list
def generate_fpcore(fileName):
    cmd = "/home/yixin/software/HBG/herbgrind/herbgrind.sh --full-precision-exprs --output-sexp ./" + fileName
    os.system(cmd)
    return 0
    # os.system("cp ../input/2/*.txt ./")
#
def generate_fpcore_1T4():
    file_lst = get_exe_2vfiles()
    p = Pool(6)
    p.map(generate_fpcore, file_lst)
    p.close()
    p.join()





id_lst_2v = [1, 2, 6, 16, 17, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32,33, 34, 35]
id_lst_1v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 25, 26, 33, 57, 59, 62, 78, 79, 81, 86, 87, 98, 100, 102, 103, 104]
id_lst_3v = [1,4,5,6,7]
id_lst_4v = [2,3,4,5]
print len(id_lst_1v)
print len(id_lst_2v)
def get_fun_name(n_var,i):
    data = xlrd.open_workbook('../benchmarks/GSLbenchmarks.xls')
    table = data.sheets()[n_var-1]
    detecting_res = []
    fun_name = table.row_values(i)[1]
    fun_name = fun_name.strip()
    fun_pname = table.row_values(i)[4]
    fun_pname = fun_pname.strip()
    fun_pname = fun_pname.replace('sf.','gsl_sf_')
    return fun_name,fun_pname


def generate_HBG_1vdrive_programs(i):
    fname,fnp = get_fun_name(1,i)
    aline1 = "    stream = fopen(\""+fname+".txt\", \"r\");"
    ln1 = 15
    aline2 = "        y = "+fnp+";"
    ln2 = 22
    f = open("../benchmarks/driver_functions/1v_functions/test_gsl_sf_airy_Ai.c", "r")
    contents = f.readlines()
    f.close()
    orig_stdout = sys.stdout
    f = open("../benchmarks/driver_functions/1v_functions/test_"+fname+".c", 'w')
    sys.stdout = f
    for i in range(0,len(contents)):
        if i == ln1:
            print aline1
            continue
        if i == ln2:
            print aline2
            continue
        print contents[i].replace("\n",'')
    sys.stdout = orig_stdout
    f.close()

def generate_HBG_2vdrive_programs(i):
    fname,fnp = get_fun_name(2,i)
    aline1 = "    streamx = fopen(\""+fname+"x.txt\", \"r\");"
    aline2 = "    streamy = fopen(\""+fname+"y.txt\", \"r\");"
    ln1 = 19
    ln2 = 20
    aline3 = "        z = "+fnp+";"
    ln3 = 28
    f = open("../benchmarks/driver_functions/2v_functions/test_gsl_sf_bessel_Jnu.c", "r")
    contents = f.readlines()
    f.close()
    orig_stdout = sys.stdout
    f = open("../benchmarks/driver_functions/2v_functions/test_"+fname+".c", 'w')
    sys.stdout = f
    for i in range(0,len(contents)):
        if i == ln1:
            print aline1
            continue
        if i == ln2:
            print aline2
            continue
        if i == ln3:
            print aline3
            continue
        print contents[i].replace("\n",'')
    sys.stdout = orig_stdout
    f.close()

def generate_HBG_3vdrive_programs(i):
    fname,fnp = get_fun_name(3,i)
    aline1 = "    streamx = fopen(\""+fname+"x.txt\", \"r\");"
    aline2 = "    streamy = fopen(\""+fname+"y.txt\", \"r\");"
    aline3 = "    streamz = fopen(\""+fname+"z.txt\", \"r\");"
    ln1 = 23
    ln2 = 24
    ln3 = 25
    aline4 = "        res = "+fnp+";"
    ln4 = 34
    f = open("../benchmarks/driver_functions/3v_functions/test_gsl_sf_beta_inc.c", "r")
    contents = f.readlines()
    f.close()
    orig_stdout = sys.stdout
    f = open("../benchmarks/driver_functions/3v_functions/test_"+fname+".c", 'w')
    sys.stdout = f
    for i in range(0,len(contents)):
        if i == ln1:
            print aline1
            continue
        if i == ln2:
            print aline2
            continue
        if i == ln3:
            print aline3
            continue
        if i == ln4:
            print aline4
            continue
        print contents[i].replace("\n",'')
    sys.stdout = orig_stdout
    f.close()
def generate_HBG_4vdrive_programs(i):
    fname,fnp = get_fun_name(4,i)
    aline1 = "    streamx = fopen(\""+fname+"x.txt\", \"r\");"
    aline2 = "    streamy = fopen(\""+fname+"y.txt\", \"r\");"
    aline3 = "    streamz = fopen(\""+fname+"z.txt\", \"r\");"
    aline4 = "    streamp = fopen(\""+fname+"p.txt\", \"r\");"
    ln1 = 27
    ln2 = 28
    ln3 = 29
    ln4 = 30
    aline5 = "        res = "+fnp+";"
    ln5 = 40
    f = open("../benchmarks/driver_functions/4v_functions/test_gsl_sf_beta_inc.c", "r")
    contents = f.readlines()
    f.close()
    orig_stdout = sys.stdout
    f = open("../benchmarks/driver_functions/4v_functions/test_"+fname+".c", 'w')
    sys.stdout = f
    for i in range(0,len(contents)):
        if i == ln1:
            print aline1
            continue
        if i == ln2:
            print aline2
            continue
        if i == ln3:
            print aline3
            continue
        if i == ln4:
            print aline4
            continue
        if i == ln5:
            print aline5
            continue
        print contents[i].replace("\n",'')
    sys.stdout = orig_stdout
    f.close()

# for i in id_lst_2v:
#     generate_HBG_2vdrive_programs(i)
# #
# for i in id_lst_1v:
#     generate_HBG_1vdrive_programs(i)
# for i in id_lst_3v:
#     generate_HBG_3vdrive_programs(i)
# for i in id_lst_4v:
#     generate_HBG_4vdrive_programs(i)
def generate_Test_file(X,idname):
    inp_file = idname+".txt"
    # if not os.path.exists(idname):
    #     os.makedirs(idname)
    orig_stdout = sys.stdout
    f = open(inp_file, 'w')
    sys.stdout = f
    for x in X:
        print "%.18e" % x
    sys.stdout = orig_stdout
    f.close()
    return 0

def generate_1v_inputs_debug(i):
    exname='../benchmarks/driver_functions/1v_functions/experiment_results_total1.xls'
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    func_name = str(table.row_values(i)[1])
    func_name = func_name.strip()
    func_name = '../benchmarks/driver_functions/1v_functions/'+func_name
    bound = ast.literal_eval(table.row_values(i)[3])
    X = np.random.uniform(bound[0], bound[1], 10000)
    generate_Test_file(X, func_name)

def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l

def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound

def out_to_excel(exname,i,lst,lens):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    for k in range(0,len(lst)):
        sheet.write(i, lens+k, lst[k])
    new_excel.save(exname)

def create_excel(table_name):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "id")
    sheet.write(0, 1, "Programs")
    sheet.write(0, 2, "Mean_err")
    sheet.write(0, 3, "Max_err")
    book.save(table_name)
def generate_1v_inputs(i):
    exname='../benchmarks/driver_functions/1v_functions/experiment_results_total1.xls'
    exname2='../benchmarks/driver_functions/1v_functions/final_results_1v.xls'
    data = xlrd.open_workbook(exname)
    data2 = xlrd.open_workbook(exname2)
    table = data.sheets()[0]
    table2 = data2.sheets()[0]
    func_name = str(table.row_values(i)[1])
    func_name = func_name.strip()
    fun_id = int(table.row_values(i)[0])
    sp_inp = float(table2.row_values(fun_id)[3])
    # func_name2 = '../benchmarks/driver_functions/2v_functions/'+func_name
    func_namex = '../benchmarks/driver_functions/1v_functions/'+func_name
    func_name_pj = '../benchmarks/driver_functions/1v_functions/' + func_name + '_inps'
    print func_name
    # bound = [sp_inp-1e6*bf.getulp(sp_inp),sp_inp+1e6*bf.getulp(sp_inp)]
    bound = ast.literal_eval(table.row_values(i)[3])
    input_l = list(np.random.uniform(bound[0], bound[1], 10000))
    input_l.append(sp_inp)
    rf = bench1v.rfl[fun_id-1]
    gf = bench1v.gfl[fun_id-1]
    X = []
    Y = []
    err_lst = []
    for j in input_l:
        err_lst.append(np.log2(1.0 / bf.fitness_fun(rf, gf, j)))
    mean_err = np.mean(err_lst)
    max_err = np.max(err_lst)
    # out_exname = "../benchmarks/driver_functions/1v_err_compare.xls"
    # if not os.path.exists(out_exname):
    #     create_excel(out_exname)
    # out_to_excel(out_exname,i,[fun_id,func_name,mean_err,max_err],0)
    generate_Test_file(input_l, func_namex)
    pickle_fun(func_name_pj, input_l)
    return [i, [fun_id, func_name, mean_err, max_err]]

def generate_2v_inputs(i):
    exname='../benchmarks/driver_functions/2v_functions/experiment_results_total1.xls'
    exname2='../benchmarks/driver_functions/2v_functions/final_results_2v.xls'
    data = xlrd.open_workbook(exname)
    data2 = xlrd.open_workbook(exname2)
    table = data.sheets()[0]
    table2 = data2.sheets()[0]
    func_name = str(table.row_values(i)[1])
    func_name = func_name.strip()
    fun_id = int(table.row_values(i)[0])
    sp_inp = ast.literal_eval(table2.row_values(fun_id)[3])
    # func_name2 = '../benchmarks/driver_functions/2v_functions/'+func_name
    func_namex = '../benchmarks/driver_functions/2v_functions/'+func_name+'x'
    func_namey = '../benchmarks/driver_functions/2v_functions/'+func_name+'y'
    func_name_pj = '../benchmarks/driver_functions/2v_functions/' + func_name + '_inps'
    print func_name
    bound = ast.literal_eval(table.row_values(i)[3])
    print bound
    input_l = produce_n_input(bound,100)
    input_l.append(list(sp_inp))
    rf = bench2v.rfl[fun_id-1]
    gf = bench2v.gfl[fun_id-1]
    X = []
    Y = []
    err_lst = []
    for j in input_l:
        X.append(j[0])
        Y.append(j[1])
        # err_lst.append(bf.getUlpError(rf(*j),gf(*j)))
        err_lst.append(np.log2(1.0/bf.mfitness_fun(rf, gf, j)))
    mean_err = np.mean(err_lst)
    max_err = np.max(err_lst)
    generate_Test_file(X, func_namex)
    generate_Test_file(Y, func_namey)
    pickle_fun(func_name_pj, input_l)
    return [i,[fun_id,func_name,mean_err,max_err]]


def generate_3v_inputs(i):
    exname='../benchmarks/driver_functions/3v_functions/experiment_results_total1.xls'
    exname2='../benchmarks/driver_functions/3v_functions/final_results_3v.xls'
    data = xlrd.open_workbook(exname)
    data2 = xlrd.open_workbook(exname2)
    table = data.sheets()[0]
    table2 = data2.sheets()[0]
    func_name = str(table.row_values(i)[1])
    func_name = func_name.strip()
    fun_id = int(table.row_values(i)[0])
    sp_inp = ast.literal_eval(table2.row_values(fun_id)[3])
    # func_name2 = '../benchmarks/driver_functions/2v_functions/'+func_name
    func_namex = '../benchmarks/driver_functions/3v_functions/'+func_name+'x'
    func_namey = '../benchmarks/driver_functions/3v_functions/'+func_name+'y'
    func_namez = '../benchmarks/driver_functions/3v_functions/'+func_name+'z'
    func_name_pj = '../benchmarks/driver_functions/3v_functions/'+func_name+'_inps'
    print func_name
    bound = ast.literal_eval(table.row_values(i)[3])
    input_l = produce_n_input(bound,22)
    input_l.append(list(sp_inp))
    rf = bench3v.rfl[fun_id-1]
    gf = bench3v.gfl[fun_id-1]
    X = []
    Y = []
    Z = []
    err_lst = []
    for j in input_l:
        X.append(j[0])
        Y.append(j[1])
        Z.append(j[2])
        err_lst.append(np.log2(1.0 / bf.mfitness_fun(rf, gf, j)))
    mean_err = np.mean(err_lst)
    max_err = np.max(err_lst)
    # out_exname = "3v_err_compare.xls"
    # out_to_excel(out_exname,i,[fun_id,func_name,mean_err,max_err],0)
    generate_Test_file(X, func_namex)
    generate_Test_file(Y, func_namey)
    generate_Test_file(Z, func_namez)
    pickle_fun(func_name_pj,input_l)
    return [i, [fun_id, func_name, mean_err, max_err]]


def generate_4v_inputs(i):
    exname='../benchmarks/driver_functions/4v_functions/experiment_results_total1.xls'
    exname2='../benchmarks/driver_functions/4v_functions/final_results_4v.xls'
    data = xlrd.open_workbook(exname)
    data2 = xlrd.open_workbook(exname2)
    table = data.sheets()[0]
    table2 = data2.sheets()[0]
    func_name = str(table.row_values(i)[1])
    func_name = func_name.strip()
    fun_id = int(table.row_values(i)[0])
    sp_inp = ast.literal_eval(table2.row_values(fun_id)[3])
    # func_name2 = '../benchmarks/driver_functions/2v_functions/'+func_name
    func_namex = '../benchmarks/driver_functions/4v_functions/'+func_name+'x'
    func_namey = '../benchmarks/driver_functions/4v_functions/'+func_name+'y'
    func_namez = '../benchmarks/driver_functions/4v_functions/'+func_name+'z'
    func_namep = '../benchmarks/driver_functions/4v_functions/'+func_name+'p'
    func_name_pj = '../benchmarks/driver_functions/4v_functions/' + func_name + '_inps'
    print func_name
    bound = ast.literal_eval(table.row_values(i)[3])
    input_l = produce_n_input(bound,10)
    input_l.append(list(sp_inp))
    rf = bench4v.rfl[fun_id-1]
    gf = bench4v.gfl[fun_id-1]
    X = []
    Y = []
    Z = []
    P = []
    err_lst = []
    for j in input_l:
        X.append(j[0])
        Y.append(j[1])
        Z.append(j[2])
        P.append(j[3])
        err_lst.append(np.log2(1.0 / bf.mfitness_fun(rf, gf, j)))
    mean_err = np.mean(err_lst)
    max_err = np.max(err_lst)
    # out_exname = "3v_err_compare.xls"
    # out_to_excel(out_exname,i,[fun_id,func_name,mean_err,max_err],0)
    generate_Test_file(X, func_namex)
    generate_Test_file(Y, func_namey)
    generate_Test_file(Z, func_namez)
    generate_Test_file(P, func_namep)
    pickle_fun(func_name_pj, input_l)
    return [i, [fun_id, func_name, mean_err, max_err]]

def extract_1vfp_expression():
    path = "../benchmarks/driver_functions/1v_functions"
    exname = '../benchmarks/driver_functions/1v_functions/experiment_results_total1.xls'
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    file_lst = []
    for j in range(1,table.nrows):
        func_name = str(table.row_values(j)[1])
        func_name = func_name.strip()
        func_name2 = path + func_name
        file_name = path+"/test_"+func_name+".out.gh"
        f = open(file_name, 'r')
        contents = f.readlines()
        f.close()
        flag = 0
        exp_contents = []
        ide = 0
        k = 0
        err_flag = 0
        err_lst = []
        for i in contents:
            if err_flag ==0:
                if '(avg-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    mean_err = kstr[1].rstrip(')')
                    err_lst.append(mean_err)
                if '(max-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    max_err = kstr[1].rstrip(')')
                    err_lst.append(max_err)
                    err_flag = 1
            if '(var-problematic-ranges' in i:
                flag=0
            if flag==1:
                exp_contents.append(i.lstrip())
            if '(FPCore' in i:
                flag=1
                if exp_contents!=[]:
                    exp_contents[-1]=exp_contents[-1][0:-2]+ "\n"
                    exp_contents[ide+1]=":name \""+func_name+"_"+str(ide)+"\"\n"
                    ide = k
                exp_contents.append(i.lstrip())
                exp_contents.append(":name \""+func_name+"\"\n")
            if '(compare' in i:
                break
            k = k + 1
        exp_contents.append("\n")
        f = open("1v_exp.fpcore",'a')
        if len(exp_contents)>=2:
            for i in exp_contents[0:-2]:
                f.write(i)
            f.write(exp_contents[-2][0:-2] + "\n")
            f.write(exp_contents[-1])
        f.close()
        print err_lst
        if err_lst!=[]:
            out_exname = "../benchmarks/driver_functions/1v_err_compare.xls"
            out_to_excel(out_exname,j, err_lst, 4)

# extract_1vfp_expression()
# (avg-error 50.464464)
#   (max-error 52.620048)

def extract_2vfp_expression():
    path = "../benchmarks/driver_functions/2v_functions"
    exname = '../benchmarks/driver_functions/2v_functions/experiment_results_total1.xls'
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    file_lst = []
    for j in range(1,table.nrows):
        func_name = str(table.row_values(j)[1])
        func_name = func_name.strip()
        func_name2 = path + func_name
        file_name = path+"/test_"+func_name+".out.gh"
        f = open(file_name, 'r')
        contents = f.readlines()
        f.close()
        flag = 0
        exp_contents = []
        id = 0
        k = 0
        err_flag = 0
        err_lst = []
        for i in contents:
            if err_flag ==0:
                if '(avg-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    mean_err = kstr[1].rstrip(')')
                    err_lst.append(mean_err)
                if '(max-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    max_err = kstr[1].rstrip(')')
                    err_lst.append(max_err)
                    err_flag = 1
            if '(var-problematic-ranges' in i:
                flag=0
            if flag==1:
                exp_contents.append(i.lstrip())
            if '(FPCore' in i:
                flag=1
                if exp_contents!=[]:
                    exp_contents[-1]=exp_contents[-1][0:-2]+ "\n"
                    # exp_contents[id+1]=":name \""+func_name+"_"+str(id)+"\"\n"
                    # id = k
                exp_contents.append(i.lstrip())
                exp_contents.append(":name \""+func_name+"_"+str(id)+"\"\n")
                id = id + 1
            k = k + 1
        exp_contents.append("\n")
        f = open("2v_exp.fpcore",'a')
        if len(exp_contents)>=2:
            for i in exp_contents[0:-2]:
                f.write(i)
            f.write(exp_contents[-2][0:-2] + "\n")
            f.write(exp_contents[-1])
        f.close()
        print err_lst
        if err_lst!=[]:
            out_exname = "../benchmarks/driver_functions/2v_err_compare.xls"
            out_to_excel(out_exname,j, err_lst, 4)

def extract_3vfp_expression():
    path = "../benchmarks/driver_functions/3v_functions"
    exname = '../benchmarks/driver_functions/3v_functions/experiment_results_total1.xls'
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    file_lst = []
    for j in range(1,table.nrows):
        func_name = str(table.row_values(j)[1])
        func_name = func_name.strip()
        func_name2 = path + func_name
        file_name = path+"/test_"+func_name+".out.gh"
        f = open(file_name, 'r')
        contents = f.readlines()
        f.close()
        flag = 0
        exp_contents = []
        id = 0
        k = 0
        err_flag = 0
        err_lst = []
        for i in contents:
            if err_flag ==0:
                if '(avg-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    mean_err = kstr[1].rstrip(')')
                    err_lst.append(mean_err)
                if '(max-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    max_err = kstr[1].rstrip(')')
                    err_lst.append(max_err)
                    err_flag = 1
            if '(var-problematic-ranges' in i:
                flag=0
            if flag==1:
                exp_contents.append(i.lstrip())
            if '(FPCore' in i:
                flag=1
                if exp_contents!=[]:
                    exp_contents[-1]=exp_contents[-1][0:-2]+ "\n"
                    # exp_contents[id+1]=":name \""+func_name+"_"+str(id)+"\"\n"
                    # id = id+1
                exp_contents.append(i.lstrip())
                exp_contents.append(":name \""+func_name+"_"+str(id)+"\"\n")
                id = id+1
            k = k + 1
        exp_contents.append("\n")
        f = open("3v_exp.fpcore",'a')
        if len(exp_contents)>=2:
            for i in exp_contents[0:-2]:
                f.write(i)
            f.write(exp_contents[-2][0:-2] + "\n")
            f.write(exp_contents[-1])
        f.close()
        print err_lst
        if err_lst!=[]:
            out_exname = "../benchmarks/driver_functions/3v_err_compare.xls"
            out_to_excel(out_exname,j, err_lst, 4)


def extract_4vfp_expression():
    path = "../benchmarks/driver_functions/4v_functions"
    exname = '../benchmarks/driver_functions/4v_functions/experiment_results_total1.xls'
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    file_lst = []
    for j in range(1,table.nrows):
        func_name = str(table.row_values(j)[1])
        func_name = func_name.strip()
        func_name2 = path + func_name
        file_name = path+"/test_"+func_name+".out.gh"
        f = open(file_name, 'r')
        contents = f.readlines()
        f.close()
        flag = 0
        exp_contents = []
        id = 0
        k = 0
        err_flag = 0
        err_lst = []
        for i in contents:
            if err_flag ==0:
                if '(avg-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    mean_err = kstr[1].rstrip(')')
                    err_lst.append(mean_err)
                if '(max-error' in i:
                    kstr = i.strip()
                    kstr = kstr.split()
                    max_err = kstr[1].rstrip(')')
                    err_lst.append(max_err)
                    err_flag = 1
            if '(var-problematic-ranges' in i:
                flag=0
            if flag==1:
                exp_contents.append(i.lstrip())
            if '(FPCore' in i:
                flag=1
                if exp_contents!=[]:
                    exp_contents[-1]=exp_contents[-1][0:-2]+ "\n"
                    # exp_contents[id+1]=":name \""+func_name+"_"+str(id)+"\"\n"
                    # id = id+1
                exp_contents.append(i.lstrip())
                exp_contents.append(":name \""+func_name+"_"+str(id)+"\"\n")
                id = id+1
            k = k + 1
        exp_contents.append("\n")
        f = open("4v_exp.fpcore",'a')
        if len(exp_contents)>=2:
            for i in exp_contents[0:-2]:
                f.write(i)
            f.write(exp_contents[-2][0:-2] + "\n")
            f.write(exp_contents[-1])
        f.close()
        print err_lst
        if err_lst!=[]:
            out_exname = "../benchmarks/driver_functions/4v_err_compare.xls"
            out_to_excel(out_exname,j, err_lst, 4)

# for i in range(1,3):
#     generate_1v_inputs(i)
# for i in range(1,16):
#     generate_2v_inputs(i)




def test_HBG_funcs():
    out_exname = "../benchmarks/driver_functions/1v_err_compare.xls"
    if not os.path.exists(out_exname):
        create_excel(out_exname)

    # for i in range(1,31):
    #     idx, res = generate_1v_inputs(i)
    #     out_to_excel(out_exname, idx, res, 0)
    def para_f(i):
        return generate_1v_inputs(i)

    p = Pool(10)
    # for result in p.imap(para_f, id_lst):
    #     result_lst.append(result)
    id_lst = range(1, 3)
    for result in p.imap(para_f, id_lst):
        out_to_excel(out_exname, result[0], result[1], 0)
    p.close()
    p.join()

    out_exname = "../benchmarks/driver_functions/3v_err_compare.xls"
    if not os.path.exists(out_exname):
        create_excel(out_exname)

    # for i in range(1,31):
    #     idx, res = generate_3v_inputs(i)
    #     out_to_excel(out_exname, idx, res, 0)
    def para_f(i):
        return generate_3v_inputs(i)

    p = Pool(10)
    # for result in p.imap(para_f, id_lst):
    #     result_lst.append(result)
    id_lst = range(1, 6)
    for result in p.imap(para_f, id_lst):
        out_to_excel(out_exname, result[0], result[1], 0)
    p.close()
    p.join()

    out_exname = "../benchmarks/driver_functions/4v_err_compare.xls"
    if not os.path.exists(out_exname):
        create_excel(out_exname)

    # for i in range(1,31):
    #     idx, res = generate_3v_inputs(i)
    #     out_to_excel(out_exname, idx, res, 0)
    def para_f(i):
        return generate_4v_inputs(i)

    p = Pool(10)
    # for result in p.imap(para_f, id_lst):
    #     result_lst.append(result)
    id_lst = range(1, 5)
    for result in p.imap(para_f, id_lst):
        out_to_excel(out_exname, result[0], result[1], 0)
    p.close()
    p.join()

    out_exname = "../benchmarks/driver_functions/2v_err_compare.xls"
    if not os.path.exists(out_exname):
        create_excel(out_exname)

    # for i in range(1,31):
    #     idx, res = generate_3v_inputs(i)
    #     out_to_excel(out_exname, idx, res, 0)
    def para_f(i):
        return generate_2v_inputs(i)

    p = Pool(10)
    # for result in p.imap(para_f, id_lst):
    #     result_lst.append(result)
    id_lst = range(1, 19)
    for result in p.imap(para_f, id_lst):
        out_to_excel(out_exname, result[0], result[1], 0)
    p.close()
    p.join()
    file_lst = get_exe_1vfiles()
    p = Pool(10)
    p.map(generate_fpcore, file_lst)
    p.close()
    p.join()

    file_lst = get_exe_2vfiles()
    p = Pool(10)
    p.map(generate_fpcore, file_lst)
    p.close()
    p.join()

    file_lst = get_exe_3vfiles()
    p = Pool(6)
    p.map(generate_fpcore, file_lst)
    p.close()
    p.join()

    file_lst = get_exe_4vfiles()
    p = Pool(6)
    p.map(generate_fpcore, file_lst)
    p.close()
    p.join()
    # extract_3vfp_expression()
    # extract_1vfp_expression()
    # extract_2vfp_expression()
    # extract_3vfp_expression()

# extract_4vfp_expression()
# file_lst = get_exe_1vfiles()
# p = Pool(10)
# p.map(generate_fpcore, file_lst)
# p.close()
# p.join()

# file_lst = get_exe_1vfiles()
# p = Pool(10)
# p.map(generate_fpcore, file_lst)
# p.close()
# p.join()


#
# file_lst = get_exe_2vfiles()
# p = Pool(10)
# p.map(generate_fpcore, file_lst)
# p.close()
# p.join()

# extract_2vfp_expression()
# extract_1vfp_expression()
#