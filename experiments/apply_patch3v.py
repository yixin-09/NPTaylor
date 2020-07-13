import os
import time
import sys,getopt
import xlrd
import ast
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.basic_func as bf
from src.detector import DEMC_pure
from src.detector import DDEMC_pure
from fun_index import search_line_num4f
from fun_index import get_varible_name
import importlib
import bench1v
import bench2v
import bench3v
import bench4v
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import Pool,Lock,Manager
import itertools
from mpmath import *
mp.dps = 50
def sudo_cmd(cmd):
    sudoPassword = 'hello'
    os.system('echo %s|sudo -S %s' % (sudoPassword, cmd))


def insert_patches(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    for i in range(1,table.nrows):
        print i
        if i!=1:
            name = table.row_values(i)[1]
            name = name.strip()
            fun_name = name + "_e"
            ori_bound = ast.literal_eval(table.row_values(i)[3])
            x_l = get_varible_name(fun_name, "fun_index.xls")
            x = x_l[0]
            y = x_l[1]
            z = x_l[2]
            insert_fname, insert_line = search_line_num4f(fun_name, "fun_index.xls")
            gsl_file = "../benchmarks/gsl_src/gsl-2.1-repair/specfunc/" + insert_fname
            orig_stdout = sys.stdout
            f = open("temp_file", 'w')
            sys.stdout = f
            print "#include \"patch_of_" + name + ".c\""
            print "if((" + x + "<=" + repr(ori_bound[0][1]) + ")&&(" + x + ">=" + repr(ori_bound[0][0]) + ")){"
            print " if((" + y + "<=" + repr(ori_bound[1][1]) + ")&&(" + y + ">=" + repr(ori_bound[1][0]) + ")){"
            print "     if((" + z + "<=" + repr(ori_bound[2][1]) + ")&&(" + z + ">=" + repr(ori_bound[2][0]) + ")){"
            print "         result->val = accuracy_improve_patch_of_" + name + "(" + x + "," + y + "," + z + ");"
            print "         result->err = GSL_DBL_EPSILON * fabs(result->val);"
            print "         return GSL_SUCCESS;"
            print "     }"
            print " }"
            print "}"
            sys.stdout = orig_stdout
            f.close()
            f = open("temp_file", "r")
            contents = f.readlines()
            f.close()
            f = open(gsl_file, "r")
            new_contents = f.readlines()
            f.close()
            for j in range(1, len(contents)):
                new_contents.insert(insert_line, " " + contents[j])
                insert_line = insert_line + 1
            new_contents.insert(25, contents[0])
            f = open(gsl_file, "w")
            new_contents = "".join(new_contents)
            f.write(new_contents)
            f.close()
            sudo_cmd("rm temp_file")

# insert_patches("experiment_results/table_results/experiment_results_total1.xls"," ",1)
def insert_patch_and_recompile(exname,filename):
    os.system("./unpatch_cmd.sh")
    insert_patches(exname)
    cwd = os.getcwd()
    os.chdir(filename)
    os.system("cp *.c "+"../../../../benchmarks/gsl_src/gsl-2.1-repair/specfunc")
    os.chdir(cwd)
    os.system("cp ../benchmarks/eft_files/* "+"../benchmarks/gsl_src/gsl-2.1-repair/specfunc")
    sudo_cmd("./make_install.sh > tmp_log")
    sudo_cmd("rm tmp_log")
# insert_patch_and_recompile("experiment_results/table_results/experiment_results_total1.xls","experiment_results/repair_results1/test1")
# test all functions in para

# generte n**len(i) inputs in the given bound i
def produce_n_input(i,n):
    var_l = []
    n = int(n)
    for k in i:
        var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l

def test_2v_bound(bound,pf,rf,th):
    # glob_fitness_fun = lambda x: bf.mfitness_fun(rf, pf, x)
    # ret = differential_evolution(glob_fitness_fun, popsize=25, bounds=bound)
    print bound
    res = DDEMC_pure(rf,pf,[bound],1,20000)
    max_err = res[0]
    # max_err = 1.0/glob_fitness_fun(ret.x[0])
    max_x = res[1]
    # max_x = ret.x[0]
    st = time.time()
    step = 100
    input_l = produce_n_input(bound, step)
    for i in input_l:
        pf(*i)
    end_time = time.time() - st
    step = 10
    input_l = produce_n_input(bound, step)
    pf_res = [pf(*i) for i in input_l]
    rf_res = [rf(*i) for i in input_l]
    err_lst = [bf.getUlpError(i,j) for i,j in zip(pf_res,rf_res)]
    fid = 0
    sid = 0
    for i in err_lst:
        if i > th:
            fid = fid + 1
        else:
            sid = sid + 1
    mean_err = np.mean(err_lst)
    temp_max = np.max(err_lst)
    max_err = np.max([temp_max,max_err])
    srate = sid/float(step*step*step)
    return end_time,max_err,mean_err,max_x,srate

def read_idx_bound_fromxls(exname,i):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    # ori_bound = ast.literal_eval(table.row_values(i)[3])
    idx = ast.literal_eval(table.row_values(i)[0])
    # test in whole domain
    ori_bound = bench3v.input_domain[idx-1][0]
    return idx,ori_bound,table.nrows

def test_in_bound_domain(n_var,i,idx,bound,th):
    #read bound from exname

    ld_module = importlib.import_module("bench" + str(n_var) + "v")
    pf = ld_module.gfl[idx - 1]
    rf = ld_module.rfl[idx - 1]
    #detect the time and max,mean error over 1000000 random points
    et, max_err, mean_err, max_x, srate = test_2v_bound(bound, pf, rf, th)
    return [et, max_err, mean_err, max_x, i, srate]
    # write_back_to_exacel
    # lock.acquire()
    # bf.test1vbound2excel(et,max_err,mean_err,max_x,exname,i,16)
    # lock.release()

# if __name__ == "__main__":
    # insert_patch_and_recompile("experiment_results/table_results/experiment_results_total1.xls",
    #                            "experiment_results/repair_results1/test1")
    # insert_patch_and_recompile("experiment_results/table_results/experiment_results_total2.xls",
    #                            "experiment_results/repair_results2/test2")
    # insert_patch_and_recompile("experiment_results/table_results/experiment_results_total3.xls",
    #                            "experiment_results/repair_results3/test3")
    # for i in range(1,2):
    #     table_name = 'experiment_results/table_results/experiment_results_total'+str(i)+'.xls'
    #     idx,ori_bound,nrows = read_idx_bound_fromxls(table_name,1)
    #     # manager = Manager()
    #     # lock = manager.Lock()
    #     # for i in range(1,nrows):
    #     #     idx, ori_bound, nr = read_idx_bound_fromxls(exname, i)
    #     #     test_in_bound_domain(exname, 1, i, idx, ori_bound,lock)
    #     def test_para_f(d):
    #         idx, ori_bound, nr = read_idx_bound_fromxls(table_name, d)
    #         return test_in_bound_domain(2, d, idx, ori_bound)
    #     # print test_para_f(1)
    #     id_lst = range(1, nrows)
    #     print id_lst
    #     # test_para_f(1)
    #     p = Pool(np.min([12, len(id_lst)]))
    #     # p.map(test_para_f, id_lst)
    #     manager = Manager()
    #     lock = manager.Lock()
    #     for result in p.imap(test_para_f, id_lst):
    #         bf.test1vbound2excel(result[0], result[1], result[2], result[3], table_name, result[4], 11)
    #     p.close()
    #     p.join()


#
# def main():
#     begin_time = time.time()
#     opts, args = getopt.getopt(sys.argv[1:], "n:t:i:", ["times=","threshold=","fun_id="])
#     num = 0
#     th = 0
#     status = 0
#     fun_id = 0
#     ln = [0,3,2,1]
#     for o, a in opts:
#         if o in ('-n'):
#             num = int(a)
#             print num
#         elif o in ('-t'):
#             th = int(a)
#         elif o in ('-i'):
#             fun_id = int(a)
#         else:
#             print 'unhandled option'
#             sys.exit(3)
#     table_name = 'experiment_results/table_results/experiment_results_total'+str(num)+'.xls'
#     folder_name = os.getcwd()
#     file_name = folder_name+'/experiment_results/repair_results'+str(num)+ '/test'+str(th)+'/'
#     read_bound_randomSeed(table_name,
#                           file_name, ln[th],fun_id)
#
#
#
#
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "n:t:", ["num=","threshold="])
    num = 0
    th = 0
    status = 0
    fun_id = 0
    ln = [0,3,2,1]
    for o, a in opts:
        if o in ('-n'):
            num = int(a)
            print num
        elif o in ('-t'):
            th = int(a)
        else:
            print 'unhandled option'
            sys.exit(3)
    th_lst = [16,12]
    th = np.power(2.0, th_lst[th - 1])
    table_name = 'experiment_results3v/table_results/experiment_results_total'+str(num)+'.xls'
    folder_name = os.getcwd()
    file_name = folder_name+'/experiment_results3v/repair_results'+str(num)+ '/test'+str(num)+'/'
    idx, ori_bound, nrows = read_idx_bound_fromxls(table_name, 1)
    # os.system("./unpatch_cmd.sh")
    # sudo_cmd("./make_install.sh > tmp_log")
    # manager = Manager()
    # lock = manager.Lock()
    # for i in range(1,nrows):
    #     idx, ori_bound, nr = read_idx_bound_fromxls(exname, i)
    #     test_in_bound_domain(exname, 1, i, idx, ori_bound,lock)
    def test_para_f(i):
        idx, ori_bound, nr = read_idx_bound_fromxls(table_name, i)
        return test_in_bound_domain(3, i, idx, ori_bound,th)
    id_lst = range(1, nrows)
    print id_lst
    # test_para_f(5)
    p = Pool(np.min([12, len(id_lst)]))
    # p.map(test_para_f, id_lst)
    manager = Manager()
    lock = manager.Lock()
    for result in p.imap(test_para_f, id_lst):
        bf.test2vbound2excel(result[0], result[1], result[2], result[3], table_name, result[4],result[5], 10)
    p.close()
    p.join()
    insert_patch_and_recompile(table_name,file_name)

