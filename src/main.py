from detector1v import detectHighErrs
from detector1v import searchMaxErr
from extractor import iter_liner_build
from extractor import iter_liner_build_LineTay
from extractor import iter_build_tay1v
from extractor import iter_build_tay2v
from extractor import ini_tay2v_build
from extractor import ini_tay1v_build
from extractor import iter_build_tay3v
from extractor import iter_build_tay4v
# from extractor import ini_tay4v_build
from extractor import plot_err_in_bound
import extractor as exr
from linerFinder import generate_fitting
import basic_func as bf
from GenPatch import covertToC
from GenPatch import combCovertToC
from GenPatch import convertToC_taylor
from GenPatch import combCovertToC_taylor
from GenPatch import convertToC_taylorOnL
from GenPatch import convertToC_taylor1v
from GenPatch import convertToC_taylor2v
from GenPatch import combCovertToC_tay1v
from GenPatch import combCovertToC_tay2v
from GenPatch import combCovertToC_tay3v
from GenPatch import convertToC_taylor3v
from GenPatch import patch_generate
import os
import signal
import time
from mpmath import diff
import math


class TimeoutError (RuntimeError):
    pass

def handler (signum, frame):
    raise TimeoutError()

signal.signal (signal.SIGALRM, handler)


def taylor_exp(rf,x,der_l,x0,n):
    temp = float(der_l[0])
    for i in range(1,n):
        temp = temp + math.pow((x-x0),i)*der_l[i]/math.factorial(i)
    return temp



# def derivingApproximation_taylor(th,bound,rf,name,filename,inp):
#     temp_n = 0
#     for i in range(2,20):
#         der_l = []
#         temp_n = i
#         for i in range(temp_n):
#             der_l.append(diff(rf,inp,i))
#         patch_pf = lambda x: taylor_exp(rf,x,der_l,inp,temp_n)
#         res = bf.find_max(patch_pf,rf,bound)
#         if res[0]<th:
#             print res[0]
#             print th
#             convertToC_taylor(temp_n,der_l,name,filename,inp)
#             return temp_n
def derivingApproximation_taylor(th,bound,rf,pf,name,filename,inp,lock):
    n = 7
    # iter_build_tay1v(th, bound,bound, rf,pf, n, inp,[],[])
    ini_tay1v_build(th, bound,bound, rf,pf, n, inp)
    print ">>>>>>>>>>"
    save_line = []
    print exr.glob_point_l_tay
    name = name.strip()
    lock.acquire()
    convertToC_taylor1v(exr.glob_point_l_tay, name,0, filename)
    # plot_err_in_bound(rf, pf, bound, exr.glob_point_l_tay)
    len_tay = len(exr.glob_point_l_tay)
    save_line = save_line+exr.glob_point_l_tay
    exr.glob_point_l_tay = []
    lock.release()
    return save_line,len_tay


def derivingApproximation_taylor2v(th,ini_bound,rf,pf,name,filename,point,lock,file_name):
    n = 7
    dr = ini_tay2v_build(rf,pf,point,ini_bound,th,n,file_name)
    print ">>>>>>>>>>"
    print exr.glob_tay2v_lst
    name = name.strip()
    lock.acquire()
    convertToC_taylor2v(exr.glob_tay2v_lst, name, 0, filename,dr)
    # plot_err_in_bound(rf, pf, bound, exr.glob_point_l_tay)
    len_tay = len(exr.glob_tay2v_lst)
    save_lines = exr.glob_tay2v_lst
    exr.glob_tay2v_lst = []
    lock.release()
    return save_lines,len_tay


def derivingApproximation_taylor3v(th,ini_bound,rf,pf,name,filename,point,lock):
    n = 6
    iter_build_tay3v(th,ini_bound,rf,pf,n,[],[],point)
    print ">>>>>>>>>>"
    print exr.glob_tay3v_lst
    name = name.strip()
    lock.acquire()
    convertToC_taylor3v(exr.glob_tay3v_lst, name, 0, filename)
    # plot_err_in_bound(rf, pf, bound, exr.glob_point_l_tay)
    len_tay = len(exr.glob_tay3v_lst)
    save_lines = exr.glob_tay3v_lst
    exr.glob_tay3v_lst = []
    lock.release()
    return save_lines, len_tay

def derivingApproximation_taylor4v(th,ini_bound,rf,pf,name,filename,point,lock):
    n = 5
    iter_build_tay4v(th,ini_bound,rf,pf,n,point)
    print ">>>>>>>>>>"
    print exr.glob_tay4v_lst
    name = name.strip()
    lock.acquire()
    # convertToC_taylor3v(exr.glob_tay3v_lst, name, 0, filename)
    # plot_err_in_bound(rf, pf, bound, exr.glob_point_l_tay)
    len_tay = len(exr.glob_tay4v_lst)
    save_lines = exr.glob_tay4v_lst
    exr.glob_tay4v_lst = []
    lock.release()
    return save_lines, len_tay

def derivingApproximation_lineATaylor(th,bound_l,rf,name,filename,inp):
    bound_idx = 0
    num_line = 0
    save_line = []
    exr.glob_point_l_tay = []
    for bound in bound_l:
        temp_ploy_fit = ''
        print bound
        n = int(bf.getFPNum(bound[0], bound[1]))
        ori_bound = bound
        print bound
        print
        print "%.12e" % float(bf.getFPNum(bound[0], bound[1]))
        # iter_liner_build_LineTay(th, bound, rf, n)
        # if (inp < bound[1]) & (inp > bound[0]):
        #     iter_liner_build_LineTay(th, [bound[0], inp], rf, n)
        #     iter_liner_build_LineTay(th, [inp, bound[1]], rf, n)
        # else:
        iter_liner_build_LineTay(th, bound, rf, n)
        if len(exr.glob_point_l_tay) >= 30:
            temp_ploy_fit = generate_fitting(exr.glob_point_l_tay)
        convertToC_taylorOnL(exr.glob_point_l_tay, n, name, bound_idx, filename, ori_bound,temp_ploy_fit)
        print exr.glob_point_l_tay
        print len(exr.glob_point_l_tay)
        save_line = save_line + exr.glob_point_l_tay
        num_line = num_line + len(exr.glob_point_l_tay)
        bound_idx = bound_idx + 1
        exr.glob_point_l_tay = []
    return save_line,num_line


def derivingApproximation(th,bound_l,rf,name,filename,inp):
    bound_idx = 0
    num_line = 0
    save_line = []
    exr.glob_point_l = []
    for bound in bound_l:
        temp_ploy_fit = ''
        n = int(bf.getFPNum(bound[0], bound[1]))
        ori_bound = bound
        print bound
        print
        print "%.12e" % float(bf.getFPNum(bound[0], bound[1]))
        # To make sure the length of bound less than a value (1e12)
        limit_of_bound = 1e12
        samll_bound_l = []
        ulp_x = bf.getulp(bound[0])
        # Step2: linear iterative approximation
        # partition the bound according to the limit_of_bound
        if n / limit_of_bound > 2.0:
            temp_bound0 = bound[0]
            temp_dis = limit_of_bound * ulp_x
            while (temp_bound0 + temp_dis < bound[1]):
                if (inp<temp_bound0 + temp_dis)&(inp>temp_bound0):
                    samll_bound_l.append([temp_bound0, inp])
                    samll_bound_l.append([inp, temp_bound0 + temp_dis])
                else:
                    samll_bound_l.append([temp_bound0, temp_bound0 + temp_dis])
                temp_bound0 = temp_bound0 + temp_dis
            samll_bound_l.append([temp_bound0, bound[1]])
            print len(samll_bound_l)
            print bound
            i = 0
            for idx_b in samll_bound_l:
                i = i + 1
                iter_liner_build(th, idx_b, rf, n)
        else:
            if (inp<bound[1])&(inp>bound[0]):
                iter_liner_build(th, [bound[0],inp], rf, n)
                iter_liner_build(th, [inp,bound[1]], rf, n)
            else:
                iter_liner_build(th, bound, rf, n)
        if len(exr.glob_point_l) >= 30:
            temp_ploy_fit = generate_fitting(exr.glob_point_l)
        covertToC(exr.glob_point_l, n, name, bound_idx, filename,ori_bound,temp_ploy_fit)
        save_line=save_line+exr.glob_point_l
        num_line = num_line+len(exr.glob_point_l)
        bound_idx = bound_idx + 1
        exr.glob_point_l = []
    return save_line,num_line



def main1v(rf,pf,level,th, rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock):
    fnm = fnm.strip()
    print "Begin repair the function "+fnm
    # generate file to store patch files
    filename = "../experiments/experiment_results/repair_results" + str(num) + "/cpatch" + repr(int(level * 10))
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # generate file to save lines
    line_filename = "../experiments/experiment_results/repair_results" + str(num) + "/lines" + repr(int(level * 10))
    # if not os.path.exists(line_filename):
    #     os.makedirs(line_filename)
    res = []
    res.append(idx)
    res.append(fnm)
    # if max_ret == []:
    #     max_ret = searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    # else:
    #     searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    #     num_line = 0
    #     print "max"
    #     print max_ret
    staTime = time.time()
    try:
        signal.alarm(limit_time)
        # call the detector to find I_err
        print "Executing the PTB algorithm"
        # max_x, bound, bound_l = detectHighErrs(max_ret,th, rf, pf)
        bound_l = bf.bound_partition(bound)
        t1 = time.time() - staTime
        res.append(th)
        res.append(t1)
        res.append(bound)
        bound_distance = bf.getUlpError(bound[0], bound[1])
        res.append(bound_distance)
        ori_bound = bound
        print bound
        print "The size of bound is: %.8e" % bound_distance
        print res
        staTime2 = time.time()
        max_x = max_ret[1]
        # call the derivingApproximation to produce approximation and produce patch
        # num_line = derivingApproximation_taylor(th,ori_bound,rf,pf,fnm,filename,max_x,lock)
        save_lines, num_line = derivingApproximation(th, bound_l, rf, fnm, filename, max_x)
        bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        combCovertToC(bound_l, fnm, filename)
        # combCovertToC_tay1v(bound, fnm, filename)
        temp_t = time.time() - staTime2
        res.append(temp_t)
        # patch_generate(ori_bound, fnm, filename)
        total_time = time.time() - staTime
        res.append(total_time)
        res.append(rd_seed)
        print "Repair time: " + str(temp_t)
        print "patch is generate, the name is " + 'patch_of_' + fnm + ".c"
        size_file = 0.0
        if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
            size_file = os.path.getsize(filename + '/patch_of_' + fnm + ".c")
        res.append(size_file)
        res.append(num_line)
        bf.glob_point_l = []
        return res
    except TimeoutError:
        print 'timeout'
        res.append(0.0)
        res.append(0.0)
        res.append(rd_seed)
        res.append(0)
        res.append(0.0)
        res.append(0.0)
        return res
def main1v_tay(rf,pf,level,th, rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock):
    fnm = fnm.strip()
    print "Begin repair the function "+fnm
    # generate file to store patch files
    filename = "../experiments/experiment_results/repair_results" + str(num) + "/cpatch" + repr(int(level * 10))
    if not os.path.exists(filename):
        os.makedirs(filename)
    # generate file to save lines
    line_filename = "../experiments/experiment_results/repair_results" + str(num) + "/lines" + repr(int(level * 10))
    if not os.path.exists(line_filename):
        os.makedirs(line_filename)
    res = []
    res.append(idx)
    res.append(fnm)
    # if max_ret == []:
    #     max_ret = searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    # else:
    #     searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    #     num_line = 0
    #     print "max"
    #     print max_ret
    staTime = time.time()
    try:
        signal.alarm(limit_time)
        # call the detector to find I_err
        print "Executing the PTB algorithm"
        # max_x, bound, bound_l = detectHighErrs(max_ret,th, rf, pf)
        bound_l = bf.bound_partition(bound)
        t1 = time.time() - staTime
        res.append(th)
        res.append(t1)
        res.append(bound)
        bound_distance = bf.getUlpError(bound[0], bound[1])
        res.append(bound_distance)
        ori_bound = bound
        print bound
        print "The size of bound is: %.8e" % bound_distance
        print res
        staTime2 = time.time()
        max_x = max_ret[1]
        # call the derivingApproximation to produce approximation and produce patch
        save_lines,num_line = derivingApproximation_taylor(th,ori_bound,rf,pf,fnm,filename,max_x,lock)
        # save_lines, num_line = derivingApproximation(th, bound_l, rf, fnm, filename, max_x)
        bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # combCovertToC(bound_l, fnm, filename)
        combCovertToC_tay1v(bound, fnm, filename)
        temp_t = time.time() - staTime2
        res.append(temp_t)
        # patch_generate(ori_bound, fnm, filename)
        total_time = time.time() - staTime
        res.append(total_time)
        res.append(rd_seed)
        print "Repair time: " + str(temp_t)
        print "patch is generate, the name is " + 'patch_of_' + fnm + ".c"
        size_file = 0.0
        if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
            size_file = os.path.getsize(filename + '/patch_of_' + fnm + ".c")
        res.append(size_file)
        res.append(num_line)
        bf.glob_point_l = []
        return res
    except TimeoutError:
        print 'timeout'
        res.append(0.0)
        res.append(0.0)
        res.append(rd_seed)
        res.append(0)
        res.append(0.0)
        res.append(0.0)
        return res
def main2v(rf,pf,level,th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock,file_name):
    fnm = fnm.strip()
    print "Begin repair the function "+fnm
    # generate file to store patch files
    filename = "../experiments/experiment_results/repair_results" + str(num) + "/test" + repr(int(level * 10))
    if not os.path.exists(filename):
        os.makedirs(filename)
    # generate file to save lines
    line_filename = "../experiments/experiment_results/repair_results" + str(num) + "/lines" + repr(int(level * 10))
    if not os.path.exists(line_filename):
        os.makedirs(line_filename)
    # if not os.path.exists(filename + "/patch"):
    #     os.makedirs(filename + "/patch")
    # generate shell scripts to apply patches
    # if os.path.exists(filename + "/patch/patch_cmd.sh"):
    #     os.remove(filename + "/patch/patch_cmd.sh")
    # f = open(filename + "/patch/patch_cmd.sh", "a")
    # f.write("#!/usr/bin/env bash\n")
    # f.close()
    # sudoPassword = password
    # command = 'chmod 777 ' + filename + "/patch/patch_cmd.sh"
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    # # make sure remove the old patch
    # if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
    #     os.remove(filename + '/patch_of_' + fnm + ".c")
    # A list to store the results we want to save, e.g. the repair time, the size of bound
    res = []
    res.append(idx)
    res.append(fnm)
    # if max_ret == []:
    #     max_ret = searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    # else:
    #     searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    #     num_line = 0
    #     print "max"
    #     print max_ret
    staTime = time.time()
    num_line = 0
    try:
        signal.alarm(limit_time)
        # call the detector to find I_err
        print "Executing the PTB algorithm"
        # max_x, bound, bound_l = detectHighErrs(max_ret,th, rf, pf)
        t1 = time.time() - staTime
        res.append(th)
        res.append(t1)
        res.append(bound)
        bound_distance = [bf.getUlpError(bound_i[0], bound_i[1]) for bound_i in bound]
        res.append(bound_distance)
        ori_bound = bound
        print bound
        print res
        staTime2 = time.time()
        max_x = max_ret[1]
        # call the derivingApproximation to produce approximation and produce patch
        # save_lines,num_line = derivingApproximation_taylor2v(th,ori_bound,rf,pf,fnm,filename,max_x,lock,file_name)
        # save_lines, num_line = derivingApproximation(th, bound_l, rf, fnm, filename, max_x)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # combCovertToC(bound_l, fnm, filename)
        combCovertToC_tay2v(bound, fnm, filename)
        temp_t = time.time() - staTime2
        res.append(temp_t)
        # patch_generate(ori_bound, fnm, filename)
        total_time = time.time() - staTime
        res.append(total_time)
        res.append(rd_seed)
        print "Repair time: " + str(temp_t)
        print "patch is generate, the name is " + 'patch_of_' + fnm + ".c"
        size_file = 0.0
        if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
            size_file = os.path.getsize(filename + '/patch_of_' + fnm + ".c")
        res.append(size_file)
        res.append(num_line)
        bf.glob_point_l = []
        return res
    except TimeoutError:
        print 'timeout'
        res.append(0.0)
        res.append(0.0)
        res.append(rd_seed)
        res.append(0)
        res.append(0.0)
        res.append(0.0)
        return res

def main3v(rf,pf,level,th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock):
    fnm = fnm.strip()
    print "Begin repair the function "+fnm
    # generate file to store patch files
    filename = "../experiments/experiment_results3v/repair_results" + str(num) + "/test" + repr(int(level * 10))
    if not os.path.exists(filename):
        os.makedirs(filename)
    # generate file to save lines
    line_filename = "../experiments/experiment_results3v/repair_results" + str(num) + "/lines" + repr(int(level * 10))
    if not os.path.exists(line_filename):
        os.makedirs(line_filename)
    # if not os.path.exists(filename + "/patch"):
    #     os.makedirs(filename + "/patch")
    # generate shell scripts to apply patches
    # if os.path.exists(filename + "/patch/patch_cmd.sh"):
    #     os.remove(filename + "/patch/patch_cmd.sh")
    # f = open(filename + "/patch/patch_cmd.sh", "a")
    # f.write("#!/usr/bin/env bash\n")
    # f.close()
    # sudoPassword = password
    # command = 'chmod 777 ' + filename + "/patch/patch_cmd.sh"
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    # # make sure remove the old patch
    # if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
    #     os.remove(filename + '/patch_of_' + fnm + ".c")
    # A list to store the results we want to save, e.g. the repair time, the size of bound
    res = []
    res.append(idx)
    res.append(fnm)
    # if max_ret == []:
    #     max_ret = searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    # else:
    #     searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    #     num_line = 0
    #     print "max"
    #     print max_ret
    staTime = time.time()
    num_line = 0
    try:
        signal.alarm(limit_time)
        # call the detector to find I_err
        print "Executing the PTB algorithm"
        # max_x, bound, bound_l = detectHighErrs(max_ret,th, rf, pf)
        t1 = time.time() - staTime
        res.append(th)
        res.append(t1)
        res.append(bound)
        bound_distance = [bf.getUlpError(bound_i[0], bound_i[1]) for bound_i in bound]
        res.append(bound_distance)
        ori_bound = bound
        print bound
        print res
        staTime2 = time.time()
        max_x = max_ret[1]
        # call the derivingApproximation to produce approximation and produce patch
        # save_lines,num_line = derivingApproximation_taylor3v(th,ori_bound,rf,pf,fnm,filename,max_x,lock)
        # save_lines, num_line = derivingApproximation(th, bound_l, rf, fnm, filename, max_x)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # combCovertToC(bound_l, fnm, filename)
        combCovertToC_tay3v(bound, fnm, filename)
        temp_t = time.time() - staTime2
        res.append(temp_t)
        # patch_generate(ori_bound, fnm, filename)
        total_time = time.time() - staTime
        res.append(total_time)
        res.append(rd_seed)
        print "Repair time: " + str(temp_t)
        print "patch is generate, the name is " + 'patch_of_' + fnm + ".c"
        size_file = 0.0
        if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
            size_file = os.path.getsize(filename + '/patch_of_' + fnm + ".c")
        res.append(size_file)
        res.append(num_line)
        bf.glob_point_l = []
        return res
    except TimeoutError:
        print 'timeout'
        res.append(0.0)
        res.append(0.0)
        res.append(rd_seed)
        res.append(0)
        res.append(0.0)
        res.append(0.0)
        return res



def main4v(rf,pf,level,th,rd_seed, fnm, limit_time, num, password,max_ret,bound,idx,lock):
    fnm = fnm.strip()
    print "Begin repair the function "+fnm
    # generate file to store patch files
    filename = "../experiments/experiment_results4v/repair_results" + str(num) + "/test" + repr(int(level * 10))
    if not os.path.exists(filename):
        os.makedirs(filename)
    # generate file to save lines
    line_filename = "../experiments/experiment_results4v/repair_results" + str(num) + "/lines" + repr(int(level * 10))
    if not os.path.exists(line_filename):
        os.makedirs(line_filename)
    # if not os.path.exists(filename + "/patch"):
    #     os.makedirs(filename + "/patch")
    # generate shell scripts to apply patches
    # if os.path.exists(filename + "/patch/patch_cmd.sh"):
    #     os.remove(filename + "/patch/patch_cmd.sh")
    # f = open(filename + "/patch/patch_cmd.sh", "a")
    # f.write("#!/usr/bin/env bash\n")
    # f.close()
    # sudoPassword = password
    # command = 'chmod 777 ' + filename + "/patch/patch_cmd.sh"
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    # # make sure remove the old patch
    # if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
    #     os.remove(filename + '/patch_of_' + fnm + ".c")
    # A list to store the results we want to save, e.g. the repair time, the size of bound
    res = []
    res.append(idx)
    res.append(fnm)
    # if max_ret == []:
    #     max_ret = searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    # else:
    #     searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n)
    #     num_line = 0
    #     print "max"
    #     print max_ret
    num_line = 0
    staTime = time.time()
    try:
        signal.alarm(limit_time)
        # call the detector to find I_err
        print "Executing the PTB algorithm"
        # max_x, bound, bound_l = detectHighErrs(max_ret,th, rf, pf)
        t1 = time.time() - staTime
        res.append(th)
        res.append(t1)
        res.append(bound)
        bound_distance = [bf.getUlpError(bound_i[0], bound_i[1]) for bound_i in bound]
        res.append(bound_distance)
        ori_bound = bound
        print bound
        print res
        staTime2 = time.time()
        max_x = max_ret[1]
        # call the derivingApproximation to produce approximation and produce patch
        # save_lines,num_line = derivingApproximation_taylor4v(th,ori_bound,rf,pf,fnm,filename,max_x,lock)
        # save_lines, num_line = derivingApproximation(th, bound_l, rf, fnm, filename, max_x)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # bf.save_line_list(line_filename + '/' + fnm + '.txt', save_lines)
        # combCovertToC(bound_l, fnm, filename)
        # combCovertToC_tay2v(bound, fnm, filename)
        temp_t = time.time() - staTime2
        res.append(temp_t)
        # patch_generate(ori_bound, fnm, filename)
        total_time = time.time() - staTime
        res.append(total_time)
        res.append(rd_seed)
        print "Repair time: " + str(temp_t)
        print "patch is generate, the name is " + 'patch_of_' + fnm + ".c"
        size_file = 0.0
        if os.path.exists(filename + '/patch_of_' + fnm + ".c"):
            size_file = os.path.getsize(filename + '/patch_of_' + fnm + ".c")
        res.append(size_file)
        res.append(num_line)
        bf.glob_point_l = []
        return res
    except TimeoutError:
        print 'timeout'
        res.append(0.0)
        res.append(0.0)
        res.append(rd_seed)
        res.append(0)
        res.append(0.0)
        res.append(0.0)
        return res