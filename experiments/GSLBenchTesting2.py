import sys
import os
import bench1v as bv1
import bench2v as bv2
import bench3v as bv3
import bench4v as bv4
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.detector import DDEMC
from src.detector import DEMC
from mpmath import *
from xlutils.copy import copy
import xlrd
import xlwt
import importlib
import numpy as np
from multiprocessing import Pool,Lock,Manager
mp.dps = 30


def add_to_excel(exname,res_l):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    funname = res_l[-1]
    i = res_l[-2]
    sheet.write(i+1, 0, funname)
    sheet.write(i+1, 1, "DEMC")
    for j in range(0,len(res_l[0:-2])):
        sheet.write(i+1,j+2,repr(res_l[j]))
    new_excel.save(exname)


def ini_xls_file(exname):
    new_excel = xlwt.Workbook()
    sheet = new_excel.add_sheet("sheet1")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "Approach")
    sheet.write(0, 2, "max_error")
    sheet.write(0, 3, "input")
    sheet.write(0, 4, "interval")
    sheet.write(0, 5, "execute time")
    sheet.write(0, 6, "repeats")
    sheet.write(0, 7, "fitNum(C)")
    sheet.write(0, 8, "fitNum(ErrBits)")
    # sheet.write(0, 9, "fitNum(MCMC_jump)")
    new_excel.save(exname)

def testGSLfuncs(n_var,i,rf, pf, ipdm, fname):
    if n_var==1:
        res = DEMC(rf, pf, ipdm, fname, 1, 10800)
    else:
        res = DDEMC(rf, pf, ipdm, fname, 1,360000, n_var)
    res.append(i)
    res.append(fname)
    return res

if __name__ == "__main__":
    for n_var in range(2,3):
        ld_module = importlib.import_module("bench" + str(n_var) + "v")
        res_xls_name = "final_results_" + str(n_var) + "v.xls"
        ini_xls_file(res_xls_name)
        fname_lst = ld_module.ngfl_fname
        funcs_ids = range(1,len(fname_lst)+1)
        print funcs_ids
        def para_test_f(i):
            pf = ld_module.gfl[i - 1]
            rf = ld_module.rfl[i - 1]
            fname = ld_module.ngfl_fname[i - 1]
            if n_var==1:
                ipdm = ld_module.input_domain[i - 1][0]
            else:
                ipdm = ld_module.input_domain[i - 1]
            return testGSLfuncs(n_var, i-1, rf, pf, ipdm, fname)
        # for i in range(71,len(fname_lst)+1):
        #     print para_test_f(i)
        p = Pool(np.min([12, len(funcs_ids)]))
        for result in p.imap(para_test_f, funcs_ids):
            add_to_excel(res_xls_name, result)


