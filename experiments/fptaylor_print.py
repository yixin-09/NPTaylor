import bench2v as bv2
import pickle
from src.eft import *
import src.basic_func as bf
import numpy as np
from sympy import *
import sys

init_printing(use_unicode=True)
def gen_interval(bound,point):
    inv_lst = []
    for i,j in zip(bound,point):
        inv_lst.append([i[0]-j,i[1]-j])
    return inv_lst
def TwoSumstr(a,b):
    x = a + '+' + b
    z = '('+x + '-' + a +')'
    y = '('+a + '-' + '('+x + '-' + z+')'+')' + '+' + '('+b + '-' + z+')'
    return x, y

def Splitstr(a):
    z = '('+a + '*' + str(134217729.0)+')'
    x = '('+z + '-' + '('+z + '-' + a+')'+')'
    y = '('+a + '-' + x+')'
    return x,y

def TwoProstr(a,b):
    x = a + '*' + b
    ah,al = Splitstr(a)
    bh,bl = Splitstr(b)
    y = '('+al + '*' + bl + '-' + '((('+x+ '-' +ah+ '*' +bh+')'+ '-' +al+ '*' + bh+')'+ '-' +ah+ '*' + bl+')'+')'
    return x,y

def horner_eft_1vstr(cof,x):
    pi_res = "0.0"
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1,len(cof)):
        ta,tb = TwoProstr(sa,x)
        sa,sb = TwoSumstr(ta,cof[i])
        pi_res = '('+tb+ '+' +sb + '+' + pi_res+ '*' +x+')'
        # sig_res = sig_res + sig_res*x
    return sa,pi_res
    # return sa+pi_res

def horner_eft_1vxstr(cof,x):
    pi_res = "0.0"
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1, len(cof)):
        ta, tb = TwoProstr(sa, x)
        sa, sb = TwoSumstr(ta, cof[i])
        pi_res = '('+tb+ '+' +sb + '+' + repr(pi_res)+ '*' +x+')'
        # sig_res = sig_res + sig_res*x
    # return [sa, pi_res]
    return sa+ '+' +pi_res

def horner_1vstr(cof,x):
    res = cof[0]
    for i in range(1,len(cof)):
        res = '('+res+ '*' +x + '+' + cof[i]+')'
    return res
def taylor2_horner_eft_ori2(cof,cof_err,x,y,input):
    order = int(cof[-1])
    k = 0
    lst = range(order+1)
    lst.reverse()
    temp_cof_x = []
    temp_cof_x2 = []
    temp_cof_x_err = []
    for i in lst:
        cof1,cof2 = horner_eft_1vstr(cof[k:k+i+1], y-input[1])
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        temp_cof_x_err.append(horner_1vstr(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vxstr(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vxstr(temp_cof_x2, x-input[0])
    temp_err = horner_1vstr(temp_cof_x_err, x-input[0])
    return temp_res+temp_res2+temp_err
def taylor2_horner_eft_ori2str(cof,x,y):
    order = int(cof[-1])
    k = 0
    lst = range(order+1)
    lst.reverse()
    temp_cof_x = []
    temp_cof_x2 = []
    temp_cof_x_err = []
    for i in lst:
        cof1,cof2 = horner_eft_1vstr(cof[k:k+i+1], y)
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vxstr(temp_cof_x, x)
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_1vstr(temp_cof_x2, x)
    return temp_res+ '+' +temp_res2
def taylor2_horner_err2str(cof_err,x,y):
    order = int(cof_err[-1])
    k = 0
    lst = range(order+1)
    lst.reverse()
    temp_cof_x_err = []
    for i in lst:
        temp_cof_x_err.append(horner_1vstr(cof_err[k:k + i + 1], y))
        k = k + i + 1
    temp_err = horner_1vstr(temp_cof_x_err, x)
    return temp_err

a = 'a'
b = 'b'
print TwoSumstr(a,b)
print Splitstr(a)
print TwoProstr(a,b)
# a = Symbol('a')
# b = Symbol('b')
# print (a-((a + b)-(a + b-a)))+(b-(a + b-a))
# print_latex((a-((a + b)-(a + b-a)))+(b-(a + b-a)))
# print(sympify("(a-((a + b)-(a + b-a)))+(b-(a + b-a))", evaluate=False))
# with evaluate(False):
#     print (a-((a + b)-(a + b-a)))+(b-(a + b-a))
#     print TwoSumsp(a,b)

tay_fun_lst = pickle.load( open( "/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_results/repair_results2/lines2/gsl_sf_bessel_Ynu.txt", "rb" ) )
print tay_fun_lst[0][2]
bound = tay_fun_lst[0][3]
point = tay_fun_lst[0][2]
# cof_float = tay_fun_lst[0][1]
cof_float = tay_fun_lst[0][0] + [tay_fun_lst[0][1][-1]]
print cof_float
cof_float_str = []
inv_lst = gen_interval(bound,point)
orig_stdout = sys.stdout
f = open('fptest.txt', 'w')
idx = 0
sys.stdout = f
print "Variables"
print " float64 x in " + str(inv_lst[0])+","
print " float64 y in " + str(inv_lst[1])+","
for i in range(len(cof_float[:-2])):
    print " float64 a"+ str(i) + " in " + str([cof_float[i],cof_float[i]])+","
    cof_float_str.append("a"+ str(i))
print " float64 a"+ str(len(cof_float[:-2])) + " in " + str([cof_float[-2],cof_float[-2]])
cof_float_str.append("a"+ str(len(cof_float[:-2])))
cof_float_str.append(cof_float[-1])
print ";"
print
print "Definitions"
print " r rnd64="+ taylor2_horner_err2str(cof_float_str,'x','y')
print ";"
print
print "Expressions"
print " r"
print ";"
sys.stdout = orig_stdout
f.close()
# print cof_float_str
# print taylor2_horner_eft_ori2str(cof_float_str,'x','y')