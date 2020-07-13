import bench2v as bv2
import pickle
from src.eft import *
import src.basic_func as bf
import numpy as np
from mpmath import *

tay_fun_lst = pickle.load( open( "/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_results/repair_results2/lines2/gsl_sf_gegenpoly_3.txt", "rb" ) )


def horner_eft_1v(cof,x):
    pi_res = 0.0
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1,len(cof)):
        ta,tb = TwoPro(sa,x)
        sa,sb = TwoSum(ta,cof[i])
        pi_res = tb+sb + pi_res*x
        # sig_res = sig_res + sig_res*x
    return sa,pi_res
    # return sa+pi_res

def horner_eft_1vx(cof,x):
    pi_res = 0.0
    # sig_res = 0.0
    sa = cof[0]
    for i in range(1, len(cof)):
        ta, tb = TwoPro(sa, x)
        sa, sb = TwoSum(ta, cof[i])
        pi_res = tb + sb + pi_res * x
        # sig_res = sig_res + sig_res*x
    # return [sa, pi_res]
    return sa+pi_res

def horner_1v(cof,x):
    res = cof[0]
    for i in range(1,len(cof)):
        res = res*x + cof[i]
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
        cof1,cof2 = horner_eft_1v(cof[k:k+i+1], y-input[1])
        temp_cof_x.append(cof1)
        temp_cof_x2.append(cof2)
        temp_cof_x_err.append(horner_1v(cof_err[k:k+i+1],y-input[1]))
        k = k+i+1
    temp_cof_x.reverse()
    temp_cof_x2.reverse()
    temp_cof_x_err.reverse()
    temp_res = horner_eft_1vx(temp_cof_x, x-input[0])
    # temp_res2 = horner_1v(temp_cof_x2, x-input[0])
    temp_res2 = horner_eft_1vx(temp_cof_x2, x-input[0])
    temp_err = horner_1v(temp_cof_x_err, x-input[0])
    return temp_res+temp_res2+temp_err

def taylor2_cof(rfdd,point,order):
    cof_l = []
    # mp.dps = 40
    mp.prec = 126
    cof_l.append(rfdd(*point))
    for i in range(1,order+1):
        for j in range(0,i+1):
            temp_cof = diff(rfdd,tuple(point),(i-j,j))/(factorial(i-j)*factorial(j))
            cof_l.append(temp_cof)
    cof_l.append(order)
    return cof_l
def taylor2_fun(cof,x,y,input):
    order = int(cof[-1])
    k = 0
    temp_l = []
    temp_res = 0.0
    mp.prec = 80
    x1 = mpf(x)
    y1 = mpf(y)
    for i in range(order+1):
        for j in range(0,i+1):
            temp_cof = cof[k]
            temp_res = fadd(temp_res,fmul(temp_cof,fmul(pow(fsub(x , input[0]),i-j),pow(fsub(y, input[1]),j),exact=True),exact=True),exact=True)
            k = k + 1
    return temp_res

def point_in_bound(point,bound):
    ls = len(bound)
    flag = 0
    for i,j in zip(point,bound):
        if (i<j[1])&(i>j[0]):
            flag = flag+1
    if flag ==ls:
        return 1
    else:
        return 0


def tay_idx_fun(tay_fun_lst,x,y):
    point = [x,y]
    for tal in tay_fun_lst:
        if point_in_bound(point,tal[3]):
            cof_float = tal[1]
            cof_float_err = tal[0]
            new_point = tal[2]
            return taylor2_horner_eft_ori2(cof_float, cof_float_err, x, y, new_point)

rf = bv2.rfl[24]
# cof = taylor2_cof(rf,point,int(cof_float[-1]))
# mp.dps = 50
tay_fun = lambda x,y: tay_idx_fun(tay_fun_lst,x,y)
# tay_fun2 = lambda x, y: taylor2_fun(cof, x, y, point)
tpoint = [-1.9999999997623754, -79451.10510568558-bf.getulp(-79451.10510568558)]
point = [-1.9999999997623754, -79451.10510568558-bf.getulp(-79451.10510568558)]
bound = tay_fun_lst[0][3]
a = rf(*tpoint)
b = tay_fun(*tpoint)
print "%.18e" % point[0]
print "%.18e" % point[1]
print "a %.18e" %a
print "%.18e" % (a+bf.getulp(a))
print "%.18e" % (b+bf.getulp(a))
print rf(*point)
print bf.getUlpError(a,b)
print np.log2(bf.getUlpError(a,b))
print bf.getUlpError(tpoint[0],bound[0][1])/1e12
print bf.getUlpError(tpoint[1],bound[1][1])/1e12
#-3.560754883951525005e-18
#-3.560754883952511081e-18