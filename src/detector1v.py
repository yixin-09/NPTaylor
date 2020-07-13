from detector import DEMC
import basic_func as bf
import numpy as np


# step back to refine the "badland"
# step back to refine the "badland"
def step_back(th,temp_b1,step,rf,pf,sign,ulp_p):
    print "step back point"
    ori_value = temp_b1 + ulp_p*step*sign
    print "%.18e" % ori_value
    print "%.18e" % temp_b1
    step = step/2.0
    while(sign*temp_b1<sign*ori_value):
        temp_b1 = temp_b1 + ulp_p*step*sign
        # max_err_mid = estimate_error(point,p0_err,rf,pf,temp_b1)
        max_err_mid,temp_b1 = bf.max_errorOnPoint(rf,pf,temp_b1,step)
        # if (max_err_mid-th <=5.0):
        #     return temp_b1
        if (max_err_mid > th):
            if step < 100.0:
                return temp_b1
            else:
                temp_b1 = temp_b1 - ulp_p*step*sign
                step = step / 2.0
    return ori_value
def step_back2(th,temp_b1,step,rf,pf,sign,ulp_p,temp_max_err):
    print "step back point"
    ori_value = temp_b1 + ulp_p*step*sign
    print "%.18e" % ori_value
    print "%.18e" % temp_b1
    step = step/2.0
    while(sign*temp_b1<sign*ori_value):
        temp_b1 = temp_b1 + ulp_p*step*sign
        # max_err_mid = estimate_error(point,p0_err,rf,pf,temp_b1)
        max_err_mid,temp_b1 = bf.max_errorOnPoint(rf,pf,temp_b1,step)
        # if (max_err_mid-th <=5.0):
        #     return temp_b1
        if step < 100.0:
            return temp_b1
        if (max_err_mid < th):
            return temp_b1
        if (np.abs(max_err_mid-temp_max_err) < 100):
            return ori_value+(temp_b1-ori_value)/2.0
        else:
            if max_err_mid<temp_max_err:
                temp_b1 = temp_b1 - ulp_p * step * sign
                step = step / 2.0
            else:
                step = step / 2.0
    return ori_value
# exec the pointToBound algorithm
def pointToBound(th,rf,pf,point):
    print "Begin Find the bound around the inputs and under the threshold"
    # right ward iteration
    step = 4e2
    print "Right forward to find the up bound"
    print point
    ulp_p = bf.getulp(point)
    print ulp_p
    p0_err = bf.getUlpError(rf(point), pf(point))
    temp_b1 = point
    temp_max = p0_err
    for i in range(0,int(4e2)):
        temp_b1 = temp_b1 + ulp_p*step
        max_err_mid, temp_b1 = bf.max_errorOnPoint(rf, pf, temp_b1, step)
        try:
            times = np.max([np.log10(max_err_mid / th), 2.0])
        except AttributeError:
            times = 1.0
        if (max_err_mid < th)|(temp_max<max_err_mid):
            if (temp_max<max_err_mid):
                temp_b1 = step_back2(th, temp_b1, step, rf, pf, -1, ulp_p,temp_max)
            else:
                temp_b1 = step_back(th,temp_b1,step,rf,pf,-1,ulp_p)
            bound_up = temp_b1
            break
        step = int(step * times)
        temp_max = max_err_mid
    print "Left forward to find the down bound"
    step = 4e2
    temp_b1 = point
    temp_max = p0_err
    for i in range(0, int(4e2)):
        temp_b1 = temp_b1 - ulp_p * step
        max_err_mid, temp_b1 = bf.max_errorOnPoint(rf, pf, temp_b1, step)
        try:
            times = np.max([np.log10(max_err_mid / th), 2.0])
        except AttributeError:
            times = 1.0
        # print step
        # times = np.max([np.log2(max_err_mid / th), 2.0])
        if (max_err_mid < th)|(temp_max<max_err_mid):
            # print step / times
            if (temp_max < max_err_mid):
                temp_b1 = step_back2(th, temp_b1, step, rf, pf, 1, ulp_p, temp_max)
            else:
                temp_b1 = step_back(th, temp_b1, step, rf, pf, 1, ulp_p)
            bound_down = temp_b1
            break
        step = int(step * times)
        temp_max = max_err_mid
    return [bound_down,bound_up]


def searchMaxErr(rf,pf,inpdm,fnm,limit_time,limit_n):
    # DEMC: find the input max_x that trigger the maximum floating-point in a input domain
    res = DEMC(rf, pf, inpdm, fnm, limit_n, limit_time)
    return res

def detectHighErrs(ret, th,rf,pf):
    max_x = ret[1]
    max_error = ret[0]
    #calculate the threshold
    #PTB: find the input interval includes the inputs higher than an give threshold
    bound = pointToBound(th, rf, pf, max_x)
    #Partition the bound into small intervals to keep same ulp value in each interval
    bound_l = bf.bound_partition(bound)
    return max_x, bound, bound_l