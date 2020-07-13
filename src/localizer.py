# localizing the input interval around the input that can trigger the possible maximum error
import basic_func as bf
import numpy as np
import itertools
from numpy.polynomial import Polynomial as P
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numpy.polynomial.polynomial import polyval
from scipy.optimize import curve_fit
from detector import DDEMC_pure
from scipy.optimize import differential_evolution
from mpmath import diff
import itertools


def plot_3D_error(X,Y,Z1,Z2,U,Yl):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(X, Y, Z,'.', label='parametric curve')
    # ax2.plot(Xl3, Yl3, U2,'.', label='parametric curve')
    ax.plot(X, Y, Z1, '.')

    # ax.plot(X, Y, Z2, '.')
    Zeo = []
    for i in U:
        Zeo.append(0)
    ax.plot(X, Yl,Zeo,'-')
    # ax.plot(X, Y, U, '.')
    # ax.plot(X, Y, Zeo, '.')
    # ax.plot(X, Yl, label='line approximation')
    # ax.plot(X, Y, label='line approximation')
    # ax2.plot(Xl2, Yl, label='line approximation')
    ax.legend()
    plt.show()

# generate a  bound for a given point
def generate_bound(point,ini_step):
    ini_bound = []
    for i in point:
        ini_bound.append([i-ini_step*bf.getulp(i),i+ini_step*bf.getulp(i)])
    return ini_bound

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


def get_poly_fit(xdata,ydata,i):
    # get the polynomial fit
    z = np.polyfit(xdata, ydata, i)
    return z

# extract a line function to trace the maximum error
def mid_line_finder(th,input_l,res_l):
    step = 0.2
    iter_n = int((64 - th)/step)
    cof_l = []
    for th_i in range(0, iter_n):
        temp_th = th + th_i*step
        X = []
        Y = []
        Z = []
        for i,j in zip(input_l,res_l):
            if float(math.log(j)) > temp_th:
                X.append(i[0])
                Y.append(i[1])
                Z.append(float(math.log(j)))
        if (len(X) < 100)&(len(X) > 30):
            z = get_poly_fit(X, Y, 1)
            cof_l.append(z)
    p_cof = []
    for j in range(0,len(cof_l[0])):
        sum = 0
        for i in cof_l:
            sum = sum + i[j]
        p_cof.append(sum/(len(cof_l)))
    return np.poly1d(p_cof)


def generate_point(point,step,p,sgn):
    x = point[0]+sgn*step*bf.getulp(point[0])
    y = p(x)
    return [x,y]

def plot_line_err(rf,pf,input_l,p):
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    X = []
    Y = []
    Yl = []
    Z = []
    Z2 = []
    U = []
    st = time.time()
    for i in input_l:
        temp_err = 1.0 / glob_fitness_real(i)
        # temp_err2 = glob_fitness_real2(i)
        # print log(temp_err)
        X.append(i[0])
        Y.append(i[1])
        Yl.append(p(i[0]))
        Z.append(float(math.log(temp_err)))
        U.append(float(rf(i[0], i[1])))
    print "run time is "
    print time.time() - st
    print max(Z)
    print Yl[0:10]
    plot_3D_error(X, Y, Z, Z2, U, Yl)



def find_bound(point,ini_step,p,rf,pf,th,sgn):
    temp_point = point
    temp_step = ini_step
    new_point = []
    temp_p = p
    for i in range(0, 400):
        new_point = generate_point(temp_point, temp_step, temp_p, sgn)
        new_bound = generate_bound(new_point, ini_step*100)
        res_demc = DDEMC_pure(rf, pf, [new_bound], 3, 100)
        # glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
        # res_l = []
        # sum_err = 0
        # input_l = produce_n_input(new_bound, 50)
        # for j in input_l:
        #     temp_err = 1.0 / glob_fitness_real(i)
        #     res_l.append(temp_err)
        # # plot_line_err(rf, pf, input_l, temp_p)
        temp_max_err = math.log(res_demc[0])
        # temp_p = mid_line_finder(th, input_l, res_l)
        print bf.getUlpError(res_demc[1][1], temp_p(res_demc[1][0]))
        if temp_max_err < th:
            print "get here"
            print temp_max_err
            print th
            print temp_step
            temp_step = temp_step / 2.0
            ori_point = temp_point
            now_point = new_point
            while 1:
                print temp_step
                print temp_point
                now_point = generate_point(now_point, temp_step, temp_p, -1 * sgn)
                print now_point
                new_bound = generate_bound(now_point, ini_step*100)
                res_demc = DDEMC_pure(rf, pf, [new_bound], 3, 100)
                temp_max_err = math.log(res_demc[0])
                if (temp_max_err > th):
                    print temp_max_err
                    print th
                    now_point = generate_point(now_point, temp_step, temp_p, sgn)
                    temp_step = temp_step / 2.0
                    if temp_step < 10.0:
                        return now_point
                else:
                    if temp_step < 100.0:
                        print "get first break"
                        return now_point
    #                     break
    #         break
            # new_point = res_demc[1]
            # break
        print th
        print temp_max_err
        temp_point = new_point
        temp_step = temp_step * 3
    # new_bound = generate_bound(new_point, ini_step)
    # res_demc = DDEMC_pure(rf, pf, [new_bound], "bessel_jnu", 2, 100)
    # temp_max_err = math.log(res_demc[0])
    # print temp_step
    # print new_point
    # print point
    # print bf.getUlpError(new_point[0], point[0])
    # print bf.getUlpError(new_point[1], point[1])
    # print th
    # print temp_max_err
    # new_bound = [[point[0], new_point[0]], [point[1], new_point[1]]]
    # input_l = produce_n_input(new_bound, 100)
    # plot_line_err(rf, pf, input_l, p)
    # return 0

# Only two demension is consider in the PTB function
def PTB_2D(max_err,point,rf,pf,th):
    ini_step = 1e4
    print "Begin to find the bound"
    # getting a initial bound
    ini_bound = generate_bound(point,ini_step)
    print ini_bound
    # sampling in ini_bound
    input_l = produce_n_input(ini_bound, 100)
    # extract a line function to trace the maximum error
    glob_fitness_real = lambda x: bf.mfitness_fun(rf, pf, x)
    res_l = []
    sum_err = 0
    for i in input_l:
        temp_err = 1.0/glob_fitness_real(i)
        res_l.append(temp_err)
        sum_err = sum_err+math.log(temp_err)
    print sum_err
    temp_th = (th-sum_err/len(res_l))/2.0 + sum_err/len(res_l)
    print temp_th
    p = mid_line_finder(temp_th, input_l,res_l)
    print p
    # trace the line function to find new bound
    up_point = find_bound(point, ini_step, p, rf, pf, th, 1)
    down_point = find_bound(point, ini_step, p, rf, pf, th, -1)
    new_bound = [[down_point[0], up_point[0]], [down_point[1], up_point[1]]]
    temp_res = DDEMC_pure(rf, pf, [new_bound],3, 100)
    new_point = temp_res[1]
    return new_bound,new_point
def generate_mix_bound(old_bound,new_bound):
    mix_bound = []
    for i,j in zip(old_bound,new_bound):
        mix1 = [j[0], i[0]]
        mix2 = [i[1], j[1]]
        mix_bound.append([mix1, mix2])
    return mix_bound
def generate_testing_box(old_bound,new_bound):
    mix_bound = generate_mix_bound(old_bound,new_bound)
    testing_box = []
    for i in range(0,len(new_bound)):
        bl_id = range(0, len(new_bound))
        bl_id.remove(i)
        for kj in mix_bound[i]:
            temp_box = []
            for pk in bl_id:
                temp_box.append(new_bound[pk])
            temp_box.insert(i,kj)
            testing_box.append(temp_box)
    return testing_box

def check_bound_over_inpdm(new_bound,inpdm):
    stop_flag = 0
    for i in range(0,len(new_bound)):
        if new_bound[i][0] <= inpdm[i][0]:
            new_bound[i][0] = inpdm[i][0]
            stop_flag = stop_flag + 1
        if new_bound[i][1] >= inpdm[i][1]:
            new_bound[i][1] >= inpdm[i][1]
            stop_flag = stop_flag + 1
    if stop_flag == 2:
        return new_bound,1
    else:
        return new_bound,0
# the algorithm tris to find the bound for multiple arguments functions

def PTB_Box(max_err,point,rf,pf,th):
    ini_step = 8e13
    ini_bound = generate_bound(point,ini_step)
    old_bound = ini_bound
    new_step = 1e15
    for j in range(0,100):
        print "testing number %d" % j
        new_step = new_step+ini_step
        new_bound = generate_bound(point, new_step)
        testing_box = generate_testing_box(old_bound, new_bound)
        old_bound = new_bound
        for i in testing_box:
            # print i
            res_demc = DDEMC_pure(rf, pf, [i], 1, 100)
            print res_demc
        if j == 25:
            print point
            print new_bound
            print bf.getUlpError(new_bound[0][0], point[0])
            print bf.getUlpError(new_bound[0][1], point[0])
            print bf.getUlpError(new_bound[1][0], point[1])
            print bf.getUlpError(new_bound[1][1], point[1])
    print point
    print new_bound
    print bf.getUlpError(new_bound[0][0], point[0])
    print bf.getUlpError(new_bound[0][1], point[0])
    print bf.getUlpError(new_bound[1][0], point[1])
    print bf.getUlpError(new_bound[1][1], point[1])
    return 0
def generate_bound_id(point,ini_step,new_step,id):
    ini_bound = []
    count = 0
    if id == 0:
        for i in point:
            ini_bound.append([i-new_step*bf.getulp(i),i+new_step*bf.getulp(i)])
    else:
        count = count + 1
        for i in point:
            if id != count:
                ini_bound.append([i - new_step * bf.getulp(i), i + new_step * bf.getulp(i)])
            else:
                ini_bound.append([i - ini_step * bf.getulp(i), i + ini_step * bf.getulp(i)])
            count = count + 1
    return ini_bound

def drop_not_box(testing_box):
    count = 0
    live_box = []
    for i in testing_box:
        for j in i:
            if j[0]==j[1]:
                count = 1
        if count == 0:
            live_box.append(i)
        count = 0
    return live_box


def PTB_Err_tracing(max_err,point,rf,pf,th):
    ini_step = 1e3
    ini_bound = generate_bound_id(point,ini_step,ini_step,1)
    old_bound = ini_bound
    new_step = ini_step
    for j in range(0, 50):
        print "testing number %d" % j
        new_step = new_step*2.0
        new_bound = generate_bound_id(point,ini_step, new_step,1)
        testing_box = drop_not_box(generate_testing_box(old_bound,new_bound))
        for i in testing_box:
            # print i
            res_demc = DDEMC_pure(rf, pf, [i], 1, 100)
            print res_demc
            max_err=res_demc[0]
            if np.log2(max_err)<th:
                print "reach a point"
                print point
                print old_bound
                print new_bound
                print bf.getUlpError(new_bound[0][0], point[0])
                print bf.getUlpError(new_bound[0][1], point[0])
                print bf.getUlpError(new_bound[1][0], point[1])
                print bf.getUlpError(new_bound[1][1], point[1])
        old_bound = new_bound

def get_mid(bound):
    return bound[0]+(bound[1]-bound[0])/2.0


def accuracy_condition(point, rf):
    der_tuple = [0]*len(point)
    point_condition = 0.0
    for i in range(0,len(point)):
        temp_tuple = der_tuple
        temp_tuple[i] = 1
        point_condition += diff(rf, tuple(point),tuple(temp_tuple))*bf.getulp(point[i])
    ulp_point = bf.getulp(rf(*point))
    point_condition = point_condition/ulp_point
    return math.fabs(point_condition)

def generate_points_from_box(testing_box_i):
    temp_points = []
    for j in itertools.product(*testing_box_i):
        temp_points.append(j)
    return temp_points

def estimate_point_err(testing_box_i,rf,backward_err):
    # ulp_pf = bf.getulp(pf(*temp_point))
    # ulp_rf = bf.getulp(rf(*temp_point))
    # ulp_temp_point_out = min(ulp_pf,ulp_rf)
    temp_points = generate_points_from_box(testing_box_i)
    # temp_point_out = []
    max_cod = 0.0
    temp_cod = 0.0
    sign = 1
    temp_point = []
    for i in temp_points:
        temp_cod = accuracy_condition(i,rf)
        if temp_cod > max_cod:
            max_cod = temp_cod
    return max_cod*backward_err


def estimate_backward_error(inpdm,rf,pf):
    input_l = produce_n_input(inpdm, 5)
    back_err = []
    for i in input_l:
        err = math.fabs((rf(*i)-pf(*i))/bf.getulp(rf(*i)))
        cod = accuracy_condition(i,rf)
        back_err.append(err/cod)
    return max(back_err)

glob_bound = []
index_bound = 0
def generate_next_bound(rf,pf,ini_bound,i,ibt,flag_up,th):
    # print "get the one bound"
    # print ibt
    # print bf.getUlpError(ibt[0], ibt[1])
    global glob_bound
    global index_bound
    index_bound = index_bound + 1
    print index_bound
    if flag_up == 1:
        temp_bound = ini_bound
        temp_bound[i] = [get_mid(ibt),ibt[1]]
        if (bf.getUlpError(ibt[0], ibt[1]) < 1000):
            # print "reach the out point"
            ini_bound[i] = [ini_bound[i][0],ibt[1]]
            glob_bound = ini_bound
            return 0
        temp_res = DDEMC_pure(rf, pf, [temp_bound], 1, 1000)
        max_err = np.log2(temp_res[0])
        if max_err < th:
            ibt = [ibt[0], get_mid(ibt)]
            generate_next_bound(rf, pf, ini_bound, i, ibt, flag_up, th)
        else:
            ibt = [temp_res[1][i]+bf.getulp(temp_res[1][i]), ibt[1]]
            generate_next_bound(rf, pf, ini_bound, i, ibt, flag_up, th)
    else:
        temp_bound = ini_bound
        temp_bound[i] = [ibt[0],get_mid(ibt)]
        if (bf.getUlpError(ibt[0], ibt[1]) < 1000):
            # print "reach the out point"
            ini_bound[i] = [ibt[0], ini_bound[i][1]]
            glob_bound = ini_bound
            return 0
        temp_res = DDEMC_pure(rf, pf, [temp_bound], 1, 1000)
        max_err = np.log2(temp_res[0])
        if max_err < th:
            ibt = [get_mid(ibt), ibt[1]]
            generate_next_bound(rf, pf, ini_bound, i, ibt, flag_up, th)
        else:
            ibt = [ibt[0], temp_res[1][i]-bf.getulp(temp_res[1][i])]
            generate_next_bound(rf, pf, ini_bound, i, ibt, flag_up, th)
# let us try 2-inputs functions first
def step_back_2v(point,ini_bound,rf,pf,th,id):
    # print "get the step back point"
    global glob_bound
    for i in range(0,len(point)):
        if i != id-1:
            ipt = point[i]
            ibt = ini_bound[i]
            if ipt < ibt[0]:
                flag_up = 1
                # temp_bound = ini_bound
                temp_res = DDEMC_pure(rf, pf, [ini_bound], 1, 1000)
                ibt = [temp_res[1][i]+bf.getulp(temp_res[1][i]), ibt[1]]
                generate_next_bound(rf,pf,ini_bound, i, ibt, flag_up,th)
            else:
                flag_up = 0
                # temp_bound = ini_bound
                temp_res = DDEMC_pure(rf, pf, [ini_bound], 1, 1000)
                ibt = [ibt[0], temp_res[1][i]-bf.getulp(temp_res[1][i])]
                generate_next_bound(rf,pf,ini_bound, i, ibt, flag_up,th)
    new_bound = glob_bound
    glob_bound = []
    return new_bound

def bound_arround_one_point(point, rf, pf, th,id,inpdm,point_max_err):
    ini_step = 1e3
    ini_bound = generate_bound_id(point, ini_step, ini_step, id)
    old_bound = ini_bound
    point_condition = accuracy_condition(point,rf)
    print "condition is "
    print np.log2(point_condition)
    log_point_condition = np.log2(point_condition)
    new_step = np.power(2.0,(log_point_condition-th))
    stop_var_fig = 0
    temp_testing_box = []
    count_n = 0
    new_boxing_l = []
    stop_lst = []
    step_ajust = 2.0
    for j in range(0, 1000):
        print "testing number %d" % j
        new_step = new_step * step_ajust
        print new_step
        # print new_step
        new_bound = generate_bound_id(point, ini_step, new_step, id)
        new_bound, stop_search_flag = check_bound_over_inpdm(new_bound, inpdm)
        if stop_search_flag == 1:
            return new_bound
        testing_box = drop_not_box(generate_testing_box(old_bound, new_bound))
        count_id = 0
        for i in testing_box:
            if count_id not in stop_lst:
                res_demc = DDEMC_pure(rf, pf, [i], 1, 1000)
                # print res_demc
                max_err = res_demc[0]
                max_point = res_demc[1]
                max_point_condition = accuracy_condition(max_point, rf)
                print "max_err_condition is "
                print np.log2(max_point_condition)
                print np.log2(max_point_condition)/log_point_condition
                print np.log2(max_err)/np.log2(point_max_err)
                print np.log2(max_err)
                if np.log2(max_err) < th:
                    # print "reach a point"
                    if temp_testing_box != []:
                        st = time.time()
                        new_boxing = step_back_2v(point, temp_testing_box[count_id], rf, pf, th, id)
                        print "back time is"
                        print time.time()-st
                        new_boxing_l.append(new_boxing)
                        stop_lst.append(count_id)
                        # print point
                        # print temp_testing_box[count_id]
                        # print new_boxing
                        # print bf.getUlpError(new_boxing[1][1], point[1])
                    count_n = count_n + 1
            count_id = count_id + 1
        if count_n == (len(point) - 1) * 2:
            final_bound = new_boxing_l[0]
            # print new_boxing_l
            for i in new_boxing_l:
                for j in range(len(i)):
                    if j != id - 1:
                        if i[j][0] > point[0]:
                            final_bound[j][1] = i[j][1]
                        else:
                            final_bound[j][0] = i[j][0]
            return check_bound_over_inpdm(final_bound,inpdm)[0]
        temp_testing_box = testing_box
        old_bound = new_bound












def bound_arround_one_point3(point, rf, pf, th,id,inpdm):
    ini_bound = generate_bound_id(point, 1e3, 1e3, id)
    back_err = estimate_backward_error(inpdm, rf, pf)
    old_bound = ini_bound
    new_step = np.power(2.0,45-th)
    temp_testing_box = []
    count_n = 0
    new_boxing_l = []
    stop_lst = []
    stop_search_flag = 0
    step_log_lst = []
    estimate_just = 1.0
    for j in range(0, 1000):
        print "testing number %d" % j
        new_step = new_step * 2.0
        step_log_lst.append(new_step)
        # print new_step
        new_bound = generate_bound_id(point, 1e3, new_step, id)
        new_bound,stop_search_flag = check_bound_over_inpdm(new_bound,inpdm)
        if stop_search_flag == 1:
            return new_bound
        testing_box = drop_not_box(generate_testing_box(old_bound, new_bound))
        count_id = 0
        for i in testing_box:
            if count_id not in stop_lst:
                # res_demc = DDEMC_pure(rf, pf, [i], 1, 1000)
                # print res_demc
                # max_err = res_demc[0]
                # estimate the maximum error in the i testing_box
                estimate_max_error = estimate_point_err(i, rf, back_err) * estimate_just
                if np.log2(estimate_max_error) < th:
                    res_demc = DDEMC_pure(rf, pf, [i], 1, 1000)
                    print res_demc
                    max_err = res_demc[0]
                    print max_err
                    if np.log2(max_err) < th:
                        print "max_error"
                        print max_err
                        # jump back to accuracy estimate the bound
                        if temp_testing_box != []:
                            st = time.time()
                            print "step is"
                            print new_step
                            # new_boxing = step_back_2v(point, temp_testing_box[count_id], rf, pf, th, id)
                            print "back time is"
                            print time.time() - st
                            new_boxing_l.append(temp_testing_box[count_id])
                            # new_boxing_l.append(new_boxing)
                            stop_lst.append(count_id)
                            # print point
                            # print temp_testing_box[count_id]
                            # print new_boxing
                            # print bf.getUlpError(new_boxing[1][1], point[1])
                        count_n = count_n + 1
                    else:
                        estimate_just = max_err / estimate_max_error
                # if np.log2(max_err) < th:
                #     # print "reach a point"
                #     if temp_testing_box != []:
                #         st = time.time()
                #         new_boxing = step_back_2v(point, temp_testing_box[count_id], rf, pf, th, id)
                #         print "back time is"
                #         print time.time()-st
                #         new_boxing_l.append(new_boxing)
                #         stop_lst.append(count_id)
                #         # print point
                #         # print temp_testing_box[count_id]
                #         # print new_boxing
                #         # print bf.getUlpError(new_boxing[1][1], point[1])
                #     count_n = count_n + 1
            count_id = count_id + 1
        if count_n == (len(point) - 1) * 2:
            final_bound = new_boxing_l[0]
            # print new_boxing_l
            for i in new_boxing_l:
                for j in range(len(i)):
                    if j != id - 1:
                        if i[j][0] > point[0]:
                            final_bound[j][1] = i[j][1]
                        else:
                            final_bound[j][0] = i[j][0]
            return check_bound_over_inpdm(final_bound,inpdm)[0]
        temp_testing_box = testing_box
        old_bound = new_bound


def gen_next_test_bound(bound_around_point,id,sign):
    next_test_bound = []
    for i in range(len(bound_around_point)):
        if i == id-1:
            ulp_dis = bf.getUlpError(bound_around_point[i][0],bound_around_point[i][1])
            ulp_p = bf.getulp(bound_around_point[i][0])
            if sign == 1:
                next_test_bound.append([bound_around_point[i][1],bound_around_point[i][1]+sign*ulp_dis*ulp_p])
            else:
                next_test_bound.append([bound_around_point[i][0] + sign * ulp_dis * ulp_p,bound_around_point[i][0]])
        else:
            next_test_bound.append(bound_around_point[i])
    return next_test_bound

def extract_bound_point(bound_around_point,point,id):
    point0 = []
    point1 = []
    for i in range(len(point)):
        if i != id - 1:
            point0.append(bound_around_point[i][0])
            point1.append(bound_around_point[i][1])
        else:
            point0.append(point[0])
            point1.append(point[0])
    return point0,point1


def get_poly_2v(lst,n):
    xdata = []
    ydata = []
    for i in lst:
        xdata.append(i[0])
        ydata.append(i[1])
    p = np.poly1d(get_poly_fit(xdata,ydata,n))
    return p

# trying to find the bound
def PTB_Err_tracing_2v_small(point,rf,pf,th,inpdm,point_max_err):
    print "begin to search the bound"
    max_err_point_lst = []
    next_test_bound = []
    print pf(*point)
    glob_fitness_pf = lambda x: np.fabs(pf(*x))
    print glob_fitness_pf(point)
    id = 2
    bound_around_point = bound_arround_one_point(point, rf, pf, th,id,inpdm,point_max_err)
    next_test_bound = bound_around_point
    print "id is 2"
    print bound_around_point
    print bf.getUlpError(bound_around_point[0][0],bound_around_point[0][1])
    print bf.getUlpError(bound_around_point[1][0],bound_around_point[1][1])
    point0, point1 = extract_bound_point(bound_around_point, point, id)
    print point0
    print point1
    id = 1
    print "id is 1"
    bound_around_point = bound_arround_one_point(point, rf, pf, th, id,inpdm,point_max_err)
    print bound_around_point
    print bf.getUlpError(bound_around_point[0][0], bound_around_point[0][1])
    print bf.getUlpError(bound_around_point[1][0], bound_around_point[1][1])
    point0, point1 = extract_bound_point(bound_around_point, point, id)
    print point0
    print point1
    next_test_bound[id] = bound_around_point[id]
    print next_test_bound
    return next_test_bound

def bound_divide(ini_bound):
    temp_bound = []
    for i in ini_bound:
        temp_size = (i[1]-i[0])/3.0
        temp_bound.append([i[0]+temp_size,i[1]-temp_size])
    input_lst = []
    for i, j in zip(ini_bound, temp_bound):
        input_lst.append([[i[0], j[0]], [j[0], j[1]], [j[1], i[1]]])
    new_bound_l = []
    for element in itertools.product(*input_lst):
        new_bound_l.append(list(element))
    new_bound_l.remove(temp_bound)
    return new_bound_l


def fake_rf2v(rf,inp):
    return math.fabs(float(rf(*inp)))

def root_find_rf2v(rf,point,new_bound):
    try:
        glob_fitness_con = lambda x: fake_rf2v(rf, x)
        res = differential_evolution(glob_fitness_con, popsize=25, bounds=new_bound, polish=True, strategy='best1bin')
        return res.x
    except (ValueError, ZeroDivisionError, OverflowError, Warning, TypeError):
        return point

def PTB_MaxErr_tracing(point,rf,pf,new_bound):
    new_bound_l = bound_divide(new_bound)
    # print new_bound_l
    points_lst = []
    temp_lst = []
    new_points_lst = []
    for i in new_bound_l:
        temp_res = DDEMC_pure(rf, pf, [i], 3, 20000)
        max_err = temp_res[0]
        points_lst.append([max_err,temp_res[1]])
    temp_lst = sorted(points_lst,reverse=True)[0:1]
    new_point = temp_lst[0][1]
    k = (point[1]-new_point[1])/(point[0]-new_point[0])
    b = point[1]
    # temp_lst.append([1, point])
    # print temp_lst
    # A = []
    # b = []
    # for i in temp_lst:
    #     A.append([i[1][0],1])
    #     b.append(i[1][1])
    # x = lu_solve(A,b)
    return [k,b]


# print bf.getUlpError(3.35729845657e-11,-3.96885976895536587436017891139349349726927554967111458141376e-16)