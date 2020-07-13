import xlrd
import sys,ast

def final_bench(exname,id):
    orig_stdout = sys.stdout
    data = xlrd.open_workbook(exname)
    table = data.sheets()[id-1]
    print table.nrows
    f = open("bench"+str(id)+"v.py", 'w')
    sys.stdout = f
    count = 1
    bound_l = []
    rfl = []
    gfl = []
    efl = []
    nrfl = []
    ngfl = []
    snfl = []
    print "from pygsl.testing import sf"
    print "from mpmath import *"
    print "import numpy as np"
    # print "import warnings"
    # print "from scipy import special"''
    print "mp.dps = 30"
    for i in range(1,table.nrows):
        if (table.row_values(i)[2] != ''):
            print
            print "#f" + str(count)
            rfl.append("rf" + str(count))
            gfl.append("gf" + str(count))
            if id == 1:
                print "gf" + str(count) + " = lambda x: " + (table.row_values(i)[4])
                print "rf" + str(count) + " = lambda x: " + table.row_values(i)[3]
            if id == 2:
                print "gf" + str(count) + " = lambda x,y: " + (table.row_values(i)[4])
                print "rf" + str(count) + " = lambda x,y: " + table.row_values(i)[3]
            if id == 3:
                print "gf" + str(count) + " = lambda x,y,z: " + (table.row_values(i)[4])
                print "rf" + str(count) + " = lambda x,y,z: " + table.row_values(i)[3]
            if id == 4:
                print "gf" + str(count) + " = lambda x,y,z,p: " + (table.row_values(i)[4])
                print "rf" + str(count) + " = lambda x,y,z,p: " + table.row_values(i)[3]
            bound = ast.literal_eval(table.row_values(i)[2])
            rfname = table.row_values(i)[1]
            nrfl.append(rfname)
            ngfl.append(table.row_values(i)[1])
            count = count + 1
            bound_l.append([bound])
    print "input_domain = " + str(bound_l)
    print "rfl = [" + ', '.join(rfl) + "]"
    print "gfl = [" + ', '.join(gfl) + "]"
    print "nrfl_fname = " + str(nrfl)
    print "ngfl_fname = " + str(ngfl)
    sys.stdout = orig_stdout
    f.close()
for i in range(1,5):
    final_bench("../benchmarks/GSLbenchmarks.xls",i)