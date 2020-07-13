import sys
import basic_func as bf
import os
import math
from fun_index import search_line_num4f
from fun_index import get_varible_name



def exists_in_file(filename,s):
    exists = 0
    fo = open(filename)
    line = fo.readline()
    # Loop until EOF
    while line != '':
        # Search for string in line
        index = line.find(s)
        if (index != -1):
            exists = 1
            break
        # Read next line
        line = fo.readline()
    # Close the files
    fo.close()
    return exists

def line_in_file(filename,s):
    fo = open(filename)
    line = fo.readline()
    # Initialize counter for line number
    line_no = 1
    exists = 0
    # Loop until EOF
    while line != '':
        # Search for string in line
        index = line.find(s)
        if (index != -1):
            insert_line = line_no + 1
            break
        # Read next line
        line = fo.readline()
        # Increment line counter
        line_no += 1
    # Close the files
    fo.close()
    return insert_line

def patch_generate(ori_bound, name, ori_filename):
    name = name.strip()
    fun_name = name + "_e"
    pwd = os.getcwd()
    print fun_name
    x_l = get_varible_name(fun_name,pwd+"/fun_index.xls")
    x = x_l[0]
    print "Generate patch"
    ori_filename = ori_filename+"/"
    orig_stdout = sys.stdout
    filename = "../benchmarks/GSL_function/" + name + '_patch.txt'
    f = open(filename, 'w')
    sys.stdout = f
    print "#include \"patch_of_" + name + ".c\""
    print "if(("+x+"<=" + repr(ori_bound[1]) + ")&&("+x+">=" + repr(ori_bound[0]) + ")){"
    print " result->val = accuracy_improve_patch_of_" + name + "("+x+");"
    print " result->err = GSL_DBL_EPSILON * fabs(result->val);"
    print " return GSL_SUCCESS;"
    print "}"
    sys.stdout = orig_stdout
    f.close()
    f = open(filename, "r")
    contents = f.readlines()
    f.close()
    insert_fname,insert_line = search_line_num4f(fun_name, pwd+"/fun_index.xls")
    patch_name = name+"_patch.c"
    cp_cmd = "cp -f "+ "../benchmarks/GSL_function/specfunc4patch/" + insert_fname+" "+ori_filename+"patch/" + patch_name
    print pwd
    print cp_cmd
    os.system(cp_cmd)
    f = open(ori_filename+"patch/" +patch_name, "r")
    new_contents = f.readlines()
    f.close()
    for j in range(1,len(contents)):
        new_contents.insert(insert_line," "+contents[j])
        insert_line = insert_line + 1
    new_contents.insert(25, contents[0])
    f = open(ori_filename+"patch/" + patch_name, "w")
    new_contents = "".join(new_contents)
    f.write(new_contents)
    f.close()
    gen_path_cmd = "diff -Naur "+ "../benchmarks/GSL_function/specfunc4patch/" + insert_fname+" "+ori_filename+"patch/" +patch_name + "> "+ori_filename+"patch/" +"patch_of_" + name
    os.system(gen_path_cmd)
    cp_back = "cp -f "+ ori_filename+"patch/" +patch_name + " " + "../benchmarks/GSL_function/specfunc4patch/" + insert_fname
    os.system(cp_back)
    rm_code = "rm " + ori_filename+"patch/" +patch_name
    os.system(rm_code)
    osname = os.path.dirname(os.getcwd())
    apply_patch = "patch " + osname+"/benchmarks/gsl_src/gsl-2.1-repair/specfunc/"+insert_fname + " "+"patch_of_" + name+"\n"
    f = open(ori_filename + "patch/patch_cmd.sh", "a")
    f.write(apply_patch)
    f.close()

def taylor_exp(rf,x,der_l,x0,n):
    temp = float(der_l[0])
    for i in range(1,n):
        temp = temp + math.pow((x-x0),i)*der_l[i]/math.factorial(i)
    return temp

def convertToC_taylor(temp_n,der_l,name,filename,inp):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    idx = 0
    print 'int factorial(int i){'
    print '   int fact=1;'
    print '   for(int k=1;k<=i;k++){'
    print '	fact=fact*k;'
    print '	}'
    print '    return fact;}'
    print 'static double der_l' +'['+ str(temp_n) + '] = {'
    for i in der_l:
        print "%.18e," % i
    print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x)"
    print "{"
    print " double x_0 = " + repr(inp)+";"
    print " double temp = der_l[0];"
    print " double idx = " + str(temp_n)+";"
    print " for(int i=1;i<idx;i++){"
    print " 	temp = temp+pow(x-x_0,i)*(der_l[i])/factorial(i);"
    print " }"
    print " return temp;"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def convertToC_taylorOnL(glob_l,n,name,idx,filename,bound,temp_ploy_fit):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    x_0 = bound[0]
    ulp_x = bf.getulp(glob_l[0][0][0])
    temp_n = 0.0
    temp_n_max = 0.0
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    if idx == 0:
        print 'int factorial(int i){'
        print '   int fact=1;'
        print '   for(int k=1;k<=i;k++){'
        print '	        fact=fact*k;'
        print '	  }'
        print '    return fact;}'
    print 'static double array_x_'+name+'_'+str(idx)+'['+ str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[0][0]
    print '};'
    print 'static double array_y_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[0][1]
    print '};'
    print 'static double array_e_y_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][1]
    print '};'
    print 'static double array_detla_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[3]
    print '};'
    print 'static double array_idx_'+name+'_'+str(idx)+'[' + str(len_glob+1) + '] = {'
    print "%.18e," % temp_n
    for i in glob_l:
        print "%.18e," % (i[2]+temp_n)
        temp_n = i[2]+temp_n
    print '};'
    print 'static double array_midpoint_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[6]
    print '};'
    print 'static double array_Taylor_' +name+'_'+ str(idx) + '[' + str(len_glob) + ']'+'[' + str(5) + '] = {'
    for i in glob_l[0:-1]:
        print '{'
        print str(len(i[4]))+','
        if i[4]!=[]:
            for j in i[4]:
                print  "%.18e," % j
        print '},'
    print '{'
    print str(len(glob_l[-1][4]))+','
    if glob_l[-1] != []:
        for j in glob_l[-1][4]:
            print  "%.18e," % j
    print '}'
    print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x)"
    print "{"
    print " long int n = "+str(n)+";"
    print " int len_glob = "+str(len_glob)+";"
    print " double ulp_x = " + repr(ulp_x)+";"
    print " double x_0 = " + repr(x_0)+";"
    print " double compen = 0.0;"
    print " double n_x = ((x-x_0)/ulp_x);"
    if temp_ploy_fit == '':
        # print " int idx = floor(n_x*len_glob/n);"
        print " int idx = floor(len_glob/2);"
    else:
        print temp_ploy_fit
        print " if(idx>=len_glob){"
        print "         idx = len_glob-1;"
        print " }"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((n_x>array_idx_"+name+'_'+str(idx)+"[idx])&&(n_x<array_idx_"+name+'_'+str(idx)+"[idx+1])){"
    print "         double compen = array_Taylor_" +name+'_'+ str(idx)+'[idx][1];'
    print "         double tayn = array_Taylor_" +name+'_'+ str(idx)+'[idx][0];'
    print "         double midpoint = array_midpoint_" + name + '_' + str(idx) + '[idx];'
    print "         for(int i=2;i<tayn+1;i++){"
    print " 	    compen = compen+pow(x-midpoint,i-1)*(array_Taylor_" +name+'_'+ str(idx)+"[idx][i])/factorial(i-1);"
    print "         }"
    print "         return (x-array_x_"+name+'_'+str(idx)+"[idx])/ulp_x*array_detla_"+name+'_'+str(idx)+"[idx]+array_y_"+name+'_'+str(idx)+"[idx]+compen;"
    print "     }"
    print "     else if(n_x<array_idx_"+name+'_'+str(idx)+"[idx]){"
    print "         idx = idx - 1;"
    print "     }"
    print "     else if(n_x>array_idx_"+name+'_'+str(idx)+"[idx+1]){"
    print "         idx = idx + 1;"
    print "     }"
    print "     else if(x==array_x_"+name+'_'+str(idx)+"[idx]){"
    print "         return array_y_"+name+'_'+str(idx)+"[idx];"
    print "     }"
    print "     else{"
    print "         return array_e_y_"+name+'_'+str(idx)+"[idx];"
    print "     }"
    print " }"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def convertToC_taylor1v(glob_l,name,idx,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    max_len= 0
    temp_len= 0
    for i in glob_l:
        temp_len = len(i[0])
        if temp_len>max_len:
            max_len = temp_len
    idx_lst = []
    for i in glob_l:
        idx_lst.append(i[3][0])
    idx_lst.append(glob_l[-1][3][1])
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_idx_'+name+'_'+str(idx)+'['+ str(len(idx_lst)) + '] = {'
    for i in idx_lst:
        print "%.18e," % i
    print '};'
    print 'static double array_cof_float_'+name+'_'+str(idx)+'[' + str(len_glob) + ']'+'[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][1]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']'+'[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][0]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_point_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[-1]
    print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x)"
    print "{"
    print " int len_glob = "+str(len_glob)+";"
    print " int idx = floor(len_glob/2);"
    print " int dw_idx = 0;"
    print " int up_idx = len_glob;"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((x>=array_idx_"+name+'_'+str(idx)+"[idx])&&(x<=array_idx_"+name+'_'+str(idx)+"[idx+1])){"
    print "         double point = array_point_" +name+'_'+ str(idx)+'[idx];'
    print "         double res = 0.0;"
    # print "         int length = sizeof(array_cof_float_" +name+'_'+ str(idx)+"[idx])/sizeof(double);"
    print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
    print "         eft_tay1v(array_cof_float_" +name+'_'+ str(idx)+'[idx],array_cof_err_' +name+'_'+ str(idx)+"[idx],point,x,&res,length);"
    print "         return res;"
    print "     }"
    # print "     else if(x<array_idx_"+name+'_'+str(idx)+"[idx]){"
    # print "         idx = idx - 1;"
    # print "     }"
    # print "     else if(x>array_idx_"+name+'_'+str(idx)+"[idx+1]){"
    # print "         idx = idx + 1;"
    # print "     }"
    print "     else if(x<array_idx_" + name + '_' + str(idx) + "[idx]){"
    print "         up_idx = idx;"
    print "         idx = dw_idx + floor((idx-dw_idx)/2.0);"
    print "     }"
    print "     else if(x>array_idx_" + name + '_' + str(idx) + "[idx+1]){"
    print "         dw_idx = idx;"
    print "         idx = idx + floor((up_idx-idx)/2.0);"
    print "     }"
    print " }"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def convertToC_taylor2v_debug(glob_l,name,idx,filename,dr):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    len_glob = len(glob_l)
    max_len = 0
    temp_len = 0
    for i in glob_l:
        temp_len = len(i[1])
        if temp_len > max_len:
            max_len = temp_len
    f = open(filename + '/' + name + '.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_cof_float_' + name + '_' + str(idx) + '[' + str(len_glob) + ']' + '[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][1]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']' + '[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][0]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_pointx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][0]
    print '};'
    print 'static double array_pointy_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][1]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][-1]
    print '};'
    for i in range(2):
        print 'static double array_idx_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][0]
        print '};'
    for i in range(2):
        print 'static double array_idx1_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][1]
        print '};'
    print "int find_idx_gsl_"+name+"(double x,double y)"
    print "{"
    print " int len_glob = " + str(len_glob) + ";"
    print " int idx = 0;"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((x>=array_idx_0" + name + '_' + str(idx) + "[idx])&&(x<=array_idx1_0" + name + '_' + str(idx) + "[idx])){"
    print "         if((y>=array_idx_1" + name + '_' + str(idx) + "[idx])&&(y<=array_idx1_1" + name + '_' + str(idx) + "[idx])){"
    print "                 return idx+1;"
    print "         }"
    print "         else{"
    print "             idx = idx+1;"
    print "         }"
    print "     }"
    print "     else{"
    print "             idx = idx+1;"
    print "     }"
    print " }"
    print " return 0;"
    print "}"
    print "double accuracy_improve_patch_of_gsl_" + name + '_' + str(idx) + "(double x,double y,int id)"
    print "{"
    # print "         int len_glob = " + str(len_glob) + ";"
    print "         int idx = id-1;"
    print "         double pointx = array_pointx_" + name + '_' + str(idx) + '[idx];'
    print "         double pointy = array_pointy_" + name + '_' + str(idx) + '[idx];'
    print "         double res = 0.0;"
    print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
    print "         eft_tay2v(array_cof_float_" + name + '_' + str(idx) + '[idx],array_cof_err_' + name + '_' + str(idx) + "[idx],pointx,pointy,x,y,&res,length);"
    print "         return res;"
    print "}"
    sys.stdout = orig_stdout
    f.close()
def convertToC_taylor2v(glob_l,name,idx,filename,dr):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    max_len= 0
    temp_len= 0
    for i in glob_l:
        temp_len = len(i[1])
        if temp_len>max_len:
            max_len = temp_len
    idx_lst = []
    if dr == 0:
        for i in glob_l:
            idx_lst.append(i[3][0][0])
        idx_lst.append(glob_l[-1][3][0][1])
    else:
        for i in glob_l:
            idx_lst.append(i[3][1][0])
        idx_lst.append(glob_l[-1][3][1][1])
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_idx_'+name+'_'+str(idx)+'['+ str(len(idx_lst)) + '] = {'
    for i in idx_lst:
        print "%.18e," % i
    print '};'
    print 'static double array_cof_float_'+name+'_'+str(idx)+'[' + str(len_glob) + ']'+'[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][1]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']'+'[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][0]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_pointx_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][0]
    print '};'
    print 'static double array_pointy_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][1]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][-1]
    print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x,double y)"
    print "{"
    print " int len_glob = "+str(len_glob)+";"
    print " int idx = floor(len_glob/2);"
    print " int dw_idx = 0;"
    print " int up_idx = len_glob;"
    if dr == 0:
        print " while((idx>=0)&&(idx<len_glob)){"
        print "     if((x>=array_idx_"+name+'_'+str(idx)+"[idx])&&(x<=array_idx_"+name+'_'+str(idx)+"[idx+1])){"
        print "         double pointx = array_pointx_" +name+'_'+ str(idx)+'[idx];'
        print "         double pointy = array_pointy_" +name+'_'+ str(idx)+'[idx];'
        print "         double res = 0.0;"
        print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
        print "         eft_tay2v(array_cof_float_" +name+'_'+ str(idx)+'[idx],array_cof_err_' +name+'_'+ str(idx)+"[idx],pointx,pointy,x,y,&res,length);"
        print "         return res;"
        print "     }"
        print "     else if(x<array_idx_"+name+'_'+str(idx)+"[idx]){"
        print "         up_idx = idx;"
        print "         idx = dw_idx + floor((idx-dw_idx)/2.0);"
        print "     }"
        print "     else if(x>array_idx_"+name+'_'+str(idx)+"[idx+1]){"
        print "         dw_idx = idx;"
        print "         idx = idx + floor((up_idx-idx)/2.0);"
        print "     }"
        print " }"
        print "}"
    else:
        print " while((idx>=0)&&(idx<len_glob)){"
        print "     if((y>=array_idx_" + name + '_' + str(idx) + "[idx])&&(y<=array_idx_" + name + '_' + str(idx) + "[idx+1])){"
        print "         double pointx = array_pointx_" + name + '_' + str(idx) + '[idx];'
        print "         double pointy = array_pointy_" + name + '_' + str(idx) + '[idx];'
        print "         double res = 0.0;"
        print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
        print "         eft_tay2v(array_cof_float_" + name + '_' + str(idx) + '[idx],array_cof_err_' + name + '_' + str(idx) + "[idx],pointx,pointy,x,y,&res,length);"
        print "         return res;"
        print "     }"
        print "     else if(y<array_idx_" + name + '_' + str(idx) + "[idx]){"
        print "         up_idx = idx;"
        print "         idx = dw_idx + floor((idx-dw_idx)/2.0);"
        print "     }"
        print "     else if(y>array_idx_" + name + '_' + str(idx) + "[idx+1]){"
        print "         dw_idx = idx;"
        print "         idx = idx + floor((up_idx-idx)/2.0);"
        print "     }"
        print " }"
        print "}"
    sys.stdout = orig_stdout
    f.close()


#[cof_err, cof_float, point, i]
def convertToC_taylor3v(glob_l,name,idx,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    max_len= 0
    max_len2= 0
    temp_len= 0
    temp_len2= 0
    for i in glob_l:
        temp_len = len(i[1])-1
        temp_len2 = len(i[1][0])
        if temp_len>max_len:
            max_len = temp_len
        if temp_len2>max_len2:
            max_len2 = temp_len2
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    print glob_l[-1][1]
    print glob_l[0]
    print glob_l[0][1][-2]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_cof_float_'+name+'_'+str(idx)+'[' + str(len_glob) + ']'+'[' + str(max_len) + ']'+'[' + str(max_len2) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1][0:-2]:
            print "{"
            for k in j[:-1]:
                print " %.18e," % k
            print " %.18e" % j[-1]
            print "},"
        for j in i[1][-2]:
            print "{"
            # for k in j[:-1]:
            #     print " %.18e," % k
            print " %.18e" % j
            print "}"
        print "},"
    print "{"
    for j in glob_l[-1][1][0:-2]:
        print "{"
        for k in j[:-1]:
            print " %.18e," % k
        print " %.18e" % j[-1]
        print "},"
    for j in glob_l[-1][1][-2]:
        print "{"
        # for k in j[:-1]:
        #     print " %.18e," % k
        print " %.18e" % j
        print "}"
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']'+'[' + str(max_len) + ']'+'[' + str(max_len2) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0][0:-1]:
            print "{"
            for k in j[:-1]:
                print " %.18e," % k
            print " %.18e," % j[-1]
            print "},"
        for j in i[0][-1]:
            print "{"
            # for k in j[:-1]:
            #     print " %.18e," % k
            print " %.18e" % j
            print "}"
        print "},"
    print "{"
    for j in glob_l[-1][0][0:-1]:
        print "{"
        for k in j[:-1]:
            print " %.18e," % k
        print " %.18e" % j[-1]
        print "},"
    for j in glob_l[-1][0][-1]:
        print "{"
        # for k in j[:-1]:
        #     print " %.18e," % k
        print " %.18e" % j
        print "}"
    print "}"
    print '};'
    print 'static double array_pointx_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][0]
    print '};'
    print 'static double array_pointy_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][1]
    print '};'
    print 'static double array_pointz_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][2]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][-1]
    print '};'
    for i in range(3):
        print 'static double array_idx_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][0]
        print '};'
    for i in range(3):
        print 'static double array_idx1_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][1]
        print '};'
    print "double accuracy_improve_patch_of_gsl_" + name + '_' + str(idx) + "(double x,double y,double z)"
    print "{"
    print " int len_glob = " + str(len_glob) + ";"
    print " int idx = 0;"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((x>=array_idx_0" + name + '_' + str(idx) + "[idx])&&(x<=array_idx1_0" + name + '_' + str(idx) + "[idx])){"
    print "         if((y>=array_idx_1" + name + '_' + str(idx) + "[idx])&&(y<=array_idx1_1" + name + '_' + str(idx) + "[idx])){"
    print "             if((z>=array_idx_2" + name + '_' + str(idx) + "[idx])&&(z<=array_idx1_2" + name + '_' + str(idx) + "[idx])){"
    print "                 break;"
    print "             }"
    print "             else{"
    print "                 idx = idx+1;"
    print "             }"
    print "         }"
    print "         else{"
    print "             idx = idx+1;"
    print "         }"
    print "     }"
    print "     else{"
    print "             idx = idx+1;"
    print "     }"
    print " }"
    print "         double pointx = array_pointx_" + name + '_' + str(idx) + '[idx];'
    print "         double pointy = array_pointy_" + name + '_' + str(idx) + '[idx];'
    print "         double pointz = array_pointz_" + name + '_' + str(idx) + '[idx];'
    print "         double res = 0.0;"
    print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
    print "         eft_tay3v(array_cof_float_" + name + '_' + str(idx) + '[idx],array_cof_err_' + name + '_' + str(idx) + "[idx],pointx,pointy,pointz,x,y,z,&res,length);"
    print "         return res;"
    print "}"
    sys.stdout = orig_stdout
    f.close()


def convertToC_taylor3v_debug2(glob_l,name,idx,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    max_len= 0
    max_len2= 0
    temp_len= 0
    temp_len2= 0
    for i in glob_l:
        temp_len = len(i[1])-1
        temp_len2 = len(i[1][0])
        if temp_len>max_len:
            max_len = temp_len
        if temp_len2>max_len2:
            max_len2 = temp_len2
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    print glob_l[-1][1]
    print glob_l[0]
    print glob_l[0][1][-2]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_cof_float_'+name+'_'+str(idx)+'[' + str(len_glob) + ']'+'[' + str(max_len) + ']'+'[' + str(max_len2) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1][0:-2]:
            print "{"
            for k in j[:-1]:
                print " %.18e," % k
            print " %.18e" % j[-1]
            print "},"
        for j in i[1][-2]:
            print "{"
            # for k in j[:-1]:
            #     print " %.18e," % k
            print " %.18e" % j
            print "}"
        print "},"
    print "{"
    for j in glob_l[-1][1][0:-2]:
        print "{"
        for k in j[:-1]:
            print " %.18e," % k
        print " %.18e" % j[-1]
        print "},"
    for j in glob_l[-1][1][-2]:
        print "{"
        # for k in j[:-1]:
        #     print " %.18e," % k
        print " %.18e" % j
        print "}"
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']'+'[' + str(max_len) + ']'+'[' + str(max_len2) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0][0:-1]:
            print "{"
            for k in j[:-1]:
                print " %.18e," % k
            print " %.18e," % j[-1]
            print "},"
        for j in i[0][-1]:
            print "{"
            # for k in j[:-1]:
            #     print " %.18e," % k
            print " %.18e" % j
            print "}"
        print "},"
    print "{"
    for j in glob_l[-1][0][0:-1]:
        print "{"
        for k in j[:-1]:
            print " %.18e," % k
        print " %.18e" % j[-1]
        print "},"
    for j in glob_l[-1][0][-1]:
        print "{"
        # for k in j[:-1]:
        #     print " %.18e," % k
        print " %.18e" % j
        print "}"
    print "}"
    print '};'
    print 'static double array_pointx_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][0]
    print '};'
    print 'static double array_pointy_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][1]
    print '};'
    print 'static double array_pointz_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][2]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][-1]
    print '};'
    for i in range(3):
        print 'static double array_idx_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][0]
        print '};'
    for i in range(3):
        print 'static double array_idx1_' + str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][1]
        print '};'
    print "double accuracy_improve_patch_of_gsl_" + name + '_' + str(idx) + "(double x,double y,double z)"
    print "{"
    print " int len_glob = " + str(len_glob) + ";"
    print " int idx = floor(len_glob/2);"
    print " int dw_idx = 0;"
    print " int up_idx = len_glob;"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((x>=array_idx_0" + name + '_' + str(idx) + "[idx])&&(x<=array_idx1_0" + name + '_' + str(idx) + "[idx])){"
    print "         if((y>=array_idx_1" + name + '_' + str(idx) + "[idx])&&(y<=array_idx1_1" + name + '_' + str(idx) + "[idx])){"
    print "             if((z>=array_idx_2" + name + '_' + str(idx) + "[idx])&&(z<=array_idx1_2" + name + '_' + str(idx) + "[idx])){"
    print "                 break;"
    print "             }"
    print "             else if(z<array_idx_2" + name + '_' + str(idx) + "[idx]){"
    print "                 idx = idx-1;"
    print "             }"
    print "             else if(z>array_idx1_2" + name + '_' + str(idx) + "[idx]){"
    print "                 idx = idx+1;"
    print "             }"
    print "         }"
    print "         else if(y<array_idx_1" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx-1;"
    print "         }"
    print "         else if(y>array_idx1_1" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx+1;"
    print "         }"
    print "     }"
    print "     else if(x<array_idx_0" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx-1;"
    print "     }"
    print "     else if(x>array_idx1_0" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx+1;"
    print "     }"
    print " }"
    print "         double pointx = array_pointx_" + name + '_' + str(idx) + '[idx];'
    print "         double pointy = array_pointy_" + name + '_' + str(idx) + '[idx];'
    print "         double pointz = array_pointz_" + name + '_' + str(idx) + '[idx];'
    print "         double res = 0.0;"
    print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
    print "         eft_tay3v(array_cof_float_" + name + '_' + str(idx) + '[idx],array_cof_err_' + name + '_' + str(idx) + "[idx],pointx,pointy,pointz,x,y,z,&res,length);"
    print "         return res;"
    print "}"
    sys.stdout = orig_stdout
    f.close()


def convertToC_taylor3v_debug(glob_l,name,idx,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    max_len= 0
    max_len2= 0
    temp_len= 0
    for i in glob_l:
        temp_len = len(i[1])
        if temp_len>max_len:
            max_len = temp_len
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    print '#include "eft_patch.h"'
    print 'static double array_cof_float_'+name+'_'+str(idx)+'[' + str(len_glob) + ']'+'[' + str(max_len) + ']= {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[1]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][1]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_cof_err_' + name + '_' + str(idx) + '[' + str(len_glob) + ']'+'[' + str(max_len) + '] = {'
    for i in glob_l[0:-1]:
        print "{"
        for j in i[0]:
            print "%.18e," % j
        print "},"
    print "{"
    for j in glob_l[-1][0]:
        print "%.18e," % j
    print "}"
    print '};'
    print 'static double array_pointx_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][0]
    print '};'
    print 'static double array_pointy_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][1]
    print '};'
    print 'static double array_pointz_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[2][2]
    print '};'
    print 'static double array_cofidx_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1][-1]
    print '};'
    for i in range(3):
        print 'static double array_idx_'+str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][0]
        print '};'
    for i in range(3):
        print 'static double array_idx1_'+str(i) + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
        for j in glob_l:
            print "%.18e," % j[3][i][1]
        print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x,double y,double z)"
    print "{"
    print " int len_glob = "+str(len_glob)+";"
    print " int idx = floor(len_glob/2);"
    print " int dw_idx = 0;"
    print " int up_idx = len_glob;"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((x>=array_idx_0"+name+'_'+str(idx)+"[idx])&&(x<=array_idx1_0"+name+'_'+str(idx)+"[idx])){"
    print "         if((y>=array_idx_1" + name + '_' + str(idx) + "[idx])&&(y<=array_idx1_1" + name + '_' + str(idx) + "[idx])){"
    print "             if((z>=array_idx_2" + name + '_' + str(idx) + "[idx])&&(z<=array_idx1_2" + name + '_' + str(idx) + "[idx])){"
    print "                 break;"
    print "             }"
    print "             else if(z<array_idx_2" + name + '_' + str(idx) + "[idx]){"
    print "                 idx = idx-1;"
    print "             }"
    print "             else if(z>array_idx1_2" + name + '_' + str(idx) + "[idx]){"
    print "                 idx = idx+1;"
    print "             }"
    print "         }"
    print "         else if(y<array_idx_1" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx-1;"
    print "         }"
    print "         else if(y>array_idx1_1" + name + '_' + str(idx) + "[idx]){"
    print "             idx = idx+1;"
    print "         }"
    print "     }"
    print "     else if(x<array_idx_0"+name+'_'+str(idx)+"[idx]){"
    print "             idx = idx-1;"
    print "     }"
    print "     else if(x>array_idx1_0"+name+'_'+str(idx)+"[idx]){"
    print "             idx = idx+1;"
    print "     }"
    print " }"
    print "         double pointx = array_pointx_" + name + '_' + str(idx) + '[idx];'
    print "         double pointy = array_pointy_" + name + '_' + str(idx) + '[idx];'
    print "         double pointz = array_pointz_" + name + '_' + str(idx) + '[idx];'
    print "         double res = 0.0;"
    print "         int length = (int)array_cofidx_" + name + '_' + str(idx) + "[idx];"
    print "         eft_tay3v(array_cof_float_" + name + '_' + str(idx) + '[idx],array_cof_err_' + name + '_' + str(idx) + "[idx],pointx,pointy,pointz,x,y,z,&res,length);"
    print "         return res;"
    print "}"
    sys.stdout = orig_stdout
    f.close()



def covertToC(glob_l,n,name,idx,filename,bound,temp_ploy_fit):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_'+name
    len_glob = len(glob_l)
    x_0 = bound[0]
    ulp_x = bf.getulp(glob_l[0][0][0])
    temp_n = 0.0
    temp_n_max = 0.0
    f = open(filename+'/'+name+'.c', 'a')
    name = name.split("gsl_")[1]
    sys.stdout = f
    print 'static double array_x_'+name+'_'+str(idx)+'['+ str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[0][0]
    print '};'
    print 'static double array_y_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[0][1]
    print '};'
    print 'static double array_e_y_' + name + '_' + str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[7][1]
    print '};'
    print 'static double array_detla_'+name+'_'+str(idx)+'[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[1]
    print '};'
    print 'static double array_idx_'+name+'_'+str(idx)+'[' + str(len_glob+1) + '] = {'
    print "%.18e," % temp_n
    for i in glob_l:
        print "%.18e," % (i[2]+temp_n)
        temp_n = i[2]+temp_n
    print '};'
    # print 'static double array_maxX_' + str(idx) + '[' + str(len_glob) + '] = {'
    # for i in glob_l:
    #     print "%.15e," % (i[3]+temp_n_max)
    #     temp_n_max = (i[2]+temp_n_max)
    # print '};'
    print 'static double array_maxE_' +name+'_'+ str(idx) + '[' + str(len_glob) + '] = {'
    for i in glob_l:
        print "%.18e," % i[4]
    print '};'
    print "double accuracy_improve_patch_of_gsl_"+name+'_'+str(idx)+"(double x)"
    print "{"
    print " long int n = "+str(n)+";"
    print " int len_glob = "+str(len_glob)+";"
    print " double ulp_x = " + repr(ulp_x)+";"
    print " double x_0 = " + repr(x_0)+";"
    print " double compen = 0.0;"
    print " double n_x = ((x-x_0)/ulp_x);"
    if temp_ploy_fit == '':
        # print " int idx = floor(n_x*len_glob/n);"
        print " int idx = floor(len_glob/2);"
    else:
        print temp_ploy_fit
        print " if(idx>=len_glob){"
        print "         idx = len_glob-1;"
        print " }"
    print " while((idx>=0)&&(idx<len_glob)){"
    print "     if((n_x>array_idx_"+name+'_'+str(idx)+"[idx])&&(n_x<array_idx_"+name+'_'+str(idx)+"[idx+1])){"
    print "         compen = ulp_x*ulp_x * (n_x-array_idx_"+name+'_'+str(idx)+"[idx+1])*(n_x-array_idx_"+name+'_'+str(idx)+"[idx])*array_maxE_"+name+'_'+str(idx)+"[idx];"
    print "         return (x-array_x_"+name+'_'+str(idx)+"[idx])/ulp_x*array_detla_"+name+'_'+str(idx)+"[idx]+array_y_"+name+'_'+str(idx)+"[idx]+compen;"
    print "     }"
    print "     else if(n_x<array_idx_"+name+'_'+str(idx)+"[idx]){"
    print "         idx = idx - 1;"
    print "     }"
    print "     else if(n_x>array_idx_"+name+'_'+str(idx)+"[idx+1]){"
    print "         idx = idx + 1;"
    print "     }"
    print "     else if(x==array_x_"+name+'_'+str(idx)+"[idx]){"
    print "         return array_y_"+name+'_'+str(idx)+"[idx];"
    print "     }"
    print "     else{"
    print "         return array_e_y_"+name+'_'+str(idx)+"[idx];"
    print "     }"
    print " }"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def combCovertToC_taylor(bound,name,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    f = open(filename +'/'+name + '.c', 'a')
    idx = 0
    sys.stdout = f
    print "double accuracy_improve_" + name +"(double x)"
    print "{"
    print "if(x<="+repr(bound[1])+"){"
    print " return accuracy_improve_" + name + '_' + str(idx) + "(x);"
    print "}"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def combCovertToC(bound_l,name,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    f = open(filename +'/'+name + '.c', 'a')
    idx = 0
    print bound_l
    sys.stdout = f
    print "double accuracy_improve_" + name +"(double x)"
    print "{"
    for i in bound_l:
        print "if(x<="+repr(i[1])+"){"
        print " return accuracy_improve_" + name + '_' + str(idx) + "(x);"
        print "}"
        idx = idx+1
    print "}"
    sys.stdout = orig_stdout
    f.close()


def combCovertToC_tay1v(bound,name,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    f = open(filename +'/'+name + '.c', 'a')
    idx = 0
    sys.stdout = f
    print "double accuracy_improve_" + name +"(double x)"
    print "{"
    print "if(x<="+repr(bound[1])+"){"
    print " return accuracy_improve_" + name + '_' + str(idx) + "(x);"
    print "}"
    print "}"
    sys.stdout = orig_stdout
    f.close()

# def combCovertToC_tay2v(bound,name,filename):
#     print "Cover To C code"
#     orig_stdout = sys.stdout
#     name = 'patch_of_' + name
#     f = open(filename +'/'+name + '.c', 'a')
#     idx = 0
#     sys.stdout = f
#     print "double accuracy_improve_" + name +"(double x,double y,int id)"
#     print "{"
#     print "if((x<="+repr(bound[0][1])+")&&(y<="+repr(bound[1][1])+")){"
#     print " return accuracy_improve_" + name + '_' + str(idx) + "(x,y,id);"
#     print "}"
#     print "}"
#     sys.stdout = orig_stdout
#     f.close()

def combCovertToC_tay2v(bound,name,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    f = open(filename +'/'+name + '.c', 'a')
    idx = 0
    sys.stdout = f
    print "double accuracy_improve_" + name +"(double x,double y)"
    print "{"
    print "if((x<="+repr(bound[0][1])+")&&(y<="+repr(bound[1][1])+")){"
    print " return accuracy_improve_" + name + '_' + str(idx) + "(x,y);"
    print "}"
    print "}"
    sys.stdout = orig_stdout
    f.close()

def combCovertToC_tay3v(bound,name,filename):
    print "Cover To C code"
    orig_stdout = sys.stdout
    name = 'patch_of_' + name
    f = open(filename +'/'+name + '.c', 'a')
    idx = 0
    sys.stdout = f
    print "double accuracy_improve_" + name +"(double x,double y,double z)"
    print "{"
    print "if((x<="+repr(bound[0][1])+")&&(y<="+repr(bound[1][1])+")&&(z<="+repr(bound[2][1])+")){"
    print " return accuracy_improve_" + name + '_' + str(idx) + "(x,y,z);"
    print "}"
    print "}"
    sys.stdout = orig_stdout
    f.close()