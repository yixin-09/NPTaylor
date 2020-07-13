import xlrd
import xlwt
import sys
import os
import ast

def read_file(file_name):
    fp = open(file_name)  # Open file on read mode
    lines = fp.read().split("\n")  # Create a list containing all lines
    fp.close()  # Close file
    # /home/yixin/PycharmProjects/AutoEFT/benchmarkss/gsl_src/gsl-2.1-repair/specfunc/airy.c
    index_lst = []
    for i in lines:
        var_lst = []
        if i != '':
            temp_i = i.split()
            for j in range(4, len(temp_i)):
                if (temp_i[j] == 'double') | (temp_i[j] == temp_i[0] + '(double'):
                    if temp_i[j + 1] == '*':
                        var_lst.append(temp_i[j + 2].strip(',)'))
                    else:
                        var_lst.append(temp_i[j + 1].strip(',)'))
            index_lst.append([temp_i[0], temp_i[2], temp_i[3], var_lst])
    return index_lst


def read_r_values(exname):
    data = xlrd.open_workbook(exname)
    fun_name_lst = []
    for i in range(0,4):
        table = data.sheets()[i]
        for i in range(0, table.nrows - 1):
            if (table.row_values(i + 1)[1]).strip() != 'gsl_sf_exp_e10_e':
                fun_name_lst.append((table.row_values(i + 1)[1]).strip()+"_e")
    return fun_name_lst
# ex_name_l = ['final_results_1v.xls','final_results_2v.xls','final_results_3v.xls','final_results_4v.xls']
# fun_name_lst = read_r_values("/home/yixin/PycharmProjects/AutoEFT/benchmarkss/1v_benchmarks.xls")


# index_lst = read_file('funs_index.txt')
# new_name_lst = []
# # read all names of benchmarks
# for j in fun_name_lst:
#     name = j
#     for i in index_lst:
#         if name == i[0]:
#             print i
#             new_name_lst.append(i)

def funs_to_w2xls(new_name_lst,name):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "functions")
    sheet.write(0, 1, "line_nums")
    sheet.write(0, 2, "file_names")
    sheet.write(0, 3, "varibles")
    n = 1
    for t in new_name_lst:
        for k in range(0, len(t)):
            sheet.write(n, k, repr(t[k]))
        n = n + 1
    book.save(name + ".xls")

# funs_to_w2xls(new_name_lst,"fun_index")
def search_line_num4f(fun_name,exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    file_name = ''
    for i in range(0, table.nrows-1):
        temp_str = str(table.row_values(i + 1)[0]).strip('\'')
        if temp_str == fun_name:
            file_name = table.row_values(i + 1)[2].strip('\'')
    ori_file_name = file_name
    pwd1 = os.getcwd()
    os.chdir("..")
    pwd = os.getcwd()
    file_name = pwd+'/benchmarks/gsl_src/gsl-2.1-repair/specfunc/' + file_name
    os.chdir(pwd1)
    os.system('ctags -x --c-kinds=f %s > fun_index.txt' % (file_name))
    index_lst = read_file('fun_index.txt')
    line_num = 0
    for i in index_lst:
        if fun_name == i[0]:
                line_num = int(i[1])
    fp = open(file_name)  # Open file on read mode
    lines = fp.read().split("\n")  # Create a list containing all lines
    fp.close()
    insert_num = line_num + 1
    for i in range(line_num,len(lines)):
        if lines[i-1] == '{':
            insert_num = i+1
            break
    return ori_file_name,insert_num

def get_varible_name(fun_name,exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    var_name = ''
    for i in range(0, table.nrows-1):
        temp_str = str(table.row_values(i + 1)[0]).strip('\'')
        if temp_str == fun_name:
            var_name = ast.literal_eval(table.row_values(i+1)[3])
            # print var_name
    return var_name
# print search_line_num4f('gsl_sf_bessel_K1_e',"fun_index.xls")
# print get_varible_name('gsl_sf_bessel_K1_e',"fun_index.xls")

# for j in fun_name_lst:
#     name = j
#     if name not in new_name_lst:
#         print name
#
# print len(new_name_lst)
# print len(fun_name_lst)
