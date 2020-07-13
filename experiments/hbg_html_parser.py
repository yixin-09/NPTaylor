from BeautifulSoup import BeautifulSoup

f1 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl1v/report.html")  # simplified for the example (no urllib)
f2 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl2v/report.html")  # simplified for the example (no urllib)
f3 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl3v/report.html")  # simplified for the example (no urllib)
f4 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl4v/report.html")  # simplified for the example (no urllib)
def extract_html_res(f,nvar):
    soup = BeautifulSoup(f)
    f.close()
    g_a = soup.findAll("li")  # the elements from inside the div a element
    # print g_a
    alst = []  # the future result list
    for x in g_a:
        # print x
        # print x.get('title')
        alst.append(x.get('title'))
    res_lst = []
    for i in alst:
        k = i.split()
        temp_lst = []
        count = 0
        for j in k:
            if count == 0:
                temp_c = j.split('_')
                if (len(temp_c[-1])>1)&(nvar==1):
                    temp_lst.append(1)
                    name_fun = j
                else:
                    temp_lst.append(int(temp_c[-1]))
                    name_fun = temp_c[0]
                    for strc in temp_c[1:-1]:
                        name_fun = name_fun + '_' + strc
                temp_lst.append(name_fun)
            count = count + 1
            temp_st = j.strip('(')
            temp_st = temp_st.strip(')')
            temp_lst.append(temp_st)
        res_lst.append(temp_lst)
    # res_lst.sort(key = lambda x: (x[1], x[2]))
    res_lst = sorted(res_lst,key = lambda x: (x[1], x[2]))
    return res_lst

# extract_html_res(f1,1)