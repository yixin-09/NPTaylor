# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge
import xlrd
import math
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
#
# font_name = 'SIMHEI'
# plt.rcParams['font.family'] = font_name #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def read_r_values(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    r_val = []
    inp_val = []
    for i in range(0, table.nrows - 1):
        temp_err = np.log2(np.max([float(table.row_values(i + 1)[2]),1.0]))
        r_val.append(temp_err)
        if (temp_err<40)&(temp_err>30):
            print str(table.row_values(i + 1)[0])
        # temp_inp = ast.literal_eval(table.row_values(i + 1)[3])
        # inp_val.append(bv2.gfl[i](*temp_inp))
        # print table.row_values(i + 1)[0]
        # print table.row_values(i + 1)[2]
        # res_gf = bv2.gfl[i](*temp_inp)
        # res_rf = bv2.rfl[i](*temp_inp)
        # print temp_inp
        # print res_gf
        # print res_rf
        # print 1.0/bf.mfitness_fun1(bv2.rfl[i],bv2.gfl[i],temp_inp)
    return r_val
ex_name_l = ['final_results_1v.xls','final_results_2v.xls','final_results_3v.xls','final_results_4v.xls']

def read_name_list(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    name_lst = []
    for i in range(0, table.nrows - 1):
        temp_str = str(table.row_values(i + 1)[0])
        # temp_str = temp_str.strip("gsl/_sf/_")
        temp_str = temp_str.strip()
        temp_str = temp_str[7:]
        temp_str = temp_str.replace("_","\_")
        # temp_str = "{"+temp_str+"}"
        # temp_str = temp_str.encode()
        name_lst.append(temp_str)
    return name_lst

# ex_name_l = ['final_results_2v.xls']
# gsl_sf_bessel_y2
# gsl_sf_gegenpoly_3, gsl_sf_hyperg_0F1, gsl_sf_laguerre_1, gsl_sf_laguerre_2, gsl_sf_laguerre_3, gsl_sf_conicalP_0,gsl_sf_conicalP_1
# gsl_sf_bessel_Inu,gsl_sf_erf_Q,gsl_sf_hazard,gsl_sf_exp_e10_e,gsl_sf_fermi_dirac_1,gsl_sf_fermi_dirac_2,gsl_sf_lambert_Wm1,gsl_sf_psi_1
r_l = []
for i in ex_name_l:
    temp_r = read_r_values(i)
    r_l = r_l+temp_r
name_lst = []
for i in ex_name_l:
    temp_name = read_name_list(i)
    name_lst = name_lst+temp_name
name_lst2=[]
for i in range(len(name_lst)):
    temp_name = 'P'+str(i+1)+'-'+name_lst[i]
    print temp_name
    name_lst2.append(temp_name)
name_lst = list(name_lst2)
print r_l
print len(r_l)
print max(r_l)
print min(r_l)
# r = np.arange(0, 2, 0.01)
n_rl = len(r_l)
theta = np.deg2rad(np.arange(0, 350, 350.0/n_rl))
r = [i/2.0 for i in r_l]
fig = plt.figure(figsize=(14, 12))
ax = plt.subplot(111, projection='polar')
# ax.scatter(theta, r, cmap='hsv', alpha=0.75)
ax.set_rmax(34)
ax.set_ylim(-1,34)
ax.margins(x=0,y=0)
rt = np.arange(0,34,4)
# print len(rt)
ax.set_yticks(rt)  # less radial ticks
# ax.tick_params(axis='both',which='both',color='b')
# ax.set_yticklabels(['64','56','48','40','32','24','16','8','0'])

ax.set_xticks([])
# ax.set_yticks([])
ax.set_xticklabels([""])
ax.set_yticklabels([""])
tick = [0,ax.get_rmax()]
print tick
count = 1
len_lst = []
# for j in name_lst:
#     text = ax.text(1,2,'%s' % j,{'ha': 'center', 'va': 'center'},fontsize = 15)
#     rs = fig.canvas.get_renderer()
#     bb = text.get_window_extent(renderer=rs)
#     len_lst.append(bb.width)
#     text.remove()

for t,j in zip(np.deg2rad(np.arange(0, 350, 350.0/n_rl)),name_lst):
# for t in name_lst:
#     k = len(j)
    ax.plot([t,t], tick, lw=0.72, color="lightgrey",zorder=2)
    ax.plot([t,t], [ax.get_rmax()-0.5,ax.get_rmax()], lw=0.72, color="black",zorder=2)
    # if math.fmod(count,20)==0:
    ax.text(t, ax.get_rmax()+0.3, '%s' % j,{'ha': 'left', 'va': 'center'},
             # transform=trans_offset,
             # horizontalalignment='right',
             # verticalalignment='bottom',
             rotation=t*180/(math.pi),fontsize=12,rotation_mode="anchor")
    count = count + 1

# plt.arrow(theta[-1]+0.15, -1, 0, 35.6, lw=1, zorder=5,alpha=0.5, head_width=0,head_length=0.6)
ax.annotate("", xy=(theta[-1]+0.12, 34), xytext=(theta[-1]+0.12, -1), arrowprops = dict(arrowstyle="->"))
# yl = [r'\textbf{0}','8','16','24','32','40','48','56','64']
yl = ['0','8','16','24','32','40','48','56','64']
for i,j in zip(rt,yl):
    ax.annotate(j, xy=(theta[-1]+0.15, 34), xytext=(theta[-1]+0.15,i),fontsize=14)
ax.annotate('ErrBits', xy=(theta[-1]+0.06, 30), xytext=(theta[-1]+0.06,28),fontsize=16,rotation=(theta[-1]+0.12)*180/(math.pi),rotation_mode="anchor")
print "********"
print theta[-1]
# ax.plot(theta, r,'.',color='deepskyblue',linewidth=1,solid_capstyle="round")
for i,j in zip(theta,r):
    if j > 16:
        ax.plot(i, j, '.', color='r', markersize=14,zorder=2)
    else:
        ax.plot(i, j, '.', color='seagreen', markersize=9,zorder=2)

# circle1 = plt.Circle((0, 0), 13, color='seagreen',transform=ax.transData._b,fill=False,linewidth=3,zorder=3)
# ax.add_artist(circle1)
circle2 = plt.Circle((0, 0), 17, color='r',transform=ax.transData._b, fill=False,linewidth=3,clip_on=True,zorder=3)
ax.add_artist(circle2)
# ax.plot(theta, r,'.',color='r',markersize=14)
# ax.autoscale(enable=True)
# ax.axis('scaled')
fn_lst = [104,39,7,5]
angl_lst = []
temp_angl = 0
for i in fn_lst:
    temp_angl = temp_angl + (i)*350/(n_rl)
    angl_lst.append(temp_angl)
temp_angl = 0
k = 0
# color_lst = ['skyblue','azure','silver','lavender']
color_lst = ['firebrick','darkorange','yellow','green']
# color_lst = ['red','red','red','red']
count = 0
weg_lst =[]
for i in angl_lst:
    # ax.add_artist(Wedge((.0, .0), 35, temp_angl, i, transform=ax.transData._b,width=1,capstyle='round', fill=True, color=color_lst[k], alpha=0.5,zorder=4))
    if k ==0:
        # lb_name = u' '.join((str(k + 1), "输入变量")).encode('utf-8').strip()
        # lb_name = smart_str(str(k + 1) + u"输入变量")
        lb_name = str(k + 1) + "-input"
    else:
        # lb_name = u' '.join((str(k + 1), "输入变量")).encode('utf-8').strip()
        # lb_name = smart_str(str(k + 1) + u"输入变量")
        lb_name = str(k + 1) + "-input"
    print lb_name
    print type(lb_name)
    temp_weg = Wedge((.0, .0), 34.2, temp_angl, i, transform=ax.transData._b,width=1, fill=True, color=color_lst[k], alpha=0.5,zorder=2,label=lb_name)
    ax.add_artist(temp_weg)
    weg_lst.append(temp_weg)
    temp_angl = i
    k = k+1

# size = 0.4
# cmap = plt.get_cmap("tab20c")
# outer_colors = cmap(np.arange(3)*4)
# inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))
#
# ax.pie([10,12,13], radius=1, colors=outer_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))
#
# ax.pie(range(6), radius=1-size, colors=inner_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))
# for i in weg_lst:
#     ax.legend(handles=[temp_weg], loc=1)
ax.legend(handles=weg_lst,bbox_to_anchor=(0.0,0.05),prop={'size': 14})
# ax.set_yticklabels([r'\textbf{0}','8','16','24','32','40','48','56','64'],zorder=4,fontsize=14)
# ax.set_yticklabels(['0','16','32','48','64'])
# ax.set_rlabel_position(-12)  # get radial labels away from plotted line
# plt.savefig("graph/MaxErrDetectedGSL.pdf", format="pdf")
# ax.legend()
plt.show()