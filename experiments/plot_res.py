import numpy as np
import matplotlib.pyplot as plt
import xlrd
import matplotlib
import os
from matplotlib.patches import Wedge
import math
import xlwt
from scipy import stats
from hbg_html_parser import extract_html_res
from xlutils.copy import copy
# matplotlib.rcParams['text.usetex']=True
# matplotlib.rcParams['text.latex.unicode']=True
matplotlib.rcParams['text.usetex'] = True
def plot_time_overhead(z1):
    z1 = np.array(z1)
    z1 = np.sort(z1)
    y = np.arange(1, len(z1) + 1) / float(len(z1))
    return z1, y

def plot_cumulative_dis_time(b, a, name):
    ratio_b = []
    for i, j in zip(b, a):
        ratio_b.append(j / i)
    z1 = []
    # z2 = []
    # z3 = []
    # k = 0
    z1 = list(ratio_b)
    # for i in range(0, 20):
    #     if i != 14:
    #         z1.append(ratio_b[i + k])
    #         z2.append(ratio_b[i + k + 1])
    #         z3.append(ratio_b[i + k + 2])
    #     k = k + 2
    # print z2
    z1, y1 = plot_time_overhead(z1)
    # z2, y2 = plot_time_overhead(z2)
    # z3, y3 = plot_time_overhead(z3)
    plt.plot(z1, y1, 'b:', label=r'$L_\varepsilon$')
    # print np.interp(1.0, z1, y1)
    # print np.interp(1.0, z2, y2)
    # print np.interp(1.0, z3, y3)
    # plt.plot(z2, y2, 'g--', label=r'$M_\varepsilon$')
    # plt.plot(z3, y3, 'k-', label=r'$H_\varepsilon$')
    ylabel = []
    for i in np.arange(0, 105, 20):
        ylabel.append(str(i) + "%")
    plt.yticks(np.arange(0, 1.1, 0.2), ylabel)
    plt.legend(loc=4)
    plt.grid(True)
    # plt.ylim([0,1.1])
    plt.savefig(name + ".pdf", format="pdf")
    plt.show()
    # plt.close()
def process_cumulative_dis_time(b, a):
    ratio_b = []
    for i, j in zip(b, a):
        ratio_b.append(j / i)
    z1 = []
    # z2 = []
    # z3 = []
    # k = 0
    z1 = list(ratio_b)
    # for i in range(0, 20):
    #     if i != 14:
    #         z1.append(ratio_b[i + k])
    #         z2.append(ratio_b[i + k + 1])
    #         z3.append(ratio_b[i + k + 2])
    #     k = k + 2
    # print z2
    z1, y1 = plot_time_overhead(z1)
    return z1,y1
def add_arrow(line, position=None, direction='right', size=5, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.max()
    # find closest index
    start_ind = 0
    end_ind = 1
    hw = 0.5
    hl = 0.9
    lw = 0.5  # axis line width
    ohg = 0.3  # arrow overhang
    # ax.arrow(0, -0.03, 68, -0.03, fc='k', ec='k', lw=lw,
    #          head_width=hw, head_length=hl, overhang=ohg,
    #          length_includes_head=True, clip_on=False)arrowprops=dict(arrowstyle='simple',fc="0.5",color=color),
    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(headwidth=4, headlength=5, width=1.4, fc="0.5", color=color),
                       size=size
                       )
def draw_arrow_error(be, ae, th,id_lst, name):
    f = plt.figure(frameon=False, figsize=(6, 5))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    hw = 0.5
    hl = 0.9
    lw = 0.5  # axis line width
    ohg = 0.3  # arrow overhang
    ax.arrow(0, -0.03, 68, -0.03, fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg,
             length_includes_head=True, clip_on=False)
    for i in range(0, len(be)):
        s_y = len(be) - i
        plt.plot([0, 64 - be[i]], [s_y, s_y], c='black', linewidth=2, alpha=0.6)
        line2 = plt.plot([64 - be[i], 64 - ae[i]], [s_y, s_y], c='black', linewidth=2, alpha=0)[0]
        add_arrow(line2)
        plt.scatter(64 - th, s_y, c='r', s=35, marker='<')
        # plt.scatter(64 - np.log2(mean_error_l[i]), s_y, c='r', marker=(5, 1))
        plt.plot([64 - be[i], 64 - be[i]], [s_y - 0.2, s_y + 0.2], c='black', linewidth=2)

    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the top edge are off
        labelleft=False,
    )  # labels along the bottom edge are off
    # plt.annotate("Bits Correct for Maximum Error",
    #              xy=(12, 20), xycoords='data',
    #              xytext=(0, 3), textcoords='offset points', fontsize=12)
    # plt.annotate('Accuracy improving for ' + name2,
    #              xy=(12, 20), xycoords='data',
    #              xytext=(0, 3), textcoords='offset points', fontsize=12)
    x_l = []
    for j in range(len(id_lst)):
        plt.annotate(id_lst[j],
                     xy=(-3.2, j + 0.8), xycoords='data',
                     xytext=(-2.5, 0), textcoords='offset points', fontsize=9)
    for i in range(0, 65, 8):
        x_l.append(i)
    ax.set_xticks(x_l)
    x_l.reverse()
    ax.set_xticklabels(x_l)
    plt.ylabel("ID", fontsize=15)
    # plt.ylabel("Program ID", fontsize=15)
    # plt.xlabel("ErrBits", fontsize=15)
    plt.tight_layout()
    plt.savefig(name + ".pdf", format="pdf")
    plt.close()

def get_cumulative_dis_time(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    bbpf_time_l = []
    bwpf_time_l = []
    abpf_time_l = []
    awpf_time_l = []
    for i in range(0, table.nrows - 1):
        bbpf_time_l.append(float(table.row_values(i + 1)[11]))
        # bwpf_time_l.append(float(table.row_values(i + 1)[37]))
        abpf_time_l.append(float(table.row_values(i + 1)[16]))
        # awpf_time_l.append(float(table.row_values(i + 1)[47]))
    plot_cumulative_dis_time(bbpf_time_l, abpf_time_l, "timeOverheadb")
    # print np.sum(bbpf_time_l)
    # print np.sum(bwpf_time_l)
    # print np.sum(abpf_time_l)
    # print np.sum(awpf_time_l)
    # plot_cumulative_dis_time(bwpf_time_l, awpf_time_l, "timeOverheadW")
def plot_compare_timeoverhead(file_lst,name):
   be_res_list = []
   af_res_list = []
   print file_lst1
   for exname in file_lst:
      data = xlrd.open_workbook(exname)
      table = data.sheets()[0]
      bbpf_time_l = []
      abpf_time_l = []
      for i in range(0, table.nrows - 1):
         bbpf_time_l.append(float(table.row_values(i + 1)[11]))
         # bwpf_time_l.append(float(table.row_values(i + 1)[37]))
         abpf_time_l.append(float(table.row_values(i + 1)[16]))
         # awpf_time_l.append(float(table.row_values(i + 1)[47]))
      z1,y1 = process_cumulative_dis_time(bbpf_time_l, abpf_time_l)
      print stats.ttest_ind(bbpf_time_l, abpf_time_l)
      be_res_list.append(z1)
      af_res_list.append(y1)

   count = 0
   color_type = ['r--','k-']
   tool_name = ['AutoRNP','NPTaylor']
   # fig = plt.figure(figsize=(14, 12))
   ax = plt.subplot(111)
   # print be_res_list
   # print af_res_list
   for i,j in zip(be_res_list,af_res_list):
      ax.plot(i, j, color_type[count], label=tool_name[count])
      count = count + 1
   ylabels = [r'0\%']
   for i in np.arange(0, 105, 20):
      ylabels.append(str(i) + "\%")
   ax.set_yticks(np.arange(0, 1.1, 0.2),ylabels)
   for tick in ax.xaxis.get_major_ticks():
       tick.label.set_fontsize(12)
       # ax.set_xticks(np.arange(0.5, 1.3, 0.1))
   # ax.set_xticklabels(['0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2'], fontsize=12)
   # plt.yticks(np.arange(0, 1.1, 0.2), ylabel)
   # plt.ylabel('% of benchmarks',fontsize=16)
   ax.set_yticklabels(ylabels,fontsize=12)
   ax.set_ylabel(r'\% of benchmarks',fontsize=16)
   # plt.xlabel('Time overhead ratio: Repair program/origin program',fontsize=20)
   ax.legend(loc=4,prop={'size': 14})
   ax.grid(True)
   # plt.ylim([0,1.1])
   plt.savefig(name + ".pdf", format="pdf")
   plt.show()


def read_time_list(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    bbpf_time_l = []
    abpf_time_l = []
    for i in range(0, table.nrows - 1):
        bbpf_time_l.append(float(table.row_values(i + 1)[10]))
        # bwpf_time_l.append(float(table.row_values(i + 1)[37]))
        abpf_time_l.append(float(table.row_values(i + 1)[16]))
            # awpf_time_l.append(float(table.row_values(i + 1)[47]))
    return bbpf_time_l,abpf_time_l


def plot_nv_timeoverhead(be_res_list,af_res_list,name):
   count = 0
   color_type = ['r--','k-']
   tool_name = [r'$Target1$',r'$Target2$']
   # fig = plt.figure(figsize=(14, 12))
   ax = plt.subplot(111)
   # print be_res_list
   # print af_res_list
   for i,j in zip(be_res_list,af_res_list):
      ax.plot(i, j, color_type[count], label=tool_name[count])
      count = count + 1
   # ylabels = [r'0\%']
   ylabels = []
   for i in np.arange(0, 105, 20):
      ylabels.append(str(i) + "\%")
   ax.set_yticks(np.arange(0, 1.1, 0.2),ylabels)
   for tick in ax.xaxis.get_major_ticks():
       tick.label.set_fontsize(12)
       # ax.set_xticks(np.arange(0.5, 1.3, 0.1))
   # ax.set_xticklabels(['0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2'], fontsize=12)
   # plt.yticks(np.arange(0, 1.1, 0.2), ylabel)
   # plt.ylabel('% of benchmarks',fontsize=16)
   ax.set_yticklabels(ylabels,fontsize=12)
   ax.set_ylabel(r'\% of benchmarks',fontsize=16)
   # plt.xlabel('Time overhead ratio: Repair program/origin program',fontsize=20)
   ax.legend(loc=4,prop={'size': 14})
   ax.grid(True)
   # plt.ylim([0,1.1])
   plt.savefig(name + ".pdf", format="pdf")
   plt.show()

def read_origin_err(n_var):
    pwd = os.getcwd()
    exname = pwd + "/final_results_" + str(n_var) + "v.xls"
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    r_val = []
    inp_val = []
    for i in range(0, table.nrows - 1):
        temp_err = np.log2(np.max([float(table.row_values(i + 1)[2]), 1.0]))
        if temp_err>32:
            r_val.append(temp_err)
    return r_val
def read_repaired_max_err(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    r_val = []
    inp_val = []
    for i in range(0, table.nrows - 1):
        temp_err = np.log2(np.max([float(table.row_values(i + 1)[17]), 1.0]))
        r_val.append(temp_err)
    return r_val

def read_repaired_avg_err(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    r_val = []
    r_val2 = []
    for i in range(0, table.nrows - 1):
        temp_err = np.log2(np.max([float(table.row_values(i + 1)[12]), 1.0]))
        temp_err2 = np.log2(np.max([float(table.row_values(i + 1)[18]), 1.0]))
        r_val.append(temp_err)
        r_val2.append(temp_err2)
    return r_val,r_val2

def read_repaired_name_id(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    name_lst = []
    id_lst = []
    rp_time_lst = []
    ph_lst = []
    for i in range(0, table.nrows - 1):
        temp_str = str(table.row_values(i + 1)[1])
        # temp_str = temp_str.strip("gsl/_sf/_")
        temp_str = temp_str.strip()
        # temp_str = temp_str[7:]
        # temp_str = temp_str.replace("_", "\_")
        name_lst.append(temp_str)
        id_lst.append(int(table.row_values(i + 1)[0]))
        rp_time_lst.append(float(table.row_values(i + 1)[6]))
        ph_lst.append(float(table.row_values(i + 1)[7])/1024.0)
    return name_lst,id_lst,rp_time_lst,ph_lst
def read_line_number(exname):
    data = xlrd.open_workbook(exname)
    table = data.sheets()[0]
    ln_lst = []
    for i in range(0, table.nrows - 1):
        ln_lst.append(int(table.row_values(i + 1)[8]))
    return ln_lst


def plot_max_err_improving_circle(err_lst1,err_lst2,name_lst,id_lst,name,th):
    r_l = []
    r_l = list(err_lst1)
    print r_l
    print len(r_l)
    print max(r_l)
    print min(r_l)
    # r = np.arange(0, 2, 0.01)
    n_rl = len(r_l)
    theta = np.deg2rad(np.arange(0, 350, 350.0 / n_rl))
    r = [i / 2.0 for i in r_l]
    r2 = [i / 2.0 for i in err_lst2]
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(111, projection='polar')
    # ax.scatter(theta, r, cmap='hsv', alpha=0.75)
    ax.set_rmax(34)
    ax.set_ylim(-1, 34)
    ax.margins(x=0, y=0)
    rt = np.arange(0, 34, 4)
    # print len(rt)
    ax.set_yticks(rt)  # less radial ticks
    # ax.tick_params(axis='both',which='both',color='b')
    # ax.set_yticklabels(['64','56','48','40','32','24','16','8','0'])
    ax.set_xticks([])
    ax.set_xticklabels([""])
    ax.set_yticklabels([""])
    tick = [0, ax.get_rmax()]
    print tick
    count = 1
    len_lst = []
    # for j in name_lst:
    #     text = ax.text(1,2,'%s' % j,{'ha': 'center', 'va': 'center'},fontsize = 15)
    #     rs = fig.canvas.get_renderer()
    #     bb = text.get_window_extent(renderer=rs)
    #     len_lst.append(bb.width)
    #     text.remove()

    for t, j in zip(np.deg2rad(np.arange(0, 350, 350.0 / n_rl)), id_lst):
        # for t in name_lst:
        #     k = len(j)
        ax.plot([t, t], tick, lw=0.72, color="lightgrey", zorder=2)
        ax.plot([t, t], [ax.get_rmax() - 0.5, ax.get_rmax()], lw=0.72, color="black", zorder=2)
        # if math.fmod(count,20)==0:
        ax.text(t, ax.get_rmax() + 0.3, '%s' % j, {'ha': 'left', 'va': 'center'},
                # transform=trans_offset,
                # horizontalalignment='right',
                # verticalalignment='bottom',
                rotation=t * 180 / (math.pi), fontsize=24, rotation_mode="anchor")
        count = count + 1
    ax.annotate("", xy=(theta[-1] + 0.22, 34), xytext=(theta[-1] + 0.22, -1), arrowprops=dict(arrowstyle="->"))
    yl = [r'\textbf{0}', '8', '16', '24', '32', '40', '48', '56', '64']
    for i, j in zip(rt, yl):
        ax.annotate(j, xy=(theta[-1] + 0.25, 34), xytext=(theta[-1] + 0.25, i), fontsize=14)
    ax.annotate('ErrBits', xy=(theta[-1] + 0.16, 30), xytext=(theta[-1] + 0.16, 28), fontsize=16,
                rotation=(theta[-1] + 0.22) * 180 / (math.pi), rotation_mode="anchor")
    print "********"
    print theta[-1]
    # ax.plot(theta, r, '--8', color='r', linewidth=1, label="Before repair",zorder=3,markersize=25,markerfacecolor="None")
    # ax.plot(theta, r2, '--^', color='seagreen', linewidth=1, label="After repair",zorder=3,markersize=14)
    ax.plot(theta, r, '-o', color='r', linewidth=1, label="Before repair", zorder=3, markersize=20)
    ax.plot(theta, r2, '-^', color='seagreen', linewidth=1, label="After repair", zorder=3, markersize=14)
    for i, j,p in zip(theta, r,r2):
    #     # ax.scatter(i, j,marker='8',  color='r',s =100,zorder=3,facecolors='none')
    #     # ax.scatter(i, p,marker='^',  color='seagreen', s =40, zorder=3)
        plt.plot([i,i],[j,p], linewidth=1, zorder=5,alpha=0.5, linestyle='--',color='black')
    #     plt.arrow(i, j, 0, p-j+0.6, lw=1, zorder=5,alpha=0.5, linestyle='--')
        # ax.annotate("", xy=(i, p), xytext=(i, p+1), arrowprops = dict(arrowstyle="->"))
              # fc=fc, ec=ec, alpha=alpha, width=width,
              # head_width=head_width, head_length=head_length,
              # **arrow_params)

    circle1 = plt.Circle((0, 0), th/2.0 + 1.0, color='deepskyblue',transform=ax.transData._b,fill=False,linewidth=3,zorder=2,linestyle = '--',label='Repair Target')
    ax.add_artist(circle1)
    # plt.legend(handles=[circle1],bbox_to_anchor=(0.4, 0.3), prop={'size': 14})
    # circle2 = plt.Circle((0, 0), 17, color='r', transform=ax.transData._b, fill=False, linewidth=3, clip_on=True,
    #                      zorder=3)
    # ax.add_artist(circle2)
    # ax.plot(theta, r,'.',color='r',markersize=14)
    # ax.autoscale(enable=True)
    # ax.axis('scaled')
    fn_lst = [18, 5, 4]
    angl_lst = []
    temp_angl = 0
    for i in fn_lst:
        temp_angl = temp_angl + (i) * 350 / (n_rl+0.5)
        angl_lst.append(temp_angl)
    print angl_lst
    angl_lst[-1]=338
    temp_angl = 0
    k = 0
    # # color_lst = ['skyblue','azure','silver','lavender']
    color_lst = ['darkorange', 'yellow', 'green']
    # color_lst = ['red','red','red','red']
    count = 0
    weg_lst = []
    print angl_lst
    for i in angl_lst:
        # ax.add_artist(Wedge((.0, .0), 35, temp_angl, i, transform=ax.transData._b,width=1,capstyle='round', fill=True, color=color_lst[k], alpha=0.5,zorder=4))
        if k == 0:
            lb_name = str(k + 2) + "-input"
        else:
            lb_name = str(k + 2) + "-input"
        temp_weg = Wedge((.0, .0), 34.2, temp_angl, i, transform=ax.transData._b, width=1, fill=True,color=color_lst[k], alpha=0.5, zorder=2, label=lb_name)
        ax.add_artist(temp_weg)
        weg_lst.append(temp_weg)
        temp_angl = i
        k = k + 1
    sec_legend = plt.legend(handles=weg_lst, bbox_to_anchor=(-0.03, 0.9), prop={'size': 20})
    ax.add_artist(sec_legend)
    first_legend = plt.legend(handles=[circle1], bbox_to_anchor=(0.0, 0.08), prop={'size': 19.5})
    # Add the legend manually to the current Axes.
    ax.add_artist(first_legend)

    ax.legend(bbox_to_anchor=(0.0, 0.2), prop={'size': 20})
    # ax.set_yticklabels([r'\textbf{0}', '8', '16', '24', '32', '40', '48', '56', '64'], zorder=4, fontsize=20)
    # ax.set_yticklabels(['0','16','32','48','64'])
    ax.set_rlabel_position(-15)  # get radial labels away from plotted line
    plt.savefig("graph/"+name+".pdf", format="pdf")
    # ax.legend()
    plt.show()
    return 0

def plot_avg_err_improving_circle(err_lst1,err_lst2,name_lst,id_lst,name,th):
    r_l = []
    r_l = list(err_lst1)
    print r_l
    print len(r_l)
    print max(r_l)
    print min(r_l)
    # r = np.arange(0, 2, 0.01)
    n_rl = len(r_l)
    theta = np.deg2rad(np.arange(0, 350, 350.0 / n_rl))
    r = [i / 2.0 for i in r_l]
    r2 = [i / 2.0 for i in err_lst2]
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(111, projection='polar')
    # ax.scatter(theta, r, cmap='hsv', alpha=0.75)
    ax.set_rmax(9)
    ax.set_ylim(-1,9)
    ax.margins(x=0, y=0)
    rt = np.arange(0, 14, 2)
    # print len(rt)
    ax.set_yticks(rt)  # less radial ticks
    # ax.tick_params(axis='both',which='both',color='b')
    # ax.set_yticklabels(['64','56','48','40','32','24','16','8','0'])
    ax.set_xticks([])
    ax.set_xticklabels([""])
    tick = [0, ax.get_rmax()]
    print tick
    count = 1
    len_lst = []
    # for j in name_lst:
    #     text = ax.text(1,2,'%s' % j,{'ha': 'center', 'va': 'center'},fontsize = 15)
    #     rs = fig.canvas.get_renderer()
    #     bb = text.get_window_extent(renderer=rs)
    #     len_lst.append(bb.width)
    #     text.remove()

    for t, j in zip(np.deg2rad(np.arange(0, 350, 350.0 / n_rl)), id_lst):
        # for t in name_lst:
        #     k = len(j)
        ax.plot([t, t], tick, lw=0.72, color="lightgrey", zorder=2)
        ax.plot([t, t], [ax.get_rmax() - 0.5, ax.get_rmax()], lw=0.72, color="black", zorder=2)
        # if math.fmod(count,20)==0:
        ax.text(t, ax.get_rmax() + 0.3, '%s' % j, {'ha': 'left', 'va': 'center'},
                # transform=trans_offset,
                # horizontalalignment='right',
                # verticalalignment='bottom',
                rotation=t * 180 / (math.pi), fontsize=24, rotation_mode="anchor")
        count = count + 1
    # ax.plot(theta, r, '--8', color='r', linewidth=1, label="Before repair", zorder=3,markersize=12)
    # ax.plot(theta, r2, '--^', color='seagreen', linewidth=1, label="After repair", zorder=3,markersize=10)
    # for i, j, p in zip(theta, r, r2):
        # ax.plot(i, j, '.', color='r', markersize=14, zorder=3)
        # ax.plot(i, p, '.', color='seagreen', markersize=12, zorder=3)
        # plt.arrow(i, j, 0, p - j, lw=1, zorder=5, alpha=0.5, linestyle='--')
        # ax.annotate("", xy=(i, p), xytext=(i, p + 0.5), arrowprops=dict(arrowstyle="->"))
    ax.plot(theta, r, '-o', color='r', linewidth=1, label="Before repair", zorder=3, markersize=20)
    ax.plot(theta, r2, '-^', color='seagreen', linewidth=1, label="After repair", zorder=3, markersize=14)
    for i, j, p in zip(theta, r, r2):
        #     # ax.scatter(i, j,marker='8',  color='r',s =100,zorder=3,facecolors='none')
        #     # ax.scatter(i, p,marker='^',  color='seagreen', s =40, zorder=3)
        plt.plot([i, i], [j, p], linewidth=1, zorder=5, alpha=0.5, linestyle='--', color='black')
    # ax.plot(theta, r,'-8',color='r',linewidth=1,label="Before repair")
    # ax.plot(theta, r2,'-s',color='seagreen',linewidth=1,label="After repair")
    # for i, j,p in zip(theta, r,r2):
    #     # ax.plot(i, j, '.', color='r', markersize=14, zorder=3)
    #     # ax.plot(i, p, '.', color='seagreen', markersize=12, zorder=3)
    #     # plt.arrow(i, j, 0, p-j+0.6, lw=1, zorder=5,alpha=0.5, head_width=0.15,head_length=0.6)
    #     ax.annotate("", xy=(i, p), xytext=(i, j), arrowprops = dict(arrowstyle="->"))
              # fc=fc, ec=ec, alpha=alpha, width=width,
              # head_width=head_width, head_length=head_length,
              # **arrow_params)

    circle1 = plt.Circle((0, 0), th/2.0 + 1.0, color='deepskyblue',transform=ax.transData._b,fill=False,linewidth=3,zorder=2,linestyle = '--',label='Repair Target')
    ax.add_artist(circle1)
    first_legend = plt.legend(handles=[circle1], bbox_to_anchor=(0.0, 0.08), prop={'size': 19.5})
    # Add the legend manually to the current Axes.
    ax.add_artist(first_legend)
    # circle2 = plt.Circle((0, 0), 17, color='r', transform=ax.transData._b, fill=False, linewidth=3, clip_on=True,
    #                      zorder=3)
    # ax.add_artist(circle2)
    # ax.plot(theta, r,'.',color='r',markersize=14)
    # ax.autoscale(enable=True)
    # ax.axis('scaled')
    ax.set_yticklabels([r'\textbf{0}', '4', '8', '12', '16', '20', '24', '28', '32'], zorder=4, fontsize=20)
    # ax.set_yticklabels([r'\textbf{0}', '8', '16', '24', '32', '40', '48', '56', '64'], zorder=4, fontsize=12)
    # ax.set_yticklabels(['0','16','32','48','64'])
    ax.set_rlabel_position(-15)  # get radial labels away from plotted line
    ax.legend(bbox_to_anchor=(0.0,0.2),prop={'size': 20})
    plt.savefig("graph/"+name+".pdf", format="pdf")
    # ax.legend()
    plt.show()
    return 0


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,fontsize=10, weight='bold')                      # Vertically align label differently for
                                        # positive and negative values.

def process_Rp_time(rpt1,rpt2,rpt3,rpt4,name_lst,id_lst,name):
    rpt1_lst = []
    for i,j in zip(rpt1,rpt2):
        # rpt1_lst.append(np.log2(i*1000))
        rpt1_lst.append(i/j)
    rpt2_lst = []
    for i,j in zip(rpt3,rpt4):
        # rpt2_lst.append(np.log2(i*1000))
        rpt2_lst.append(i/j)
    print rpt1_lst
    print rpt2_lst
    x = rpt1_lst
    y = rpt2_lst
    n_groups = len(id_lst)
    means_frank = rpt1_lst
    means_guido = rpt2_lst
    print np.max(rpt1_lst)
    print np.min(rpt1_lst)
    print np.max(rpt2_lst)
    print np.min(rpt2_lst)
    # create plot
    fig = plt.figure(figsize=(19, 8))
    ax = plt.subplot(111)
    index = np.arange(0,n_groups,1)
    # index = range(0,n_groups,2)
    print index
    bar_width = 0.45
    opacity = 0.8

    # plt.boxplot(rpt1_lst)
    # plt.boxplot(rpt2_lst)

    ax.bar(index, means_frank, bar_width,edgecolor='purple', color='None',hatch="/",label=r'$H_\varepsilon$')

    ax.bar(index + bar_width, means_guido, bar_width, label=r'$L_\varepsilon$')
    add_value_labels(ax)
    plt.xlabel('Program ID',fontsize=20)
    # plt.ylabel('Repair time ratios',fontsize=20)
    plt.ylabel('Patch size ratios',fontsize=20)
    # plt.title('Scores by person')
    # plt.yticks(index+0.5*bar_width, id_lst,rotation=0)
    plt.xticks(index+0.5*bar_width, id_lst,rotation=0,fontsize=16)
    # plt.xticks(range(0,int(np.max(rpt1_lst))+1,2), range(0,int(np.max(rpt1_lst))*10+10,20),rotation=30)
    plt.yticks(range(0,int(np.max(rpt1_lst))+10,20), range(0,int(np.max(rpt1_lst))*10+10,20),fontsize=16)
    plt.legend(prop={'size': 19.5})
    plt.tight_layout()
    plt.grid(zorder=1)
    # plt.tight_layout()
    plt.savefig("graph/" + name + ".pdf", format="pdf")
    plt.show()
    return 0


def process_Rp_time_box_plot(rpt1,rpt2,rpt3,rpt4,name_lst,id_lst,name):
    # create plot
    fig, ax = plt.subplots(1, figsize=(4, 6))
    print rpt1
    print rpt2
    print np.min(rpt1)
    print np.min(rpt2)
    bp = ax.boxplot([rpt2],[1])
    # bp2 = ax.boxplot(rpt2,2)
    # plt.tight_layout()
    # plt.savefig("graph/" + name + ".pdf", format="pdf")
    plt.show()
    return 0


def save_rpt_ph_2Excel(exname,rp_t1,rp_t2,rp_ta1,rp_ta2,id_lst):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "Program ID")
    sheet.write(1, 0, "AutoRNP")
    sheet.write(2, 0, "NPTaylor")
    sheet.write(3, 0, "Ratio")
    k = 1
    for a,i,j,p,q in zip(id_lst,rp_t1,rp_t2,rp_ta1,rp_ta2):
        if k == 10:
            sheet.write(0, k, str(a))
            sheet.write(1, k, "%.2f" % p)
            sheet.write(2, k, "%.2f" % q)
            sheet.write(3, k, "%.2f" % (p / q))
            sheet.write(4, k, "%.2f" % i)
            sheet.write(5, k, "%.2f" % j)
            sheet.write(6, k, "%.3f" % (i / j))
        # sheet.write(k+1, 1, repr(i))
        # sheet.write(k+1, 2, repr(j))
        # sheet.write(k+1, 3, repr(i/j))
        k = k+1
    book.save(exname)
def save_1v_maxErr2Excel(exname,err1,err2,id_lst):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "Program ID")
    sheet.write(0, 1, "AutoRNP")
    sheet.write(0, 2, "NPTaylor")
    k = 0
    style = xlwt.XFStyle()

    # font
    font = xlwt.Font()
    font.bold = True
    style.font = font
    for a,i,j in zip(id_lst,err1,err2):
        sheet.write(k + 1, 0, str(a))
        if i < j:
            sheet.write(k+1, 1, repr(i), style=style)
            sheet.write(k+1, 2, repr(j))
        else:
            if i==j:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j))
            else:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j), style=style)

        k = k+1
    book.save(exname)
def save_nv_maxErr2Excel(exname,err1,err2,id_lst):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "Program ID")
    sheet.write(0, 1, "Target1")
    sheet.write(0, 2, "Target2")
    k = 0
    style = xlwt.XFStyle()

    # font
    font = xlwt.Font()
    font.bold = True
    style.font = font
    for a,i,j in zip(id_lst,err1,err2):
        sheet.write(k + 1, 0, str(a))
        if i < j:
            sheet.write(k+1, 1, repr(i), style=style)
            sheet.write(k+1, 2, repr(j))
        else:
            if i==j:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j))
            else:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j), style=style)

        k = k+1
    book.save(exname)
def save_1v_reptExcel(exname,err1,err2,id_lst):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "Program ID")
    sheet.write(0, 1, "AutoRNP")
    sheet.write(0, 2, "NPTaylor")
    k = 0
    style = xlwt.XFStyle()

    # font
    font = xlwt.Font()
    font.bold = True
    style.font = font
    for a,i,j in zip(id_lst,err1,err2):
        sheet.write(k + 1, 0, str(a))
        if i < j:
            sheet.write(k+1, 1, repr(i), style=style)
            sheet.write(k+1, 2, repr(j))
        else:
            if i==j:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j))
            else:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j), style=style)
        sheet.write(k + 1, 3, repr(i/j))
        k = k+1
    book.save(exname)
def save_nv_reptExcel(exname,err1,err2,id_lst):
    book = xlwt.Workbook()
    sheet = book.add_sheet("sheet1")
    sheet.write(0, 0, "Program ID")
    sheet.write(0, 1, "AutoRNP")
    sheet.write(0, 2, "NPTaylor")
    k = 0
    style = xlwt.XFStyle()
    # font
    font = xlwt.Font()
    font.bold = True
    style.font = font
    for a,i,j in zip(id_lst,err1,err2):
        sheet.write(k + 1, 0, str(a))
        if i < j:
            sheet.write(k+1, 1, repr(i), style=style)
            sheet.write(k+1, 2, repr(j))
        else:
            if i==j:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j))
            else:
                sheet.write(k + 1, 1, repr(i))
                sheet.write(k + 1, 2, repr(j), style=style)
        sheet.write(k + 1, 3, repr(i/j))
        k = k+1
    book.save(exname)

def analysis_fpexp_res(fpexp_res,name_lst):
    final_res = []
    for i in name_lst:
        count = 0
        max_acc = 0
        temp_acc = 0
        for j in fpexp_res:
            if i == j[1]:
               count = count + 1
               print j
               if j[3] != '':
                  temp_acc = float(j[3])-float(j[5])
               if temp_acc > max_acc:
                   max_acc = temp_acc
        final_res.append([i,count,max_acc])
    return final_res

def write_to_HBG_res(exname,hbg_res,nvar):
    old_excel = xlrd.open_workbook(exname, formatting_info=True)
    # table = old_excel.sheets()[0]
    new_excel = copy(old_excel)
    sheet = new_excel.get_sheet(0)
    add_len_lst = [0,0,30,48,53]
    add_len = add_len_lst[nvar]
    k = 1
    for i in hbg_res:
        # sheet.write(k+add_len, 6, int(i[1]))
        # sheet.write(k+add_len, 7, repr(i[2]))
        sheet.write(k+add_len, 9, repr(i))
        k = k+1
    new_excel.save(exname)
    return 0
file_lst1 = ['experiment_resultslqt1612/table_results/experiment_results_total1.xls','experiment_resultstay1v1612/table_results/experiment_results_total1.xls']
file_lst2 = ['experiment_resultslqt1612/table_results/experiment_results_total2.xls','experiment_resultstay1v1612/table_results/experiment_results_total2.xls']
file_lst3 = ['experiment_resultstay2v1612/table_results/experiment_results_total1.xls','experiment_resultstay2v1612/table_results/experiment_results_total2.xls']
file_lst4 = ['experiment_resultstay3v1612/table_results/experiment_results_total1.xls','experiment_resultstay3v1612/table_results/experiment_results_total2.xls']

# file_lst3 = ['back_up_whole/experiment_resultstay2v1612/table_results/experiment_results_total1.xls','back_up_whole/experiment_resultstay2v1612/table_results/experiment_results_total2.xls']
# file_lst4 = ['back_up_whole/experiment_resultstay3v1612/table_results/experiment_results_total1.xls','back_up_whole/experiment_resultstay3v1612/table_results/experiment_results_total2.xls']

file_lst5 = ['experiment_resultstay4v1612/table_results/experiment_results_total1.xls','experiment_resultstay4v1612/table_results/experiment_results_total2.xls']
# hbg_exname = '../benchmarks/driver_functions/HBG_res_compare.xls'
hbg_exname = '../benchmarks/driver_functions/HBG_res_compare2.xls'
# for n v functions time ovehead
f1 = open("test_gsl_herbie/gsl1v/report.html")  # simplified for the example (no urllib)
f2 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl2v/report.html")  # simplified for the example (no urllib)
f3 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl3v/report.html")  # simplified for the example (no urllib)
f4 = open("/home/yixin/software/HBG/test_gsl_herbie/gsl4v/report.html")  # simplified for the example (no urllib)


avg_errb1, avg_erra1 = read_repaired_avg_err(file_lst2[1])
avg_errb0, avg_erra0 = read_repaired_avg_err(file_lst2[0])
avg_errb = []
for i, j in zip(avg_errb1, avg_errb0):
    # avg_errb.append(np.log2((math.pow(2.0, i) + math.pow(2.0, j)) / 2.0))
    avg_errb.append(np.log2((math.pow(2.0, i) + math.pow(2.0, j)) / 2.0))
avg_imp_1v = [i-j for i,j in zip(avg_errb,avg_erra0)]
print avg_imp_1v
avg_errb1, avg_erra1 = read_repaired_avg_err(file_lst3[1])
avg_imp_2v = [i-j for i,j in zip(avg_errb1,avg_erra1)]
print avg_imp_2v
avg_errb1, avg_erra1 = read_repaired_avg_err(file_lst4[1])
avg_imp_3v = [i-j for i,j in zip(avg_errb1,avg_erra1)]
print avg_imp_3v
avg_errb1, avg_erra1 = read_repaired_avg_err(file_lst5[1])
avg_imp_4v = [i-j for i,j in zip(avg_errb1,avg_erra1)]
print avg_imp_4v
avg_err_all = avg_imp_1v + avg_imp_2v + avg_imp_3v + avg_imp_4v
print avg_err_all
# write_to_HBG_res(hbg_exname, avg_err_all, 1)
# repair accuracy 12
# org_err2,err_lst2 = read_repaired_avg_err(file_lst2[1])
# org_err3,err_lst3 = read_repaired_avg_err(file_lst2[0])
# save_1v_maxErr2Excel('graph/1v_avg_12.xls', err_lst3, err_lst2, id_lst)
# process_Rp_time(ph_lst1, ph_lst2, ph_lsta1, ph_lsta2, name_lst, id_lst, "Patch_size_ratio")
# get_cumulative_dis_time('/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_resultslqt1612/table_results/experiment_results_total2.xls')
# get_cumulative_dis_time('/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_resultstay1v1612/table_results/experiment_results_total2.xls')

def extract_graph():
    # time overhead AutoRNP vs NPTaylor
    file_lst1 = ['experiment_resultslqt1612/table_results/experiment_results_total1.xls',
                 'experiment_resultstay1v1612/table_results/experiment_results_total1.xls']
    file_lst2 = ['experiment_resultslqt1612/table_results/experiment_results_total2.xls',
                 'experiment_resultstay1v1612/table_results/experiment_results_total2.xls']
    # file_lst3 = ['/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_resultslqt1612/table_results/experiment_results_total2.xls','/home/yixin/PycharmProjects/NPTaylor/experiments/experiment_resultstay1v1612/table_results/experiment_results_total2.xls']
    #
    plot_compare_timeoverhead(file_lst1,'timeOverhead1v16')
    plot_compare_timeoverhead(file_lst2,'timeOverhead1v12')
    # max_err improving AutoRNP vs NPTaylor
    # repair accuracy 16
    err_lst1 = read_origin_err(1)
    err_lst2 = read_repaired_max_err(file_lst1[1])
    name_lst, id_lst, rp_time_lst,ph_lst = read_repaired_name_id(file_lst1[0])
    plot_max_err_improving_circle(err_lst1, err_lst2, name_lst, id_lst, 'MaxErr_tay1v16', 16)
    err_lst2 = read_repaired_max_err(file_lst1[0])
    plot_max_err_improving_circle(err_lst1, err_lst2, name_lst, id_lst, 'MaxErr_line1v16', 16)
    # repair accuracy 12
    err_lst2 = read_repaired_max_err(file_lst2[1])
    plot_max_err_improving_circle(err_lst1, err_lst2, name_lst, id_lst, 'MaxErr_tay1v12', 12)
    err_lst2 = read_repaired_max_err(file_lst2[0])
    plot_max_err_improving_circle(err_lst1, err_lst2, name_lst, id_lst, 'MaxErr_line1v12', 12)

    # avg_err improving AutoRNP vs NPTaylor
    # accuracy 16
    avg_errb1,avg_erra1 = read_repaired_avg_err(file_lst1[1])
    avg_errb0,avg_erra0 = read_repaired_avg_err(file_lst1[0])
    avg_errb = []
    for i,j in zip(avg_errb1,avg_errb0):
        avg_errb.append(np.log2((math.pow(2.0,i)+math.pow(2.0,j))/2.0))
    plot_avg_err_improving_circle(avg_errb, avg_erra1, name_lst, id_lst, 'AvgErr_tay1v16', 16)
    plot_avg_err_improving_circle(avg_errb, avg_erra0, name_lst, id_lst, 'AvgErr_line1v16', 16)
    # accuracy 12
    avg_errb1, avg_erra1 = read_repaired_avg_err(file_lst2[1])
    avg_errb0, avg_erra0 = read_repaired_avg_err(file_lst2[0])
    avg_errb = []
    for i, j in zip(avg_errb1, avg_errb0):
        avg_errb.append(np.log2((math.pow(2.0, i) + math.pow(2.0, j)) / 2.0))
    plot_avg_err_improving_circle(avg_errb, avg_erra1, name_lst, id_lst, 'AvgErr_tay1v12', 12)
    plot_avg_err_improving_circle(avg_errb, avg_erra0, name_lst, id_lst, 'AvgErr_line1v12', 12)

    # Repair time compare AutoRNP vs NPTaylor
    # accuracy 16
    name_lst, id_lst, rp_time_lsta1, ph_lsta1 = read_repaired_name_id(file_lst1[0])
    name_lst, id_lst, rp_time_lsta2, ph_lsta2 = read_repaired_name_id(file_lst1[1])
    name_lst, id_lst, rp_time_lst1, ph_lst1 = read_repaired_name_id(file_lst2[0])
    name_lst, id_lst, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst2[1])
    process_Rp_time(rp_time_lst1, rp_time_lst2, rp_time_lsta1, rp_time_lsta2, name_lst, id_lst, "Repair_time_ratio")
    process_Rp_time(ph_lst1, ph_lst2, ph_lsta1, ph_lsta2, name_lst, id_lst, "Patch_size_ratio")
    save_rpt_ph_2Excel('rpt_vs.xls', rp_time_lst1, rp_time_lst2, rp_time_lsta1, rp_time_lsta2, id_lst)
    save_rpt_ph_2Excel('pch_vs.xls', ph_lst1, ph_lst2, ph_lsta1, ph_lsta2, id_lst)

    # for n v functions max_err improving
    err_lst1 = read_origin_err(2) + read_origin_err(3) + read_origin_err(4)
    err_lst2 = read_repaired_max_err(file_lst3[0]) + read_repaired_max_err(file_lst4[0]) + read_origin_err(4)
    err_lst3 = read_repaired_max_err(file_lst3[1]) + read_repaired_max_err(file_lst4[1]) + read_origin_err(4)
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst3[0])
    name_lst3, id_lst3, rp_time_lst3, ph_lst3 = read_repaired_name_id(file_lst4[0])
    name_lst4, id_lst4, rp_time_lst4, ph_lst4 = read_repaired_name_id(file_lst5[0])
    id_lst = [i + 104 for i in id_lst2] + [i + 143 for i in id_lst3] + [i + 150 for i in id_lst4]
    name_lst = name_lst2 + name_lst3 + name_lst4
    id_lst_new = ['P' + str(i) for i in id_lst]
    plot_max_err_improving_circle(err_lst1, err_lst2, name_lst, id_lst_new, 'MaxErr_taynv16', 16)
    plot_max_err_improving_circle(err_lst1, err_lst3, name_lst, id_lst_new, 'MaxErr_taynv12', 12)

    # avg_err improving AutoRNP vs NPTaylor
    # accuracy 16
    org_err2, err_lst2 = read_repaired_avg_err(file_lst1[1])
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst1[0])
    org_err3, err_lst3 = read_repaired_avg_err(file_lst1[0])
    save_1v_maxErr2Excel('graph/1v_avg_16.xls', err_lst3, err_lst2, id_lst)
    # repair accuracy 12
    org_err2, err_lst2 = read_repaired_avg_err(file_lst2[1])
    org_err3, err_lst3 = read_repaired_avg_err(file_lst2[0])
    save_1v_maxErr2Excel('graph/1v_avg_12.xls', err_lst3, err_lst2, id_lst)

    # repair time AutoRNP vs NPTaylor
    # accuracy 16
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst1[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst1[1])
    save_1v_reptExcel('graph/1v_rept_16.xls', rp_time_lst, rp_time_lst2, id_lst)
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst2[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst2[1])
    save_1v_reptExcel('graph/1v_rept_12.xls', rp_time_lst, rp_time_lst2, id_lst)

    # patch size AutoRNP vs NPTaylor
    # accuracy 16
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst1[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst1[1])
    save_1v_reptExcel('graph/1v_phsz_16.xls', ph_lst, ph_lst2, id_lst)
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst2[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst2[1])
    save_1v_reptExcel('graph/1v_phsz_12.xls', ph_lst, ph_lst2, id_lst)

    # patch size AutoRNP vs NPTaylor
    # accuracy 16
    from scipy import stats
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst1[0])
    ln_lst = read_line_number(file_lst1[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst1[1])
    ln_lst2 = read_line_number(file_lst1[1])
    print stats.ttest_ind(rp_time_lst, rp_time_lst2)
    save_1v_reptExcel('graph/1v_ln_16.xls', ln_lst, ln_lst2, id_lst)
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst2[0])
    ln_lst = read_line_number(file_lst2[0])
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst2[1])
    ln_lst2 = read_line_number(file_lst2[1])
    print stats.ttest_ind(rp_time_lst, rp_time_lst2)
    save_1v_reptExcel('graph/1v_ln_12.xls', ln_lst, ln_lst2, id_lst)

    # nv results table
    # accuracy 16 read_line_number(exname)
    err_lst1 = read_origin_err(2) + read_origin_err(3) + read_origin_err(4)
    err_lst2 = read_repaired_max_err(file_lst3[0]) + read_repaired_max_err(file_lst4[0]) + read_origin_err(4)
    err_lst3 = read_repaired_max_err(file_lst3[1]) + read_repaired_max_err(file_lst4[1]) + read_origin_err(4)
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst3[0])
    name_lst3, id_lst3, rp_time_lst3, ph_lst3 = read_repaired_name_id(file_lst4[0])
    name_lst4, id_lst4, rp_time_lst4, ph_lst4 = read_repaired_name_id(file_lst5[0])
    ln_number1 = read_line_number(file_lst3[0])
    ln_number2 = read_line_number(file_lst4[0])
    ln_number3 = read_line_number(file_lst5[0])
    name_lst2, id_lst2b, rp_time_lst2b, ph_lst2b = read_repaired_name_id(file_lst3[1])
    name_lst3, id_lst3b, rp_time_lst3b, ph_lst3b = read_repaired_name_id(file_lst4[1])
    name_lst4, id_lst4b, rp_time_lst4b, ph_lst4b = read_repaired_name_id(file_lst5[1])
    ln_number1b = read_line_number(file_lst3[1])
    ln_number2b = read_line_number(file_lst4[1])
    ln_number3b = read_line_number(file_lst5[1])
    id_lst = [i + 104 for i in id_lst2] + [i + 143 for i in id_lst3] + [i + 150 for i in id_lst4]
    id_lst_new = ['P' + str(i) for i in id_lst]
    name_lst = name_lst2 + name_lst3 + name_lst4
    rp_time_lstL = rp_time_lst2 + rp_time_lst3 + rp_time_lst4
    rp_time_lstH = rp_time_lst2b + rp_time_lst3b + rp_time_lst4b
    ph_lstL = ph_lst2 + ph_lst3 + ph_lst4
    ph_lstH = ph_lst2b + ph_lst3b + ph_lst4b
    ln_numberL = ln_number1 + ln_number2 + ln_number3
    ln_numberH = ln_number1b + ln_number2b + ln_number3b
    save_nv_maxErr2Excel('graph/nv_avg_12.xls', err_lst3, err_lst2, id_lst_new)
    save_nv_maxErr2Excel('graph/nv_rept_12.xls', rp_time_lstL, rp_time_lstH, id_lst_new)
    save_nv_maxErr2Excel('graph/nv_phsz_12.xls', ph_lstL, ph_lstH, id_lst_new)
    save_nv_maxErr2Excel('graph/nv_ln_12.xls', ln_numberL, ln_numberH, id_lst_new)

    # for n v functions avg_err improving
    org_err2, err_lst2 = read_repaired_avg_err(file_lst3[0])
    org_err3, err_lst3 = read_repaired_avg_err(file_lst4[0])
    org_err4, err_lst4 = read_repaired_avg_err(file_lst5[0])
    org_err = org_err2 + org_err3 + org_err4
    err_lstL = err_lst2 + err_lst3 + org_err4
    org_err2b, err_lst2b = read_repaired_avg_err(file_lst3[1])
    org_err3b, err_lst3b = read_repaired_avg_err(file_lst4[1])
    org_err4b, err_lst4b = read_repaired_avg_err(file_lst5[1])
    # org_err = org_err2+org_err3+org_err4
    err_lstH = err_lst2b + err_lst3b + org_err4
    err_lstL[8] = org_err[8]
    err_lstL[15] = org_err[15]
    err_lstH[8] = org_err[8]
    err_lstH[15] = org_err[15]
    name_lst2, id_lst2, rp_time_lst2, ph_lst2 = read_repaired_name_id(file_lst3[0])
    name_lst3, id_lst3, rp_time_lst3, ph_lst3 = read_repaired_name_id(file_lst4[0])
    name_lst4, id_lst4, rp_time_lst4, ph_lst4 = read_repaired_name_id(file_lst5[0])
    id_lst = [i + 104 for i in id_lst2] + [i + 143 for i in id_lst3] + [i + 150 for i in id_lst4]
    name_lst = name_lst2 + name_lst3 + name_lst4
    id_lst_new = ['P' + str(i) for i in id_lst]
    id_lst_new.reverse()
    # plot_max_err_improving_circle(org_err, err_lstL, name_lst, id_lst_new, 'AvgErr_taynv16', 16)
    plot_compare_timeoverhead(file_lst1, 'timeOverhead1v16')
    draw_arrow_error(org_err, err_lstL, 16, id_lst_new, 'graph/AvgErr_taynv16')
    draw_arrow_error(org_err, err_lstH, 12, id_lst_new, 'graph/AvgErr_taynv12')

    # for n v functions time ovehead
    tohb2, toha2 = read_time_list(file_lst3[0])
    tohb3, toha3 = read_time_list(file_lst4[0])
    tohbL = tohb2 + tohb3
    tohaL = toha2 + toha3
    for i in [8, 15, 18]:
        tohbL.pop(i)
        tohaL.pop(i)
    print len(tohbL)
    z1, y1 = process_cumulative_dis_time(tohbL, tohaL)
    tohb2, toha2 = read_time_list(file_lst3[1])
    tohb3, toha3 = read_time_list(file_lst4[1])
    tohbH = tohb2 + tohb3
    tohaH = toha2 + toha3
    for i in [8, 15, 18]:
        tohbH.pop(i)
        tohaH.pop(i)
    z2, y2 = process_cumulative_dis_time(tohbH, tohaH)
    betolst = [z1, z2]
    aftolst = [y1, y2]
    plot_nv_timeoverhead(betolst, aftolst, "graph/nv_timeoverhead")

    # HBG results
    fpexp_res_1v = extract_html_res(f1, 1)
    fpexp_res_2v = extract_html_res(f2, 2)
    fpexp_res_3v = extract_html_res(f3, 2)
    fpexp_res_4v = extract_html_res(f4, 2)
    # name of all 57 functions
    # name_lst, id_lst, rp_time_lst,ph_lst = read_repaired_name_id(file_lst1[0])
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst1[0])
    hbg_res = analysis_fpexp_res(fpexp_res_1v, name_lst)
    write_to_HBG_res(hbg_exname, hbg_res, 1)
    # 2v
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst3[0])
    hbg_res = analysis_fpexp_res(fpexp_res_2v, name_lst)
    write_to_HBG_res(hbg_exname, hbg_res, 2)
    # 3v
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst4[0])
    hbg_res = analysis_fpexp_res(fpexp_res_3v, name_lst)
    write_to_HBG_res(hbg_exname, hbg_res, 3)
    # 4v
    name_lst, id_lst, rp_time_lst, ph_lst = read_repaired_name_id(file_lst5[0])
    hbg_res = analysis_fpexp_res(fpexp_res_4v, name_lst)
    write_to_HBG_res(hbg_exname, hbg_res, 4)

tohb2, toha2 = read_time_list(file_lst3[0])
tohb3, toha3 = read_time_list(file_lst4[0])
tohbL = tohb2 + tohb3
tohaL = toha2 + toha3
for i in [8, 15, 18]:
    tohbL.pop(i)
    tohaL.pop(i)
print len(tohbL)
z1, y1 = process_cumulative_dis_time(tohbL, tohaL)
tohb2, toha2 = read_time_list(file_lst3[1])
tohb3, toha3 = read_time_list(file_lst4[1])
tohbH = tohb2 + tohb3
tohaH = toha2 + toha3
for i in [8, 15, 18]:
    tohbH.pop(i)
    tohaH.pop(i)
z2, y2 = process_cumulative_dis_time(tohbH, tohaH)
betolst = [z1, z2]
aftolst = [y1, y2]
# plot_nv_timeoverhead(betolst, aftolst, "graph/nv_timeoverhead_whole")
plot_nv_timeoverhead(betolst, aftolst, "graph/nv_timeoverhead")