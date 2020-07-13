import os
# for i in range(1,4):
#     cmd = "./run_experiment.sh " + str(i) + " " + str(i)
#     os.system(cmd)
def sudo_cmd(cmd):
    sudoPassword = 'hello'
    os.system('echo %s|sudo -S %s' % (sudoPassword, cmd))
os.system("./unpatch_cmd.sh")
sudo_cmd("./make_install.sh > tmp_log")
for i in range(2,3):
    cmd = "./run_experiment2v.sh " + str(i) + " " + str(i)
    os.system(cmd)
# os.system("./unpatch_cmd.sh")
# sudo_cmd("./make_install.sh > tmp_log")
# for i in range(1,4):
#     cmd = "./run_experiment2.sh " + str(i) + " " + str(i)
#     os.system(cmd)