import os
nProcs = 64
cmd = 'taskset -c 0-{} python3 rbf_pe.py'.format(nProcs-1, nProcs)
os.system(cmd)

