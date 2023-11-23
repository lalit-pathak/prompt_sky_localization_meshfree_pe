import os
nProcs = 32
cmd = 'taskset -c 0-{} python3 run_pe.py {}'.format(nProcs-1, nProcs)
os.system(cmd)