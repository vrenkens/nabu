'''@file kill_processes.py
contains the kill_processes method'''

import os

def kill_processes(processdir):
    '''kill all processes that reported in the processdir'''

    files = os.listdir(processdir)

    for f in files:
        splitfile = f.split('-')
        machine = splitfile[0]
        pid = splitfile[1]
        os.system('ssh %s "kill %s"' % (machine, pid))
