'''@file kill_processes.py
contains the kill_processes method'''

import os
import socket

def kill_processes(processdir):
    '''kill all processes that reported in the processdir'''

    files = os.listdir(processdir)

    for f in files:
        splitfile = f.split('-')
        machine = splitfile[0]
        pid = splitfile[1]
        if (machine == socket.gethostbyname(socket.gethostname())
                or machine == '127.0.0.1'):
            os.system('kill %s' % pid)
        else:
            os.system('ssh -o StrictHostKeyChecking=no -o '
                      'UserKnownHostsFile=/dev/null %s "kill %s"'
                      % (machine, pid))
