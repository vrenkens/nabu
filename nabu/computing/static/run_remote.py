'''@file run_remote.py
contains the run_remote function'''

import os
import subprocess
import socket

def run_remote(host, command):
    '''run a command remotely over ssh

    Args:
        host: the host where the command should be ran
        command: the command that should be executed
        out: the file where the output should be written

    Returns:
        a popen process'''

    if (host == socket.gethostbyname(socket.gethostname())
            or host == '127.0.0.1'):
        pid = subprocess.Popen(command.split(' '))
    else:
        print 'running remote'

        pid = subprocess.Popen(['ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                                'UserKnownHostsFile=/dev/null',
                                host, 'cd %s && %s' % (os.getcwd(), command)])

    return pid
