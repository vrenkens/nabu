'''@file run_remote.py
contains the run_remote function'''

import os
import subprocess

def run_remote(host, command, out):
    '''run a command remotely over ssh

    Args:
        host: the host where the command should be ran
        command: the command that should be executed
        out: the file where the output should be written

    Returns:
        a popen process'''

    pid = subprocess.Popen(['ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                            'UserKnownHostsFile=/dev/null',
                            host, 'cd %s && %s' % (os.getcwd(), command)],
                           stdout=out,
                           stderr=subprocess.STDOUT)

    return pid
