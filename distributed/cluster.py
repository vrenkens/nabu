'''@file cluster.py
contains functionality for creating a cluster'''

import os
import socket

def get_machines(machine_dir):
    ''' gets the machines that reported in the machine directory

    Args:
        machine_dir: the directory where the machines are reporting

    Returns: a dictionary with keys ps and worker and lists containing the
        machines as elements
    '''

    files = os.listdir(machine_dir)
    machines = dict()
    machines['worker'] = []
    machines['ps'] = []

    for f in files:
        with open(f) as fid:
            job_name = fid.read()
        splitfile = f.split('-')
        if job_name not in ['ps', 'worker']:
            continue
        machines[job_name].append((splitfile[0], int(splitfile[1])))

    return machines

def port_available(port):
    '''check if port is available'''

    sock = socket.socket()
    result = sock.connect_ex(('localhost', port))
    return not result == 0
