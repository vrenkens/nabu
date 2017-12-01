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
        with open(os.path.join(machine_dir, f)) as fid:
            job_name = fid.read()
        splitfile = f.split('-')
        if job_name not in ['ps', 'worker']:
            continue
        machines[job_name].append(
            (socket.gethostbyname(splitfile[0]), int(splitfile[1])))

    return machines

def read_cluster(filename):
    '''read a cluster file

    Args:
        filename: the cluster file

    Returns:
        a dictionary with ps and worker containing a list of machines'''

    machines = dict()
    machines['worker'] = []
    machines['ps'] = []

    with open(filename) as fid:
        for line in fid:
            if line.strip() > 0:
                split = line.strip().split(',')
                machines[split[0]].append(
                    (split[1], int(split[2]), split[3]))

    return machines

def port_available(port):
    '''check if port is available'''

    sock = socket.socket()
    result = sock.connect_ex(('localhost', port))
    return not result == 0
