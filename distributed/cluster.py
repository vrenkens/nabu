'''@file cluster.py
contains functionality for creating a cluster'''

import os

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
        splitfile = f.split('-')
        if splitfile[0] not in ['ps', 'worker']:
            continue
        machines[splitfile[0]].append(splitfile[1])

    return machines
