'''@file create_cluster.py
contains the create_cluster method'''

import os
import subprocess
import tensorflow as tf
from nabu.distributed import cluster

def create_cluster(clusterfile, job_name, task_index, expdir, ssh_tunnel):
    '''creates the tensorflow cluster and server based on the clusterfile

    Args:
        clusterfile: the path to the clusterfile
        job_name: the name of the job
        task_index: the task index
        expdir: the experiments directory
        ssh_tunnel: wheter or not communication should happen through an ssh
            tunnel

    Returns: a tensorflow cluster and server'''

    if clusterfile is None:
        #no distributed training
        tfcluster = None
        server = None
    else:
        #read the cluster file
        machines = dict()
        machines['worker'] = []
        machines['ps'] = []
        with open(clusterfile) as fid:
            for line in fid:
                if len(line.strip()) > 0:
                    split = line.strip().split(',')
                    machines[split[0]].append(
                        (split[1], int(split[2]), split[3]))

        #build the cluster and create ssh tunnels to machines in the cluster
        port = 1024
        clusterdict = dict()
        clusterdict['worker'] = []
        clusterdict['ps'] = []
        localmachine = machines[job_name][task_index][0]

        #report that this job is running
        open(os.path.join(
            expdir, 'processes', '%s-%d' % (localmachine, os.getpid()))
             , 'w').close()

        #specify the GPU that should be used
        localGPU = machines[job_name][task_index][2]
        os.environ['CUDA_VISIBLE_DEVICES'] = localGPU


        #get a list of ports used on this machine
        localports = []
        for job in machines:
            for remote in machines[job]:
                if localmachine == remote[0] or remote[0] == 'localhost':
                    localports.append(remote[1])

        for job in machines:
            for remote in machines[job]:

                #create an ssh tunnel if the local machine is not the same as
                #the remote machine
                if localmachine != remote[0] and ssh_tunnel:

                    #look for an available port
                    while (port in localports
                           or not cluster.port_available(port)):

                        port += 1

                    #create the ssh tunnel
                    p = subprocess.Popen(
                        ['ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                         'UserKnownHostsFile=/dev/null', '-L',
                         '%d:localhost:%d' % (port, remote[1]), '-N',
                         remote[0]])

                    #report that the ssh tunnel is running
                    open(os.path.join(
                        expdir, 'processes', '%s-%d' % (localmachine, p.pid)),
                         'w').close()

                    #add the machine to the cluster
                    clusterdict[job].append('localhost:%d' % port)

                    port += 1

                else:
                    clusterdict[job].append('%s:%d' % (remote[0], remote[1]))

        #create the cluster
        tfcluster = tf.train.ClusterSpec(clusterdict)

        #create the server for this task
        server = tf.train.Server(tfcluster, job_name, task_index)

    return tfcluster, server
