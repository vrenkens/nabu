'''@file create_cluster.py
contains the create_cluster method'''

import os
import subprocess
from time import sleep
import tensorflow as tf
from nabu.distributed import cluster

def create_server(clusterfile, job_name, task_index, expdir, ssh_command):
    '''creates the tensorflow cluster and server based on the clusterfile

    Args:
        clusterfile: the path to the clusterfile
        job_name: the name of the job
        task_index: the task index
        expdir: the experiments directory
        ssh_command: the command to use for ssh, if 'None' no tunnel will be
            created

    Returns: a tensorflow server'''

    if clusterfile is None:
        #no distributed training
        server = tf.train.Server.create_local_server()
    else:
        #read the cluster file
        machines = cluster.read_cluster(clusterfile)


        #build the cluster and create ssh tunnels to machines in the cluster
        port = 1024
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

        #check if this task is the first one in the cluster
        First = True
        if job_name == 'worker':
            for machine in machines['ps'] + machines['worker'][:task_index]:
                First = First and not machine[0] == localmachine
        else:
            for machine in machines['ps'][:task_index]:
                First = First and not machine[0] == localmachine

        #the first task on a machine will create the cluster for this machine
        machinecluster = os.path.join(expdir, 'cluster',
                                      '%s-cluster' % localmachine)
        readyfile = os.path.join(expdir, 'cluster',
                                 '%s-ready' % localmachine)

        if First:
            with open(machinecluster, 'w') as fid:
                for job in machines:
                    for remote in machines[job]:

                        #create an ssh tunnel if the local machine is not the
                        #same as the remote machine
                        if localmachine != remote[0] and ssh_command != 'None':

                            #look for an available port
                            while (port in localports
                                   or not cluster.port_available(port)):

                                port += 1

                            #create the ssh tunnel
                            p = subprocess.Popen(
                                [ssh_command, '-o', 'StrictHostKeyChecking=no',
                                 '-o', 'UserKnownHostsFile=/dev/null', '-L',
                                 '%d:127.0.0.1:%d' % (port, remote[1]), '-N',
                                 remote[0]])

                            #report that the ssh tunnel is running
                            open(os.path.join(
                                expdir, 'processes',
                                '%s-%d' % (localmachine, p.pid)), 'w').close()

                            fid.write('%s,localhost,%s,%s\n' % (job, port,
                                                                remote[2]))

                            port += 1

                            #give the machine some time to open the ssh tunnel
                            #before opening a new one
                            sleep(0.1)

                        else:
                            if localmachine == remote[0]:
                                host = 'localhost'
                            else:
                                host = remote[0]
                            fid.write('%s,%s,%s,%s\n' % (job, host,
                                                         remote[1], remote[2]))

            #notify that the cluster is ready
            open(readyfile, 'w').close()

        #wait for the clusterfile to be ready
        while not os.path.exists(readyfile):
            sleep(1)

        #read the cluster file
        machines = cluster.read_cluster(machinecluster)

        clusterdict = dict()
        clusterdict['worker'] = []
        clusterdict['ps'] = []
        for job in machines:
            for remote in machines[job]:
                clusterdict[job].append('%s:%d' % (remote[0], remote[1]))

        #create the cluster
        tfcluster = tf.train.ClusterSpec(clusterdict)

        #create the server for this task
        server = tf.train.Server(tfcluster, job_name, task_index)

    return server
