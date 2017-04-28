'''@file build_cluster.py
this file will be ran for all machines to build the cluster'''

import os
import socket
from time import sleep
from nabu.computing import cluster
from nabu.scripts.train import train
import tensorflow as tf

def main(_):
    '''main function'''
    cluster_dir = os.path.join(FLAGS.expdir, 'cluster')

    #the chief of the job should write the cluster id
    if int(FLAGS.pid) == 0:
        with open(os.path.join(cluster_dir, '%s-cid' % FLAGS.job_name)
                  , 'w') as fid:
            fid.write(FLAGS.cid)

    #wait for the preceeding cluster tasks to report
    machines = cluster.get_machines(cluster_dir)

    while len(machines[FLAGS.job_name]) < int(FLAGS.pid):
        machines = cluster.get_machines(cluster_dir)
        sleep(1)

    port = 1024
    machine_file = '%s/%s-%d' % (cluster_dir, socket.gethostname(), port)

    #look for an available port
    while os.path.exists(machine_file) or not cluster.port_available(port):

        port += 1
        machine_file = '%s/%s-%d' % (cluster_dir, socket.gethostname(), port)

    #report that the machine is ready
    with open(machine_file, 'w') as fid:
        fid.write(FLAGS.job_name)

    #wait untill the main process has given a go
    print 'waiting for cluster to be ready...'

    #read the task_index in the created file
    while not os.path.exists(cluster_dir + '/ready'):
        sleep(1)

    print 'cluster is ready'

    #start the training process
    train(clusterfile=cluster_dir + '/cluster',
          job_name=FLAGS.job_name,
          task_index=int(FLAGS.pid),
          ssh_command=FLAGS.ssh_command,
          expdir=FLAGS.expdir)

    #delete the file to notify that the porcess has finished
    os.remove(machine_file)

if __name__ == '__main__':

    tf.app.flags.DEFINE_string('job_name', None, 'One of ps, worker')
    tf.app.flags.DEFINE_string('expdir', '.',
                               'the experimental directory')
    tf.app.flags.DEFINE_string('pid', '0',
                               'the process id of this job')
    tf.app.flags.DEFINE_string('cid', '0',
                               'the cluster id of this job')
    tf.app.flags.DEFINE_string(
        'ssh_command', 'None',
        'the command that should be used to create ssh tunnels')
    FLAGS = tf.app.flags.FLAGS

    tf.app.run()
