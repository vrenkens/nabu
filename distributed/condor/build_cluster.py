'''@file build_cluster.py
this file will be ran for all machines to build the cluster'''

import os
import socket
from time import sleep
from train import train
import tensorflow as tf

def main():
    '''main function'''

    cluster_dir = FLAGS.expdir + '/cluster'

    #report that this machine is available
    machine_index = 0
    machine_file = '%s/%s-%s-%d' % (cluster_dir, FLAGS.job_name,
                                    socket.gethostname(), machine_index)

    while os.path.exists(machine_file):
        machine_index += 1
        machine_file = '%s/%s-%s-%d' % (cluster_dir, FLAGS.job_name,
                                        socket.gethostname(), machine_index)

    fid = open(machine_file, 'w')
    fid.close()

    #wait untill the main process has given a go
    print 'waiting for cluster to be ready...'

    #read the task_index in the created file
    task_index = ''
    while len(task_index) == 0 and not os.path.exists(cluster_dir + '/ready'):
        with open(machine_file) as fid:
            task_index = fid.read()
        sleep(1)

    print 'cluster is ready'

    print 'task index is %s' % task_index

    #start the training process
    train(clusterfile=cluster_dir + '/cluster',
          job_name=FLAGS.job_name,
          task_index=int(task_index),
          expdir=FLAGS.expdir)

    #delete the file to notify that the porcess has finished
    os.remove(machine_file)

if __name__ == '__main__':

    tf.app.flags.DEFINE_string('job_name', None, 'One of ps, worker')
    tf.app.flags.DEFINE_string('expdir', '.',
                               'the experimental directory')
    FLAGS = tf.app.flags.FLAGS

    tf.app.run()
