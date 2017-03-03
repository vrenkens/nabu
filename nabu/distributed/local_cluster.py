'''@file main.py
this function is used to run distributed training on a local cluster'''

import atexit
import subprocess
import tensorflow as tf


def local_cluster(expdir, class_type):
    '''main function'''

    #read the cluster file
    clusterfile = expdir + '/clusterfile'
    machines = dict()
    machines['worker'] = []
    machines['ps'] = []
    with open(clusterfile) as fid:
        for line in fid:
            if len(line.strip()) > 0:
                split = line.strip().split(',')
                machines[split[0]].append(
                    (split[1], int(split[2]), split[3]))

    #start all the jobs
    processes = []
    for job in machines:
        task_index = 0
        for _ in machines[job]:
            processes.append(subprocess.Popen(
                ['python', 'train_%s.py' % class_type,
                 '--clusterfile=%s' % clusterfile,
                 '--job_name=%s' % job, '--task_index=%d' % task_index,
                 '--ssh_command=None', '--expdir=%s' % expdir]))
            task_index += 1

    for process in processes:
        atexit.register(process.terminate)

    for process in processes:
        process.wait()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
    tf.app.flags.DEFINE_string('type', 'asr',
                               'one of asr or lm, the training type')
    FLAGS = tf.app.flags.FLAGS

    local_cluster(FLAGS.expdir, FLAGS.type)
