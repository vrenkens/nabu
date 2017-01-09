'''@file main.py
this is the file that should be run for experiments'''

import os
import shutil
import subprocess
import atexit
from time import sleep
import tensorflow as tf
from six.moves import configparser
from distributed import cluster
from train import train

tf.app.flags.DEFINE_string('expdir', '.', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''main function'''

    #pointers to the config files
    computing_cfg_file = 'config/computing/static.cfg'
    database_cfg_file = 'config/databases/TIMIT.cfg'
    feat_cfg_file = 'config/features/fbank.cfg'
    nnet_cfg_file = 'config/nnet/DBLSTM.cfg'
    trainer_cfg_file = 'config/trainer/CTCtrainer.cfg'
    decoder_cfg_file = 'config/decoder/CTCdecoder.cfg'

    #read the computing config file
    parsed_computing_cfg = configparser.ConfigParser()
    parsed_computing_cfg.read(computing_cfg_file)
    computing_cfg = dict(parsed_computing_cfg.items('computing'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(trainer_cfg_file)
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    if os.path.isdir(FLAGS.expdir) and trainer_cfg['resume_training'] != 'True':
        shutil.rmtree(FLAGS.expdir)

    if not os.path.isdir(FLAGS.expdir):
        os.mkdir(FLAGS.expdir)

    #copy the configs to the expdir so they can be read there and the experiment
    #information is stored
    shutil.copyfile(computing_cfg_file, FLAGS.expdir + '/computing.cfg')
    shutil.copyfile(database_cfg_file, FLAGS.expdir + '/database.cfg')
    shutil.copyfile(feat_cfg_file, FLAGS.expdir + '/features.cfg')
    shutil.copyfile(nnet_cfg_file, FLAGS.expdir + '/nnet.cfg')
    shutil.copyfile(trainer_cfg_file, FLAGS.expdir + '/trainer.cfg')
    shutil.copyfile(decoder_cfg_file, FLAGS.expdir + '/decoder.cfg')

    if computing_cfg['distributed'] == 'local':

        train(clusterfile=None,
              job_name=None,
              task_index=None,
              expdir=FLAGS.expdir)

    elif computing_cfg['distributed'] == 'static':

        #read the cluster file
        machines = dict()
        machines['worker'] = []
        machines['ps'] = []
        with open(computing_cfg['clusterfile']) as fid:
            for line in fid:
                if len(line.strip()) > 0:
                    split = line.strip().split(',')
                    machines[split[0]].append(split[1])

        #run all the jobs
        processes = []
        for job in machines:
            task_index = 0
            for machine in machines[job]:
                processes.append(subprocess.Popen(
                    ['distributed/static/run_remote.sh',
                     machine, computing_cfg['clusterfile'], job,
                     str(task_index), FLAGS.expdir],
                    stdout=open(
                        '%s/%s-%d' % (FLAGS.expdir, job, task_index), 'w', 0),
                    stderr=subprocess.STDOUT))
                atexit.register(processes[-1].terminate)
                task_index += 1

        #wait for all processes to finish
        for process in processes:
            process.wait()

    elif computing_cfg['distributed'] == 'condor':

        #create the directories
        if not os.path.isdir(FLAGS.expdir + '/condor'):
            os.mkdir(FLAGS.expdir + '/condor')
        if os.path.isdir(FLAGS.expdir + '/cluster'):
            shutil.rmtree(FLAGS.expdir + '/cluster')
        os.mkdir(FLAGS.expdir + '/cluster')

        #submit the parameter server jobs
        os.system('condor_submit expdir=%s numjobs=%s distributed/condor/ps.job'
                  % (FLAGS.expdir, computing_cfg['numps']))

        #submit the worker jobs
        os.system('condor_submit expdir=%s numjobs=%s '
                  'distributed/condor/worker.job' %
                  (FLAGS.expdir, computing_cfg['numworkers']))

        ready = False

        try:
            print 'waiting for the machines to report...'
            numworkers = 0
            numps = 0
            while not ready:
                #check the machines in the cluster
                machines = cluster.get_machines(FLAGS.expdir + '/cluster')

                if (len(machines['ps']) > numps
                        or len(machines['worker']) > numworkers):

                    numworkers = len(machines['worker'])
                    numps = len(machines['ps'])

                    print('parameter servers ready %d/%s' %
                          (len(machines['ps']), computing_cfg['numps']))

                    print('workers ready %d/%s' %
                          (len(machines['worker']),
                           computing_cfg['numworkers']))

                    print 'press Ctrl-C to run with the current machines'

                #check if the required amount of machines has reported
                if (len(machines['worker']) == int(computing_cfg['numworkers'])
                        and len(machines['ps']) == int(computing_cfg['numps'])):

                    ready = True

                sleep(1)

        except KeyboardInterrupt:
            #remove all jobs that are not running
            os.system('condor_rm -constraint \'JobStatus =!= 2\'')

            #check if enough machines are available
            if len(machines['worker']) == 0 or len(machines['ps']) == 0:
                os.system('condor_rm -all')
                raise Exception('at leat one ps and one worker needed')


        print ('starting training with %s parameter servers and %s workers' %
               (len(machines['ps']), len(machines['worker'])))

        #create the cluster file
        with open(FLAGS.expdir + '/cluster/cluster', 'w') as cfid:
            for job in machines:
                task_index = 0
                for machine in machines[job]:
                    with open(FLAGS.expdir + '/cluster/%s-%s'
                              % (job, machine), 'w') as fid:
                        fid.write(str(task_index))

                    cfid.write('%s,%s\n' % (job, machine))

        #notify the machine that the cluster is ready
        fid = open(FLAGS.expdir + '/cluster/ready', 'w')
        fid.close()

    else:
        raise Exception('Unknown distributed type in %s' % computing_cfg_file)

if __name__ == '__main__':
    tf.app.run()
