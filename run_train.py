'''@file main.py
this is the file that should be run for experiments'''

import os
import shutil
import atexit
import subprocess
from time import sleep
import tensorflow as tf
from six.moves import configparser
from nabu.distributed import cluster, local_cluster
from nabu.distributed.static import run_remote
from nabu.distributed.static import kill_processes
from train_asr import train_asr
from train_lm import train_lm

tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
tf.app.flags.DEFINE_string('type', 'asr', 'one of asr or lm, the training type')
FLAGS = tf.app.flags.FLAGS
if FLAGS.type not in ['asr', 'lm']:
    raise Exception('type shoud be on of asr or lang, received %s' % FLAGS.type)

def main(_):
    '''main function'''

    #pointers to the config files
    computing_cfg_file = 'config/computing/non_distributed.cfg'
    database_cfg_file = 'config/asr_databases/TIMIT.conf'
    if FLAGS.type == 'asr':
        feat_cfg_file = 'config/features/fbank.cfg'
    classifier_cfg_file = 'config/asr/LAS.cfg'
    trainer_cfg_file = 'config/trainer/cross_entropytrainer.cfg'
    decoder_cfg_file = 'config/decoder/BeamSearchDecoder.cfg'

    #read the computing config file
    parsed_computing_cfg = configparser.ConfigParser()
    parsed_computing_cfg.read(computing_cfg_file)
    computing_cfg = dict(parsed_computing_cfg.items('computing'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(trainer_cfg_file)
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    if os.path.isdir(os.path.join(FLAGS.expdir, 'processes')):
        shutil.rmtree(os.path.join(FLAGS.expdir, 'processes'))
    os.makedirs(os.path.join(FLAGS.expdir, 'processes'))

    if trainer_cfg['resume_training'] == 'True':
        if not os.path.isdir(FLAGS.expdir):
            raise Exception('cannot find %s, please set resume_training to '
                            'False if you want to start a new training process'
                            % FLAGS.expdir)
    else:
        if os.path.isdir(os.path.join(FLAGS.expdir, 'logdir')):
            shutil.rmtree(os.path.join(FLAGS.expdir, 'logdir'))

        if not os.path.isdir(FLAGS.expdir):
            os.makedirs(FLAGS.expdir)

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'model')):
            os.makedirs(os.path.join(FLAGS.expdir, 'model'))

        #copy the configs to the expdir so they can be read there and the
        #experiment information is stored
        shutil.copyfile(database_cfg_file,
                        os.path.join(FLAGS.expdir, 'database.cfg'))
        if FLAGS.type == 'asr':
            shutil.copyfile(feat_cfg_file,
                            os.path.join(FLAGS.expdir, 'model', 'features.cfg'))
        shutil.copyfile(classifier_cfg_file,
                        os.path.join(FLAGS.expdir, 'model',
                                     '%s.cfg' % FLAGS.type))


    shutil.copyfile(computing_cfg_file,
                    os.path.join(FLAGS.expdir, 'computing.cfg'))
    shutil.copyfile(trainer_cfg_file, os.path.join(FLAGS.expdir, 'trainer.cfg'))
    shutil.copyfile(decoder_cfg_file,
                    os.path.join(FLAGS.expdir, 'model', 'decoder.cfg'))

    if computing_cfg['distributed'] == 'condor_non-distributed':

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))

        subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                         'memory=%s' % computing_cfg['minmemory'],
                         'type=%s' % FLAGS.type,
                         'nabu/distributed/condor/non_distributed.job'])

    elif computing_cfg['distributed'] == 'non-distributed':

        if FLAGS.type == 'asr':
            train_asr(clusterfile=None,
                      job_name='local',
                      task_index=0,
                      ssh_command='None',
                      expdir=FLAGS.expdir)
        else:
            train_lm(clusterfile=None,
                     job_name='local',
                     task_index=0,
                     ssh_command='None',
                     expdir=FLAGS.expdir)

    elif computing_cfg['distributed'] == 'local':

        #create the directories
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

        #create the cluster file
        with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'), 'w') as fid:
            port = 1024
            for _ in range(int(computing_cfg['numps'])):
                while not cluster.port_available(port):
                    port += 1
                fid.write('ps,localhost,%d,\n' % port)
                port += 1
            for i in range(int(computing_cfg['numworkers'])):
                while not cluster.port_available(port):
                    port += 1
                fid.write('worker,localhost,%d,%d\n' % (port, i))
                port += 1

        #start the training
        local_cluster.local_cluster(FLAGS.expdir, FLAGS.type)

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

        #create the outputs directory
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))

        #run all the jobs
        processes = dict()
        processes['worker'] = []
        processes['ps'] = []
        for job in machines:
            task_index = 0
            for machine in machines[job]:
                command = ('python -u train_%s.py --clusterfile=%s '
                           '--job_name=%s --task_index=%d --ssh_command=%s '
                           '--expdir=%s') % (
                               FLAGS.type, computing_cfg['clusterfile'], job,
                               task_index, computing_cfg['ssh_command'],
                               FLAGS.expdir)
                processes[job].append(run_remote.run_remote(
                    command=command,
                    host=machine
                    ))
                task_index += 1

        #make sure the created processes are terminated at exit
        for job in processes:
            for process in processes[job]:
                atexit.register(process.terminate)

        #make sure all remotely created processes are terminated at exit
        atexit.register(kill_processes.kill_processes,
                        processdir=os.path.join(FLAGS.expdir, 'processes'))

        #wait for all worker processes to finish
        for process in processes['worker']:
            process.wait()

    elif computing_cfg['distributed'] == 'condor':

        #create the directories
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))
        if os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
            shutil.rmtree(os.path.join(FLAGS.expdir, 'cluster'))
        os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

        #submit the parameter server jobs
        subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                         'numjobs=%s' % computing_cfg['numps'],
                         'type=%s' % FLAGS.type,
                         'ssh_command=%s' % computing_cfg['ssh_command'],
                         'nabu/distributed/condor/ps.job'])

        #submit the worker jobs
        subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                         'numjobs=%s' % computing_cfg['numworkers'],
                         'memory=%s' % computing_cfg['minmemory'],
                         'type=%s' % FLAGS.type,
                         'ssh_command=%s' % computing_cfg['ssh_command'],
                         'nabu/distributed/condor/worker.job'])

        ready = False

        try:
            print 'waiting for the machines to report...'
            numworkers = 0
            numps = 0
            while not ready:
                #check the machines in the cluster
                machines = cluster.get_machines(
                    os.path.join(FLAGS.expdir, 'cluster'))

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

                #stop the ps jobs
                cidfile = os.path.join(FLAGS.expdir, 'cluster', 'ps-cid')
                if os.path.exists(cidfile):
                    with open(cidfile) as fid:
                        cid = fid.read()
                    subprocess.call(['condor_rm', cid])

                #stop the worker jobs
                cidfile = os.path.join(FLAGS.expdir, 'cluster', 'worker-cid')
                if os.path.exists(cidfile):
                    with open(cidfile) as fid:
                        cid = fid.read()
                    subprocess.call(['condor_rm', cid])

                raise Exception('at leat one ps and one worker needed')


        print ('starting training with %s parameter servers and %s workers' %
               (len(machines['ps']), len(machines['worker'])))

        #create the cluster file
        with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'),
                  'w') as cfid:
            for job in machines:
                if job == 'ps':
                    GPU = ''
                else:
                    GPU = '0'
                for machine in machines[job]:
                    cfid.write('%s,%s,%d,%s\n' % (job, machine[0], machine[1],
                                                  GPU))

        #notify the machine that the cluster is ready
        fid = open(FLAGS.expdir + '/cluster/ready', 'w')
        fid.close()

        print ('training has started look in %s/outputs for the job outputs' %
               FLAGS.expdir)

        print 'waiting for worker jobs to finish'

        for machine in machines['worker']:
            machine_file = os.path.join(FLAGS.expdir, 'cluster',
                                        '%s-%d' % (machine[0], machine[1]))
            while os.path.exists(machine_file):
                sleep(1)

        #stop the ps jobs
        with open(os.path.join(FLAGS.expdir, 'cluster', 'ps-cid')) as fid:
            cid = fid.read()

        subprocess.call(['condor_rm', cid])

    elif computing_cfg['distributed'] == 'condor_local':

        #create the directories
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))
        if not os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))


        #create the cluster file
        with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'), 'w') as fid:
            port = 1024
            for _ in range(int(computing_cfg['numps'])):
                while not cluster.port_available(port):
                    port += 1
                fid.write('ps,localhost,%d,\n' % port)
                port += 1
            for i in range(int(computing_cfg['numworkers'])):
                while not cluster.port_available(port):
                    port += 1
                fid.write('worker,localhost,%d,%d\n' % (port, i))
                port += 1

        #submit the job
        subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                         'GPUs=%d' % (int(computing_cfg['numworkers'])),
                         'memory=%s' % computing_cfg['minmemory'],
                         'type=%s' % FLAGS.type,
                         'nabu/distributed/condor/local.job'])

        print ('job submitted look in %s/outputs for the job outputs' %
               FLAGS.expdir)


    else:
        raise Exception('Unknown distributed type in %s' % computing_cfg_file)

if __name__ == '__main__':
    tf.app.run()
