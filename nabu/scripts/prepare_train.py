'''@file main.py
this is the file that should be run for experiments'''

import sys
import os
sys.path.append(os.getcwd())
import shutil
import atexit
import subprocess
from time import sleep
import tensorflow as tf
from six.moves import configparser
from nabu.computing import cluster, local_cluster
from nabu.computing.static import run_remote
from nabu.computing.static import kill_processes
from train import train

tf.app.flags.DEFINE_string('expdir', None,
                           'the exeriments directory'
                          )
tf.app.flags.DEFINE_string('recipe', None,
                           'The directory containing the recipe'
                          )
tf.app.flags.DEFINE_string('mode', 'non_distributed',
                           'The computing mode, one of non_distributed, '
                           'single_machine or multi_machine'
                          )
tf.app.flags.DEFINE_string('computing', 'standart',
                           'the distributed computing system one of standart or'
                           ' condor'
                          )

FLAGS = tf.app.flags.FLAGS

def main(_):
    '''main function'''

    if FLAGS.expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if FLAGS.recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(FLAGS.recipe):
        raise Exception('cannot find recipe %s' % FLAGS.recipe)

    database_cfg_file = os.path.join(FLAGS.recipe, 'database.conf')
    model_cfg_file = os.path.join(FLAGS.recipe, 'model.cfg')
    trainer_cfg_file = os.path.join(FLAGS.recipe, 'trainer.cfg')
    evaluator_cfg_file = os.path.join(FLAGS.recipe, 'validation_evaluator.cfg')

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
        if os.path.isdir(os.path.join(FLAGS.expdir, 'model')):
            shutil.rmtree(os.path.join(FLAGS.expdir, 'model'))
        os.makedirs(os.path.join(FLAGS.expdir, 'model'))

        #copy the configs to the expdir so they can be read there and the
        #experiment information is stored

        shutil.copyfile(database_cfg_file,
                        os.path.join(FLAGS.expdir, 'database.cfg'))
        shutil.copyfile(model_cfg_file,
                        os.path.join(FLAGS.expdir, 'model.cfg'))
        shutil.copyfile(evaluator_cfg_file,
                        os.path.join(FLAGS.expdir, 'evaluator.cfg'))

    shutil.copyfile(trainer_cfg_file, os.path.join(FLAGS.expdir, 'trainer.cfg'))

    computing_cfg_file = 'config/computing/%s/%s.cfg' % (FLAGS.computing,
                                                         FLAGS.mode)

    if FLAGS.computing == 'standart':

        if FLAGS.mode == 'non_distributed':

            train(clusterfile=None,
                  job_name='local',
                  task_index=0,
                  ssh_command='None',
                  expdir=FLAGS.expdir)

        elif FLAGS.mode == 'single_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            #create the directories
            if os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
                shutil.rmtree(os.path.join(FLAGS.expdir, 'cluster'))
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

            #create the cluster file
            with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'),
                      'w') as fid:
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
            local_cluster.local_cluster(FLAGS.expdir)

        elif FLAGS.mode == 'multi_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            #read the cluster file
            machines = dict()
            machines['worker'] = []
            machines['ps'] = []
            with open(computing_cfg['clusterfile']) as fid:
                for line in fid:
                    if line.strip():
                        split = line.strip().split(',')
                        machines[split[0]].append(split[1])

            #create the outputs directory
            if os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
                shutil.rmtree(os.path.join(FLAGS.expdir, 'cluster'))
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

            #run all the jobs
            processes = dict()
            processes['worker'] = []
            processes['ps'] = []
            for job in machines:
                task_index = 0
                for machine in machines[job]:
                    command = ('python -u nabu/scripts/train.py '
                               '--clusterfile=%s '
                               '--job_name=%s --task_index=%d --ssh_command=%s '
                               '--expdir=%s') % (
                                   computing_cfg['clusterfile'], job,
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

        else:
            raise Exception('unknown mode %s' % FLAGS.mode)

    elif FLAGS.computing == 'condor':

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'outputs'))

        if FLAGS.mode == 'non_distributed':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                             'script=nabu/scripts/train.py',
                             'memory=%s' % computing_cfg['minmemory'],
                             'nabu/computing/condor/non_distributed.job'])

        elif FLAGS.mode == 'single_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            if os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
                shutil.rmtree(os.path.join(FLAGS.expdir, 'cluster'))
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

            #create the cluster file
            with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'),
                      'w') as fid:
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
                             'nabu/computing/condor/local.job'])

            print ('job submitted look in %s/outputs for the job outputs' %
                   FLAGS.expdir)

        elif FLAGS.mode == 'multi_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            if os.path.isdir(os.path.join(FLAGS.expdir, 'cluster')):
                shutil.rmtree(os.path.join(FLAGS.expdir, 'cluster'))
            os.makedirs(os.path.join(FLAGS.expdir, 'cluster'))

            #submit the parameter server jobs
            subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                             'numjobs=%s' % computing_cfg['numps'],
                             'ssh_command=%s' % computing_cfg['ssh_command'],
                             'nabu/computing/condor/ps.job'])

            #submit the worker jobs
            subprocess.call(['condor_submit', 'expdir=%s' % FLAGS.expdir,
                             'numjobs=%s' % computing_cfg['numworkers'],
                             'memory=%s' % computing_cfg['minmemory'],
                             'ssh_command=%s' % computing_cfg['ssh_command'],
                             'nabu/computing/condor/worker.job'])

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
                    if (len(machines['worker']) ==
                            int(computing_cfg['numworkers'])
                            and len(machines['ps'])
                            == int(computing_cfg['numps'])):

                        ready = True

                    sleep(1)

            except KeyboardInterrupt:

                #remove all jobs that are not running
                os.system('condor_rm -constraint \'JobStatus =!= 2\'')

                #check if enough machines are available
                if machines['worker'] or machines['ps']:

                    #stop the ps jobs
                    cidfile = os.path.join(FLAGS.expdir, 'cluster', 'ps-cid')
                    if os.path.exists(cidfile):
                        with open(cidfile) as fid:
                            cid = fid.read()
                        subprocess.call(['condor_rm', cid])

                    #stop the worker jobs
                    cidfile = os.path.join(FLAGS.expdir, 'cluster',
                                           'worker-cid')
                    if os.path.exists(cidfile):
                        with open(cidfile) as fid:
                            cid = fid.read()
                        subprocess.call(['condor_rm', cid])

                    raise Exception('at leat one ps and one worker needed')


            print ('starting training with %s parameter servers and %s workers'
                   % (len(machines['ps']), len(machines['worker'])))

            #create the cluster file
            with open(os.path.join(FLAGS.expdir, 'cluster', 'cluster'),
                      'w') as cfid:
                for job in machines:
                    if job == 'ps':
                        GPU = ''
                    else:
                        GPU = '0'
                    for machine in machines[job]:
                        cfid.write('%s,%s,%d,%s\n' % (job, machine[0],
                                                      machine[1], GPU))

            #notify the machine that the cluster is ready
            open(os.path.join(FLAGS.expdir, 'cluster', 'ready'), 'w').close()

            print ('training has started look in %s/outputs for the job outputs'
                   % FLAGS.expdir)

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

        else:
            raise Exception('unknown mode %s' % FLAGS.mode)
    else:
        raise Exception('Unknown computing type %s' % FLAGS.computing)

if __name__ == '__main__':
    tf.app.run()
