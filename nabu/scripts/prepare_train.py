'''@file main.py
this is the file that should be run for experiments'''

import sys
import os
sys.path.append(os.getcwd())
import socket
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

def main(expdir, recipe, mode, computing):
    '''main function'''

    if expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(recipe):
        raise Exception('cannot find recipe %s' % recipe)
    if mode not in ['non_distributed', 'single_machine', 'multi_machine']:
        raise Exception('unknown distributed mode: %s' % mode)
    if computing not in ['standard', 'condor']:
        raise Exception('unknown computing mode: %s' % computing)

    database_cfg_file = os.path.join(recipe, 'database.conf')
    model_cfg_file = os.path.join(recipe, 'model.cfg')
    trainer_cfg_file = os.path.join(recipe, 'trainer.cfg')
    evaluator_cfg_file = os.path.join(recipe, 'validation_evaluator.cfg')

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(trainer_cfg_file)
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    if os.path.isdir(os.path.join(expdir, 'processes')):
        shutil.rmtree(os.path.join(expdir, 'processes'))
    os.makedirs(os.path.join(expdir, 'processes'))

    if trainer_cfg['resume_training'] == 'True':
        if not os.path.isdir(expdir):
            raise Exception('cannot find %s, please set resume_training to '
                            'False if you want to start a new training process'
                            % expdir)
    else:
        if os.path.isdir(os.path.join(expdir, 'logdir')):
            shutil.rmtree(os.path.join(expdir, 'logdir'))
        if not os.path.isdir(expdir):
            os.makedirs(expdir)
        if os.path.isdir(os.path.join(expdir, 'model')):
            shutil.rmtree(os.path.join(expdir, 'model'))
        os.makedirs(os.path.join(expdir, 'model'))

        #copy the configs to the expdir so they can be read there and the
        #experiment information is stored

        shutil.copyfile(database_cfg_file,
                        os.path.join(expdir, 'database.conf'))
        shutil.copyfile(model_cfg_file,
                        os.path.join(expdir, 'model.cfg'))
        shutil.copyfile(evaluator_cfg_file,
                        os.path.join(expdir, 'validation_evaluator.cfg'))

    shutil.copyfile(trainer_cfg_file, os.path.join(expdir, 'trainer.cfg'))

    computing_cfg_file = 'config/computing/%s/%s.cfg' % (computing,
                                                         mode)

    if computing == 'standard':

        if mode == 'non_distributed':

            train(clusterfile=None,
                  job_name='local',
                  task_index=0,
                  ssh_command='None',
                  expdir=expdir)

        elif mode == 'single_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            #create the directories
            if os.path.isdir(os.path.join(expdir, 'cluster')):
                shutil.rmtree(os.path.join(expdir, 'cluster'))
            os.makedirs(os.path.join(expdir, 'cluster'))

            GPUs = computing_cfg['gpus'].split(' ')

            #create the cluster file
            with open(os.path.join(expdir, 'cluster', 'cluster'),
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
                    fid.write('worker,localhost,%d,%s\n' % (port, GPUs[i]))
                    port += 1

            #start the training
            local_cluster.local_cluster(expdir)

        elif mode == 'multi_machine':

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
                        hostip = socket.gethostbyname(split[1])
                        machines[split[0]].append(hostip)

            #create the outputs directory
            if os.path.isdir(os.path.join(expdir, 'cluster')):
                shutil.rmtree(os.path.join(expdir, 'cluster'))
            os.makedirs(os.path.join(expdir, 'cluster'))

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
                                   expdir)
                    processes[job].append(run_remote.run_remote(
                        command=command,
                        host=machine
                        ))
                    task_index += 1

            #make sure the created processes are terminated at exit
            for job in processes:
                for process in processes[job]:
                    atexit.register(cond_term, process=process)

            #make sure all remotely created processes are terminated at exit
            atexit.register(kill_processes.kill_processes,
                            processdir=os.path.join(expdir, 'processes'))

            #wait for all worker processes to finish
            for process in processes['worker']:
                process.wait()

        else:
            raise Exception('unknown mode %s' % mode)

    elif computing == 'condor':

        if not os.path.isdir(os.path.join(expdir, 'outputs')):
            os.makedirs(os.path.join(expdir, 'outputs'))

        if mode == 'non_distributed':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            subprocess.call(['condor_submit', 'expdir=%s' % expdir,
                             'script=nabu/scripts/train.py',
                             'memory=%s' % computing_cfg['minmemory'],
                             'nabu/computing/condor/non_distributed.job'])

        elif mode == 'single_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            if os.path.isdir(os.path.join(expdir, 'cluster')):
                shutil.rmtree(os.path.join(expdir, 'cluster'))
            os.makedirs(os.path.join(expdir, 'cluster'))

            #create the cluster file
            with open(os.path.join(expdir, 'cluster', 'cluster'),
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
            subprocess.call(['condor_submit', 'expdir=%s' % expdir,
                             'GPUs=%d' % (int(computing_cfg['numworkers'])),
                             'memory=%s' % computing_cfg['minmemory'],
                             'nabu/computing/condor/local.job'])

            print ('job submitted look in %s/outputs for the job outputs' %
                   expdir)

        elif mode == 'multi_machine':

            #read the computing config file
            parsed_computing_cfg = configparser.ConfigParser()
            parsed_computing_cfg.read(computing_cfg_file)
            computing_cfg = dict(parsed_computing_cfg.items('computing'))

            if os.path.isdir(os.path.join(expdir, 'cluster')):
                shutil.rmtree(os.path.join(expdir, 'cluster'))
            os.makedirs(os.path.join(expdir, 'cluster'))

            #submit the parameter server jobs
            subprocess.call(['condor_submit', 'expdir=%s' % expdir,
                             'numjobs=%s' % computing_cfg['numps'],
                             'ssh_command=%s' % computing_cfg['ssh_command'],
                             'nabu/computing/condor/ps.job'])

            #submit the worker jobs
            subprocess.call(['condor_submit', 'expdir=%s' % expdir,
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
                        os.path.join(expdir, 'cluster'))

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
                if not machines['worker'] or not machines['ps']:

                    #stop the ps jobs
                    cidfile = os.path.join(expdir, 'cluster', 'ps-cid')
                    if os.path.exists(cidfile):
                        with open(cidfile) as fid:
                            cid = fid.read()
                        subprocess.call(['condor_rm', cid])

                    #stop the worker jobs
                    cidfile = os.path.join(expdir, 'cluster',
                                           'worker-cid')
                    if os.path.exists(cidfile):
                        with open(cidfile) as fid:
                            cid = fid.read()
                        subprocess.call(['condor_rm', cid])

                    raise Exception('at leat one ps and one worker needed')


            print ('starting training with %s parameter servers and %s workers'
                   % (len(machines['ps']), len(machines['worker'])))

            #create the cluster file
            with open(os.path.join(expdir, 'cluster', 'cluster'),
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
            open(os.path.join(expdir, 'cluster', 'ready'), 'w').close()

            print ('training has started look in %s/outputs for the job outputs'
                   % expdir)

        else:
            raise Exception('unknown mode %s' % mode)
    else:
        raise Exception('Unknown computing type %s' % computing)

if __name__ == '__main__':
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
    tf.app.flags.DEFINE_string('computing', 'standard',
                               'the distributed computing system one of'
                               ' condor'
                              )

    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.expdir, FLAGS.recipe, FLAGS.mode, FLAGS.computing)

def cond_term(process):
    '''terminate pid if it exists'''

    try:
        os.kill(process.terminate)
    #pylint: disable=W0702
    except:
        pass
