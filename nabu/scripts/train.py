'''@file train.py
this file will do the training'''

import sys
import os
sys.path.append(os.getcwd())
import tensorflow as tf
from six.moves import configparser
from nabu.computing import create_server
from nabu.neuralnetworks.trainers import trainer_factory
from nabu.neuralnetworks.trainers import trainer

def train(clusterfile,
          job_name,
          task_index,
          ssh_command,
          expdir,
          testing=False):

    ''' does everything for asr training

    Args:
        clusterfile: the file where all the machines in the cluster are
            specified if None, local training will be done
        job_name: one of ps or worker in the case of distributed training
        task_index: the task index in this job
        ssh_command: the command to use for ssh, if 'None' no tunnel will be
            created
        expdir: the experiments directory
        testing: if true only the graph will be created for debugging purposes
    '''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.conf'))

    #read the asr config file
    model_cfg = configparser.ConfigParser()
    model_cfg.read(os.path.join(expdir, 'model.cfg'))

    #read the trainer config file
    trainer_cfg = configparser.ConfigParser()
    trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))

    #read the decoder config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'validation_evaluator.cfg'))

    #create the cluster and server
    server = create_server.create_server(
        clusterfile=clusterfile,
        job_name=job_name,
        task_index=task_index,
        expdir=expdir,
        ssh_command=ssh_command)

    #parameter server
    if job_name == 'ps':

        print 'starting parameter server'

        #create the parameter server
        ps = trainer.ParameterServer(
            conf=trainer_cfg,
            modelconf=model_cfg,
            dataconf=database_cfg,
            server=server,
            task_index=task_index)

        #let the ps wait untill all workers are finished
        ps.join()

        print 'parameter server stopped'

        return

    #create the trainer
    tr = trainer_factory.factory(trainer_cfg.get('trainer', 'trainer'))(
        conf=trainer_cfg,
        dataconf=database_cfg,
        modelconf=model_cfg,
        evaluatorconf=evaluator_cfg,
        expdir=expdir,
        server=server,
        task_index=task_index)

    print 'starting training'

    #train the model
    tr.train(testing)

if __name__ == '__main__':

    #define the FLAGS
    tf.app.flags.DEFINE_string('clusterfile', None,
                               'The file containing the cluster')
    tf.app.flags.DEFINE_string('job_name', 'local', 'One of local, ps, worker')
    tf.app.flags.DEFINE_integer('task_index', 0, 'The task index')
    tf.app.flags.DEFINE_string(
        'ssh_command', 'None',
        'the command that should be used to create ssh tunnels')
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experimental directory')

    FLAGS = tf.app.flags.FLAGS

    train(
        clusterfile=FLAGS.clusterfile,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        ssh_command=FLAGS.ssh_command,
        expdir=FLAGS.expdir,
        testing=False)
