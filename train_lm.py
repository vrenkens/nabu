'''@file train_lm.py
this file will do the asr training'''

import os
import shutil
from functools import partial
import tensorflow as tf
from six.moves import configparser
from nabu.distributed import create_cluster
from nabu.processing import batchdispenser, text_reader, target_coder
from nabu.neuralnetworks.classifiers.lm import lm_factory
from nabu.neuralnetworks.trainers import trainer_factory
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.neuralnetworks.trainers import trainer

def train_lm(clusterfile,
             job_name,
             task_index,
             ssh_tunnel,
             expdir):

    ''' does everything for language model training

    Args:
        clusterfile: the file where all the machines in the cluster are
            specified if None, local training will be done
        job_name: one of ps or worker in the case of distributed training
        task_index: the task index in this job
        ssh_tunnel: wheter or not communication should happen through an ssh
            tunnel
        expdir: the experiments directory
    '''
    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(expdir + '/database.cfg')
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(expdir + '/model/lm.cfg')
    nnet_cfg = dict(parsed_nnet_cfg.items('lm'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(expdir + '/trainer.cfg')
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    #read the decoder config file
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(expdir + '/model/decoder.cfg')
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    #create the cluster and server
    cluster, server = create_cluster.create_cluster(
        clusterfile=clusterfile,
        job_name=job_name,
        task_index=task_index,
        expdir=expdir,
        ssh_tunnel=ssh_tunnel)

    #copy the alphabet to the model
    if (job_name == 'ps' and task_index == 0) or cluster is not None:
        shutil.copyfile(os.path.join(database_cfg['train_dir'], 'alphabet'),
                        os.path.join(FLAGS.expdir, 'model', 'alphabet'))

    #the ps should just wait
    if cluster is not None  and job_name == 'ps':
        trainer.wait(server, task_index, len(cluster.as_dict()['worker']))
        return

    #create the coder
    with open(os.path.join(database_cfg['train_dir'], 'alphabet')) as fid:
        alphabet = fid.read().split(' ')
    coder = target_coder.TargetCoder(alphabet)

    #read the number of utterances
    with open(os.path.join(database_cfg['train_dir'], 'numlines')) as fid:
        num_utt = int(fid.read())

    #read the maximum length
    with open(os.path.join(database_cfg['train_dir'], 'max_num_chars')) as fid:
        max_length = int(fid.read())

    #create a batch dispenser for the training data
    dispenser = batchdispenser.LmBatchDispenser(
        target_coder=coder,
        size=int(trainer_cfg['batch_size']),
        textfile=os.path.join(database_cfg['train_dir'], 'text'),
        max_length=max_length,
        num_utt=num_utt)

    #create a reader for the validation data
    if 'dev_dir' in database_cfg:

        #read the maximum length
        with open(os.path.join(database_cfg['dev_dir'],
                               'max_num_chars')) as fid:
            max_length = int(fid.read())

        #create a batch dispenser for the training data
        val_reader = text_reader.TextReader(
            textfile=os.path.join(database_cfg['dev_dir'], 'text'),
            max_length=max_length,
            coder=coder)

        val_targets = val_reader.as_dict()

    else:
        if int(trainer_cfg['valid_utt']) > 0:
            val_dispenser = dispenser.split(int(trainer_cfg['valid_utt']))
            val_reader = val_dispenser.textreader
            val_targets = val_reader.asdict()
        else:
            val_reader = None
            val_targets = None

    #encode the validation targets
    if val_targets is not None:
        for utt in val_targets:
            val_targets[utt] = dispenser.textreader.coder.encode(
                val_targets[utt])

    #create the classifier
    classifier = lm_factory.factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels)

    #create the callable for the decoder
    decoder = partial(
        decoder_factory.factory,
        conf=decoder_cfg,
        classifier=classifier,
        input_dim=1,
        max_input_length=val_reader.max_length,
        coder=coder,
        expdir=expdir)

    #create the trainer
    tr = trainer_factory.factory(
        conf=trainer_cfg,
        decoder=decoder,
        classifier=classifier,
        input_dim=1,
        dispenser=dispenser,
        val_reader=val_reader,
        val_targets=val_targets,
        expdir=expdir,
        server=server,
        cluster=cluster,
        task_index=task_index)

    #train the classifier
    tr.train()

if __name__ == '__main__':

    #define the FLAGS
    tf.app.flags.DEFINE_string('clusterfile', None,
                               'The file containing the cluster')
    tf.app.flags.DEFINE_string('job_name', None, 'One of ps, worker')
    tf.app.flags.DEFINE_integer('task_index', None, 'The task index')
    tf.app.flags.DEFINE_string(
        'ssh_tunnel', 'False',
        'wheter or not communication should happen through an ssh tunnel')
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experimental directory')
    FLAGS = tf.app.flags.FLAGS

    train_lm(
        clusterfile=FLAGS.clusterfile,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        ssh_tunnel=FLAGS.ssh_tunnel == 'True',
        expdir=FLAGS.expdir)
