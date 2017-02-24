'''@file train_asr.py
this file will do the asr training'''

import os
from functools import partial
import tensorflow as tf
from six.moves import configparser
from nabu.distributed import create_cluster
from nabu.processing import batchdispenser, feature_reader, target_coder
from nabu.neuralnetworks.classifiers.asr import asr_factory
from nabu.neuralnetworks.trainers import trainer_factory
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.neuralnetworks.trainers import trainer

def train_asr(clusterfile,
              job_name,
              task_index,
              ssh_tunnel,
              expdir):

    ''' does everything for asr training

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
    parsed_database_cfg.read(os.path.join(expdir, 'database.cfg'))
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(os.path.join(expdir, 'model', 'features.cfg'))
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(os.path.join(expdir, 'model', 'asr.cfg'))
    nnet_cfg = dict(parsed_nnet_cfg.items('asr'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    #read the decoder config file
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(os.path.join(expdir, 'model', 'decoder.cfg'))
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    #create the cluster and server
    cluster, server = create_cluster.create_cluster(
        clusterfile=clusterfile,
        job_name=job_name,
        task_index=task_index,
        expdir=expdir,
        ssh_tunnel=ssh_tunnel)

    #the ps should just wait
    if cluster is not None  and job_name == 'ps':
        trainer.wait(server, task_index, len(cluster.as_dict()['worker']))
        return

    featdir = os.path.join(database_cfg['train_dir'], feat_cfg['name'])

    #create the coder
    with open(os.path.join(database_cfg['train_dir'], 'alphabet')) as fid:
        alphabet = fid.read().split(' ')
    coder = target_coder.TargetCoder(alphabet)

    #create a feature reader for the training data
    with open(featdir + '/maxlength', 'r') as fid:
        max_length = int(fid.read())

    featreader = feature_reader.FeatureReader(
        scpfile=featdir + '/feats_shuffled.scp',
        cmvnfile=featdir + '/cmvn.scp',
        utt2spkfile=featdir + '/utt2spk',
        max_length=max_length)

    #read the feature dimension
    with open(featdir + '/dim', 'r') as fid:
        input_dim = int(fid.read())

    #the path to the text file
    textfile = os.path.join(database_cfg['train_dir'], 'targets')

    #create a batch dispenser for the training data
    dispenser = batchdispenser.AsrBatchDispenser(
        feature_reader=featreader,
        target_coder=coder,
        size=int(trainer_cfg['batch_size']),
        target_path=textfile)

    #create a reader for the validation data
    if 'dev_data' in database_cfg:
        featdir = database_cfg['dev_dir'] + '/' +  feat_cfg['name']

        with open(featdir + '/maxlength', 'r') as fid:
            max_length = int(fid.read())

        val_reader = feature_reader.FeatureReader(
            scpfile=featdir + '/feats.scp',
            cmvnfile=featdir + '/cmvn.scp',
            utt2spkfile=featdir + '/utt2spk',
            max_length=max_length)

        textfile = os.path.join(database_cfg['dev_dir'], 'targets')

        #read the validation targets
        with open(textfile) as fid:
            lines = fid.readlines()

        val_targets = dict()
        for line in lines:
            splitline = line.strip().split(' ')
            val_targets[splitline[0]] = ' '.join(splitline[1:])

    else:
        if int(trainer_cfg['valid_utt']) > 0:
            val_dispenser = dispenser.split(int(trainer_cfg['valid_utt']))
            val_reader = val_dispenser.feature_reader
            val_targets = val_dispenser.target_dict
        else:
            val_reader = None
            val_targets = None

    #encode the validation targets
    if val_targets is not None:
        for utt in val_targets:
            val_targets[utt] = dispenser.target_coder.encode(
                val_targets[utt])

    #create the classifier
    classifier = asr_factory.factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels)

    #create the callable for the decoder
    decoder = partial(
        decoder_factory.factory,
        conf=decoder_cfg,
        classifier=classifier,
        input_dim=input_dim,
        max_input_length=val_reader.max_length,
        coder=coder,
        expdir=expdir)

    #create the trainer
    tr = trainer_factory.factory(
        conf=trainer_cfg,
        decoder=decoder,
        classifier=classifier,
        input_dim=input_dim,
        dispenser=dispenser,
        val_reader=val_reader,
        val_targets=val_targets,
        expdir=expdir,
        server=server,
        cluster=cluster,
        task_index=task_index)

    print 'starting training'

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

    train_asr(
        clusterfile=FLAGS.clusterfile,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        ssh_tunnel=FLAGS.ssh_tunnel == 'True',
        expdir=FLAGS.expdir)
