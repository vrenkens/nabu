'''@file train.py
this file will do the neural net training'''

import os
import sys
import subprocess
import atexit
import tensorflow as tf
from six.moves import configparser
import processing
import neuralnetworks

def train(clusterfile,
          job_name,
          task_index,
          expdir):

    ''' does everything for training

    Args:
        cluster: the file where all the machines in the cluster are specified
            if None, local training will be done
        job_name: one of ps or worker in the case of distributed training
        task_index: the task index in this job
        expdir: the experiments directory
    '''

    #unbuffer stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(expdir + '/database.cfg')
    database_cfg = dict(parsed_database_cfg.items('directories'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(expdir + '/features.cfg')
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the nnet config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(expdir + '/nnet.cfg')
    nnet_cfg = dict(parsed_nnet_cfg.items('nnet'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(expdir + '/trainer.cfg')
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    if clusterfile is None:
        #no distributed training
        cluster = None
        server = None
    else:
        #read the cluster file
        clusterdict = dict()
        clusterdict['worker'] = []
        clusterdict['ps'] = []
        machines = dict()
        machines['worker'] = []
        machines['ps'] = []
        with open(clusterfile) as fid:
            port = 2222
            for line in fid:
                if len(line.strip()) > 0:
                    split = line.strip().split(',')
                    clusterdict[split[0]].append('localhost:%d' % (port))
                    machines[split[0]].append(split[1])
                    port += 1
        cluster = tf.train.ClusterSpec(clusterdict)
        if job_name == "ps":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        server = tf.train.Server(cluster, job_name, task_index)

        #get the port and machin name for this task
        port = int(clusterdict[job_name][task_index].split(':')[1])
        local = machines[job_name][task_index]

        #create an ssh tunnel to all other machines
        tunneled = []
        for job in machines:
            for remote in machines[job]:
                if local != remote and remote not in tunneled:
                    p = subprocess.Popen(
                        ['ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                         'UserKnownHostsFile=/dev/null', '-R',
                         '%d:localhost:%d' % (port, port), '-N', remote])
                    tunneled.append(remote)
                    atexit.register(p.terminate)

        if job_name == 'ps':
            server.join()

    featdir = database_cfg['train_features'] + '/' +  feat_cfg['name']

    #create the coder
    normalizer = processing.target_normalizers.normalizer_factory(
        database_cfg['normalizer'])
    coder = processing.target_coder.coder_factory(
        normalizer, database_cfg['coder'])

    #create a feature reader for the training data
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())

    featreader = processing.feature_reader.FeatureReader(
        scpfile=featdir + '/feats_shuffled.scp',
        cmvnfile=featdir + '/cmvn.scp',
        utt2spkfile=featdir + '/utt2spk',
        context_width=0,
        max_input_length=max_input_length)

    #read the feature dimension
    with open(featdir + '/input_dim', 'r') as fid:
        input_dim = int(fid.read())

    #the path to the text file
    textfile = database_cfg['traintext']

    #create a batch dispenser for the training data
    dispenser = processing.batchdispenser.TextBatchDispenser(
        feature_reader=featreader,
        target_coder=coder,
        size=int(trainer_cfg['batch_size']),
        target_path=textfile)

    #create a batch dispenser for the validation data
    if 'dev_data' in database_cfg:
        featdir = database_cfg['dev_features'] + '/' +  feat_cfg['name']

        with open(featdir + '/maxlength', 'r') as fid:
            max_input_length = int(fid.read())

        val_featreader = processing.feature_reader.FeatureReader(
            scpfile=featdir + '/feats.scp',
            cmvnfile=featdir + '/cmvn.scp',
            utt2spkfile=featdir + '/utt2spk',
            context_width=0,
            max_input_length=max_input_length)

        textfile = database_cfg['devtext']

        val_dispenser = processing.batchdispenser.TextBatchDispenser(
            feature_reader=val_featreader,
            target_coder=coder,
            size=int(trainer_cfg['batch_size']),
            target_path=textfile)

    else:
        if int(trainer_cfg['valid_utt']) > 0:
            val_dispenser = dispenser.split(int(trainer_cfg['valid_utt']))
        else:
            val_dispenser = None

    #create the classifier
    classifier = neuralnetworks.classifier_factory.classifier_factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels + 1,
        classifier_type=nnet_cfg['classifier'])

    #create the trainer
    trainer = neuralnetworks.trainer.trainer_factory(
        conf=trainer_cfg,
        classifier=classifier,
        input_dim=input_dim,
        max_input_length=dispenser.max_input_length,
        max_target_length=dispenser.max_target_length,
        dispenser=dispenser,
        val_dispenser=val_dispenser,
        logdir=expdir + '/logdir',
        server=server,
        cluster=cluster,
        task_index=task_index,
        trainer_type=trainer_cfg['trainer'])

    #train the classifier
    trainer.train()

if __name__ == '__main__':

    #define the FLAGS
    tf.app.flags.DEFINE_string('clusterfile', None,
                               'The file containing the cluster')
    tf.app.flags.DEFINE_string('job_name', None, 'One of ps, worker')
    tf.app.flags.DEFINE_integer('task_index', None, 'The task index')
    tf.app.flags.DEFINE_string('expdir', '.', 'The experimental directory')
    FLAGS = tf.app.flags.FLAGS

    train(
        clusterfile=FLAGS.clusterfile,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        expdir=FLAGS.expdir)
