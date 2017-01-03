'''@file train.py
this file will do the neural net training'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import subprocess
import shutil
import tensorflow as tf
from six.moves import configparser
from processing import feature_reader, batchdispenser, target_coder, \
target_normalizers, prepare_data
from neuralnetworks.classifiers import *
import neuralnetworks.trainer

#parse the input
tf.app.flags.DEFINE_string('cluster', 'None',
                           'a file containing the cluster information')
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
FLAGS = tf.app.flags.FLAGS

if FLAGS.cluster == 'None':
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
    with open(FLAGS.cluster) as fid:
        port = 2222
        for line in fid:
            split = line.strip().split(',')
            clusterdict[split[0]].append('localhost:%d' % (port))
            machines[split[0]].append(split[1])
            port += 1
    cluster = tf.train.ClusterSpec(clusterdict)
    if FLAGS.job_name == "ps":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    server = tf.train.Server(cluster, FLAGS.job_name, FLAGS.task_index)

    #get the port and machin name for this task
    port = int(clusterdict[FLAGS.job_name][FLAGS.task_index].split(':')[1])
    local = machines[FLAGS.job_name][FLAGS.task_index]

    #create an ssh tunnel to all other machines
    ssh_tunnels = []
    tunneled = []
    for job in machines:
        for remote in machines[job]:
            if local != remote and remote not in tunneled:
                ssh_tunnels.append(subprocess.Popen(
                    ['ssh', '-R',  '%d:localhost:%d' % (port, port), '-N',
                     remote]))
                tunneled.append(remote)

if FLAGS.job_name == "ps" and cluster is not None:
    server.join()

else:

    #pointers to the config files
    database_cfg_file = 'config/databases/TIMIT.cfg'
    feat_cfg_file = 'config/features/fbank.cfg'
    nnet_cfg_file = 'config/nnet/DBLSTM.cfg'
    trainer_cfg_file = 'config/trainer/CTCtrainer.cfg'

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(database_cfg_file)
    database_cfg = dict(parsed_database_cfg.items('directories'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(feat_cfg_file)
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the nnet config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(nnet_cfg_file)
    nnet_cfg = dict(parsed_nnet_cfg.items('nnet'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(trainer_cfg_file)
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    featdir = database_cfg['train_features'] + '/' +  feat_cfg['name']

    #create the coder
    coder = eval('target_coder.%s(target_normalizers.%s)' % (
        database_cfg['coder'],
        database_cfg['normalizer']))

    #only shuffle if we start with initialisation
    if trainer_cfg['resume_training'] != 'True':
        #shuffle the examples on disk
        print '------- shuffling examples ----------'
        prepare_data.shuffle_examples(featdir)


    #create a feature reader for the training data
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())

    featreader = feature_reader.FeatureReader(
        scpfile=featdir + '/feats_shuffled.scp',
        cmvnfile=featdir + '/cmvn.scp',
        utt2spkfile=featdir + '/utt2spk',
        context_width=0,
        max_input_length=max_input_length)

    #read the feature dimension
    with open(featdir + '/input_dim', 'r') as fid:
        input_dim = int(fid.read())

    expdir = nnet_cfg['expdir']

    #the path to the text file
    textfile = database_cfg['traintext']

    #create a batch dispenser for the training data
    dispenser = batchdispenser.TextBatchDispenser(
        feature_reader=featreader,
        target_coder=coder,
        size=int(trainer_cfg['batch_size']),
        target_path=textfile)

    #create a batch dispenser for the validation data
    if 'dev_data' in database_cfg:
        featdir = database_cfg['dev_features'] + '/' +  feat_cfg['name']

        with open(featdir + '/maxlength', 'r') as fid:
            max_input_length = int(fid.read())

        val_featreader = feature_reader.FeatureReader(
            scpfile=featdir + '/feats.scp',
            cmvnfile=featdir + '/cmvn.scp',
            utt2spkfile=featdir + '/utt2spk',
            context_width=0,
            max_input_length=max_input_length)

        textfile = database_cfg['devtext']

        val_dispenser = batchdispenser.TextBatchDispenser(
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
    class_name = '%s.%s' % (nnet_cfg['module'],
                                           nnet_cfg['class'])
    classifier = eval(class_name)(nnet_cfg, coder.num_labels + 1)

    if trainer_cfg['resume_training'] != True:
        if os.path.isdir(expdir + '/logdir'):
            shutil.rmtree(expdir + '/logdir')

    #create the trainer
    class_name = 'neuralnetworks.trainer.%s' % (trainer_cfg['trainer'])
    trainer = eval(class_name)(
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
        task_index=FLAGS.task_index)

    #train the classifier
    trainer.train()

    #close the ssh tunnels
    for ssh_tunnel in ssh_tunnels:
        ssh_tunnel.kill()
