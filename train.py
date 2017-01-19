'''@file train.py
this file will do the neural net training'''

import os
import subprocess
import atexit
import tensorflow as tf
from six.moves import configparser
import distributed
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

    #read the decoder config file
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(expdir + '/decoder.cfg')
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    if clusterfile is None:
        #no distributed training
        cluster = None
        server = None
    else:
        #read the cluster file
        machines = dict()
        machines['worker'] = []
        machines['ps'] = []
        with open(clusterfile) as fid:
            for line in fid:
                if len(line.strip()) > 0:
                    split = line.strip().split(',')
                    machines[split[0]].append((split[1], int(split[2])))

        #build the cluster and create ssh tunnels to machines in the cluster
        port = 1024
        clusterdict = dict()
        clusterdict['worker'] = []
        clusterdict['ps'] = []
        localmachine = machines[job_name][task_index][0]

        #get a list of ports used on this machine
        localports = []
        for job in machines:
            for remote in machines[job]:
                if localmachine == remote[0]:
                    localports.append(remote[1])

        for job in machines:
            for remote in machines[job]:

                #create an ssh tunnel if the local machine is not the same as
                #the remote machine
                if localmachine != remote[0]:

                    #look for an available port
                    while (port in localports
                           or not distributed.cluster.port_available(port)):

                        port += 1

                    #create the ssh tunnel
                    p = subprocess.Popen(
                        ['ssh', '-o', 'StrictHostKeyChecking=no', '-o',
                         'UserKnownHostsFile=/dev/null', '-L',
                         '%d:localhost:%d' % (port, remote[1]), '-N',
                         remote[0]])

                    #make sure the ssh tunnel is closed at exit
                    atexit.register(p.terminate)

                    #add the machine to the cluster
                    clusterdict[job].append('localhost:%d' % port)

                    port += 1

                else:
                    clusterdict[job].append('localhost:%d' % remote[1])

        #create the cluster
        cluster = tf.train.ClusterSpec(clusterdict)

        #make sure the ps does not use a GPU
        if job_name == "ps":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        #create the server for this task
        server = tf.train.Server(cluster, job_name, task_index)

        #the ps should just wait
        if job_name == 'ps':
            neuralnetworks.trainers.trainer.wait(server, task_index,
                                                 len(machines['worker']))
            return

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

        val_reader = processing.feature_reader.FeatureReader(
            scpfile=featdir + '/feats.scp',
            cmvnfile=featdir + '/cmvn.scp',
            utt2spkfile=featdir + '/utt2spk',
            context_width=0,
            max_input_length=max_input_length)

        textfile = database_cfg['devtext']

        #read the validation targets
        with open(textfile) as fid:
            lines = fid.readlines()

        val_targets = dict()
        for line in lines:
            splitline = line.strip().split(' ')
            val_targets[splitline[0]] = coder.normalize(' '.join(splitline[1:]))

        # TEMPORARY LINE FOR TESTING! REMOVE
        print 'SPLITTING VALDATION DATA'
        val_reader = val_reader.split(32)
        utt_ids = val_reader.reader.utt_ids
        val_targets = {utt_id:val_targets[utt_id] for utt_id in utt_ids}

    else:
        if int(trainer_cfg['valid_utt']) > 0:
            val_dispenser = dispenser.split(int(trainer_cfg['valid_utt']))
            val_reader = val_dispenser.feature_reader
            val_targets = val_dispenser.target_dict
        else:
            val_dispenser = None

    #create the classifier
    classifier = neuralnetworks.classifiers.classifier_factory.factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels,
        classifier_type=nnet_cfg['classifier'])

    #create the trainer
    trainer = neuralnetworks.trainers.trainer_factory.factory(
        conf=trainer_cfg,
        decoder_conf=decoder_cfg,
        classifier=classifier,
        input_dim=input_dim,
        max_input_length=dispenser.max_input_length,
        max_target_length=dispenser.max_target_length,
        dispenser=dispenser,
        val_reader=val_reader,
        val_targets=val_targets,
        expdir=expdir,
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
