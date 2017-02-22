'''@file test.py
this file will test the asr combined with an lm'''

import os
from six.moves import configparser
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from nabu.neuralnetworks.classifiers import asr_lm_classifier
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.processing import feature_reader, target_coder


tf.app.flags.DEFINE_string('asr_expdir', 'expdir',
                           'The asr experiments directory')
tf.app.flags.DEFINE_string('lm_expdir', 'expdir',
                           'The lm experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing'''

    decoder_cfg_file = None

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(os.path.join(FLAGS.asr_expdir, 'database.cfg'))
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(
        os.path.join(FLAGS.asr_expdir, 'model', 'features.cfg'))
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the asr config file
    parsed_asr_cfg = configparser.ConfigParser()
    parsed_asr_cfg.read(os.path.join(FLAGS.asr_expdir, 'model', 'asr.cfg'))
    asr_cfg = dict(parsed_asr_cfg.items('asr'))

    #read the lm config file
    parsed_lm_cfg = configparser.ConfigParser()
    parsed_lm_cfg.read(os.path.join(FLAGS.lm_expdir, 'model', 'lm.cfg'))
    lm_cfg = dict(parsed_lm_cfg.items('lm'))

    #read the asr-lm config file
    parsed_asr_lm_cfg = configparser.ConfigParser()
    parsed_asr_lm_cfg.read('config/asr_lm.cfg')
    asr_lm_cfg = dict(parsed_asr_lm_cfg.items('asr-lm'))

    #read the decoder config file
    if decoder_cfg_file is None:
        decoder_cfg_file = os.path.join(FLAGS.asr_expdir, 'model',
                                        'decoder.cfg')
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(decoder_cfg_file)
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    #create a feature reader
    featdir = os.path.join(database_cfg['test_dir'], feat_cfg['name'])

    with open(os.path.join(featdir, 'maxlength'), 'r') as fid:
        max_length = int(fid.read())

    reader = feature_reader.FeatureReader(
        scpfile=os.path.join(featdir, 'feats.scp'),
        cmvnfile=os.path.join(featdir, 'cmvn.scp'),
        utt2spkfile=os.path.join(featdir, 'utt2spk'),
        max_length=max_length)

    #read the feature dimension
    with open(
        os.path.join(database_cfg['train_dir'], feat_cfg['name'],
                     'dim'),
        'r') as fid:

        input_dim = int(fid.read())

    #create the coder
    with open(os.path.join(database_cfg['train_dir'], 'alphabet')) as fid:
        alphabet = fid.read().split(' ')
    coder = target_coder.TargetCoder(alphabet)


    #create the classifier
    classifier = asr_lm_classifier.AsrLmClassifier(
        conf=asr_lm_cfg,
        asr_conf=asr_cfg,
        lm_conf=lm_cfg,
        output_dim=coder.num_labels)

    #create a decoder
    graph = tf.Graph()
    with graph.as_default():
        decoder = decoder_factory.factory(
            conf=decoder_cfg,
            classifier=classifier,
            input_dim=input_dim,
            max_input_length=reader.max_length,
            coder=coder,
            expdir=FLAGS.asr_expdir)


        #create the lm saver
        varnames = zip(*checkpoint_utils.list_variables(os.path.join(
            FLAGS.lm_expdir, 'model', 'network.ckpt')))[0]
        variables = [v for v in tf.all_variables()
                     if v.name.split(':')[0] in varnames]
        lm_saver = tf.train.Saver(variables)

        #create the asr saver
        varnames = zip(*checkpoint_utils.list_variables(os.path.join(
            FLAGS.asr_expdir, 'model', 'network.ckpt')))[0]
        variables = [v for v in tf.all_variables()
                     if v.name.split(':')[0] in varnames]
        asr_saver = tf.train.Saver(variables)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        #load the lm model
        lm_saver.restore(
            sess, os.path.join(FLAGS.lm_expdir, 'model', 'network.ckpt'))

        #load the asr model
        asr_saver.restore(
            sess, os.path.join(FLAGS.asr_expdir, 'model', 'network.ckpt'))

        #decode with te neural net
        decoded = decoder.decode(reader, sess)

    #the path to the text file
    textfile = database_cfg['testtext']

    #read all the reference transcriptions
    with open(textfile) as fid:
        lines = fid.readlines()

    references = dict()
    for line in lines:
        splitline = line.strip().split(' ')
        references[splitline[0]] = ' '.join(splitline[1:])

    #compute the character error rate
    score = decoder.score(decoded, references)

    print 'score: %f' % score

if __name__ == '__main__':
    tf.app.run()
