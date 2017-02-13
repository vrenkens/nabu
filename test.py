'''@file test.py
this file will do the neural net testing'''

import os
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.classifiers.asr import asr_factory
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.processing.target_coders import coder_factory
from nabu.processing.target_normalizers import normalizer_factory
from nabu.processing import feature_reader


tf.app.flags.DEFINE_string('expdir', '.', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing'''

    decoder_cfg_file = None

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(FLAGS.expdir + '/database.cfg')
    database_cfg = dict(parsed_database_cfg.items('directories'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(FLAGS.expdir + '/features.cfg')
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(FLAGS.expdir + '/asr.cfg')
    nnet_cfg = dict(parsed_nnet_cfg.items('asr'))

    #read the decoder config file
    if decoder_cfg_file is None:
        decoder_cfg_file = FLAGS.expdir + '/decoder.cfg'
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(decoder_cfg_file)
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    decodedir = FLAGS.expdir + '/decoded'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    #create a feature reader
    featdir = database_cfg['test_features'] + '/' +  feat_cfg['name']

    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())

    reader = feature_reader.FeatureReader(
        scpfile=featdir + '/feats.scp',
        cmvnfile=featdir + '/cmvn.scp',
        utt2spkfile=featdir + '/utt2spk',
        context_width=0,
        max_input_length=max_input_length)

    #read the feature dimension
    with open(
        database_cfg['train_features'] + '/' +  feat_cfg['name'] + '/input_dim',
        'r') as fid:

        input_dim = int(fid.read())

    #create the coder
    normalizer = normalizer_factory.factory(
        database_cfg['normalizer'])
    coder = coder_factory.factory(
        normalizer, database_cfg['coder'])


    #create the classifier
    classifier = asr_factory.factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels)

    #create a decoder
    graph = tf.Graph()
    with graph.as_default():
        decoder = decoder_factory.factory(
            conf=decoder_cfg,
            classifier=classifier,
            input_dim=input_dim,
            max_input_length=reader.max_input_length,
            coder=coder,
            expdir=FLAGS.expdir,
            decoder_type=decoder_cfg['decoder'])

        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Classifier'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        #load the model
        saver.restore(sess, FLAGS.expdir + '/model/network.ckpt')

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
        references[splitline[0]] = coder.normalize(' '.join(splitline[1:]))

    #compute the character error rate
    score = decoder.score(decoded, references)

    print 'score: %f' % score

if __name__ == '__main__':
    tf.app.run()
