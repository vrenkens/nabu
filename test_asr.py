'''@file test_asr.py
this file will test the asr on its own'''

import os
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.classifiers.asr import asr_factory
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.processing import feature_reader, target_coder


tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing'''

    decoder_cfg_file = 'config/decoder/attention_visualizer.cfg'

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(os.path.join(FLAGS.expdir, 'database.cfg'))
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(os.path.join(FLAGS.expdir, 'model', 'features.cfg'))
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(os.path.join(FLAGS.expdir, 'model', 'asr.cfg'))
    nnet_cfg = dict(parsed_nnet_cfg.items('asr'))

    #read the decoder config file
    if decoder_cfg_file is None:
        decoder_cfg_file = os.path.join(FLAGS.expdir, 'model', 'decoder.cfg')
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
            max_input_length=reader.max_length,
            coder=coder,
            expdir=FLAGS.expdir)

        saver = tf.train.Saver(tf.trainable_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        #load the model
        saver.restore(sess, os.path.join(FLAGS.expdir, 'model', 'network.ckpt'))

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
        references[splitline[0]] = coder.encode(' '.join(splitline[1:]))

    #compute the character error rate
    score = decoder.score(decoded, references)

    print 'score: %f' % score

    #write the resulting beams to disk
    decodedir = os.path.join(FLAGS.expdir, 'decoded')
    if not os.path.isdir(decodedir):
        os.makedirs(decodedir)
    for utt in decoded:
        with open(os.path.join(decodedir, utt), 'w') as fid:
            for hypothesis in decoded[utt]:
                fid.write('%f\t%s\n' % (hypothesis[0], hypothesis[1]))

if __name__ == '__main__':
    tf.app.run()
