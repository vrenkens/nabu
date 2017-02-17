'''@file test_asr.py
this file will test the language model on its own'''

import os
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.classifiers.lm import lm_factory
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.processing import target_coder, text_reader


tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing'''

    decoder_cfg_file = None

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(os.path.join(FLAGS.expdir, 'database.cfg'))
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(os.path.join(FLAGS.expdir, 'model', 'lm.cfg'))
    nnet_cfg = dict(parsed_nnet_cfg.items('lm'))

    #read the decoder config file
    if decoder_cfg_file is None:
        decoder_cfg_file = os.path.join(FLAGS.expdir, 'model', 'decoder.cfg')
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(decoder_cfg_file)
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    #create the coder
    with open(os.path.join(FLAGS.expdir, 'model', 'alphabet')) as fid:
        alphabet = fid.read().split(' ')
    coder = target_coder.TargetCoder(alphabet)

    #read the maximum length
    with open(os.path.join(database_cfg['test_dir'],
                           'max_num_chars')) as fid:
        max_length = int(fid.read())

    #create a text reader
    textreader = text_reader.TextReader(
        textfile=os.path.join(database_cfg['test_dir'], 'text'),
        max_length=max_length,
        coder=coder)

    #create the classifier
    classifier = lm_factory.factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels)

    #create a decoder
    graph = tf.Graph()
    with graph.as_default():
        decoder = decoder_factory.factory(
            conf=decoder_cfg,
            classifier=classifier,
            input_dim=1,
            max_input_length=max_length,
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
        decoded = decoder.decode(textreader, sess)

    #compute the character error rate
    score = decoder.score(decoded, None)

    print 'perplexity: %f' % score

if __name__ == '__main__':
    tf.app.run()
