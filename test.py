'''@file test.py
this file will do the neural net testing'''

import os
from six.moves import configparser
import tensorflow as tf
import neuralnetworks
import processing

tf.app.flags.DEFINE_string('expdir', '.', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing'''

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(FLAGS.expdir + '/database.cfg')
    database_cfg = dict(parsed_database_cfg.items('directories'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(FLAGS.expdir + '/features.cfg')
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the nnet config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(FLAGS.expdir + '/nnet.cfg')
    nnet_cfg = dict(parsed_nnet_cfg.items('nnet'))

    #read the decoder config file
    parsed_decoder_cfg = configparser.ConfigParser()
    parsed_decoder_cfg.read(FLAGS.expdir + '/decoder.cfg')
    decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

    decodedir = FLAGS.expdir + '/decoded'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    #create a feature reader
    featdir = database_cfg['test_features'] + '/' +  feat_cfg['name']

    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())

    reader = processing.feature_reader.FeatureReader(
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
    normalizer = processing.target_normalizers.normalizer_factory(
        database_cfg['normalizer'])
    coder = processing.target_coder.coder_factory(
        normalizer, database_cfg['coder'])


    #create the classifier
    classifier = neuralnetworks.classifier_factory.classifier_factory(
        conf=nnet_cfg,
        output_dim=coder.num_labels + 1,
        classifier_type=nnet_cfg['classifier'])

    #create a decoder
    decoder = neuralnetworks.decoder.decoder_factory(
        conf=decoder_cfg,
        classifier=classifier,
        input_dim=input_dim,
        max_input_length=reader.max_input_length,
        expdir=FLAGS.expdir,
        decoder_type=decoder_cfg['decoder'])

    #decode with te neural net
    decoder.decode(reader, coder)

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
    CER = processing.score.cer(decodedir, references)

    print 'character error rate: %f' % CER

if __name__ == '__main__':
    tf.app.run()
