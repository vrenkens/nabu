'''@file model.py
contains the Model class that is used to perform recognition with pre-trained
models'''

from collections import OrderedDict
from six.moves import configparser
import tensorflow as tf
from nabu.processing.feature_computers import feature_computer_factory
from nabu.neuralnetworks.classifiers.asr import asr_factory
from nabu.neuralnetworks.decoders import decoder_factory

class Model(object):
    '''the Model class can be used to decode with a pre-trained model'''
    def __init__(self,
                 modeldir,
                 decoder_cfg_file=None,
                 max_input_length=None):
        '''Model constructor

        Args:
            modeldir: The directory containing the necesary model files
            decoder_cfg_file: [optional] The decoder config file, if not
                specified the model's default decoder will be used
            max_input_length: [optional] The maximal length of an input sequence
                in the feature space. If not set the model's default will be
                used

        '''

        #load the config files

        #read the features config file
        parsed_feat_cfg = configparser.ConfigParser()
        parsed_feat_cfg.read(modeldir + '/features.cfg')
        feat_cfg = dict(parsed_feat_cfg.items('features'))

        #read the nnet config file
        parsed_nnet_cfg = configparser.ConfigParser()
        parsed_nnet_cfg.read(modeldir + '/asr.cfg')
        nnet_cfg = dict(parsed_nnet_cfg.items('asr'))

        #read the decoder config file
        if decoder_cfg_file is None:
            decoder_cfg_file = modeldir + '/decoder.cfg'
        parsed_decoder_cfg = configparser.ConfigParser()
        parsed_decoder_cfg.read(decoder_cfg_file)
        decoder_cfg = dict(parsed_decoder_cfg.items('decoder'))

        #set the batch size of the decoder to 1
        decoder_cfg['batch_size'] = '1'

        #create the feature computer
        self.feature_computer = feature_computer_factory.factory(feat_cfg)

        #read the alphabet
        with open(modeldir + '/alphabet') as fid:
            alphabet = [line.strip() for line in fid.readlines()]

        #create a target decoder
        coder = TargetDecoder(alphabet)

        #read the maximum input length
        if max_input_length is None:
            with open(modeldir + '/max_input_length') as fid:
                max_input_length = int(fid.read())

        #create the classifier
        classifier = asr_factory.factory(
            conf=nnet_cfg,
            output_dim=len(alphabet))

        #create a decoder
        graph = tf.Graph()
        with graph.as_default():
            self.decoder = decoder_factory.factory(
                conf=decoder_cfg,
                classifier=classifier,
                input_dim=self.feature_computer.get_dim(),
                max_input_length=max_input_length,
                coder=coder,
                expdir=modeldir,
                decoder_type=decoder_cfg['decoder'])

            saver = tf.train.Saver(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='Classifier'))

        #start the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        config.allow_soft_placement = True
        self.sess = tf.Session(graph=graph, config=config)

        #load the model
        saver.restore(self.sess, modeldir + '/network.ckpt')

    def __call__(self, sig, rate):
        '''perform recognition on an utterance

        Args:
            sig: a numpy array containing the input signal
            rate: the sample rate of the input signal
        '''

        #compute the features for the input signal
        features = self.feature_computer(sig, rate)

        #perform recognition
        transcription = self.decoder.decode_utt(features, self.sess)

        return transcription

    def close(self):
        '''close the model'''

        self.sess.close()


class TargetDecoder(object):
    '''Converts a sequence of labels to a sequence of targets

    Mimics the interface of the TargetCoder class'''

    def __init__(self, alphabet):
        '''TargetDecoder constrauctor

        Args:
            alphabet: the target alphabet as a lst of strings
        '''

        #create a lookup dictionary
        self.lookup = OrderedDict([(character, index) for index, character
                                   in enumerate(alphabet)])

    def decode(self, encoded_targets):
        '''
        decode an encoded target sequence

        Args:
            encoded_targets: A numpy array containing the encoded targets

        Returns:
            A string containing the decoded target sequence
        '''

        targets = [self.lookup.keys()[encoded_target]
                   for encoded_target in encoded_targets]

        return ' '.join(targets)
