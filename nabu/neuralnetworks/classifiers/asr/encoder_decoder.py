'''@file dblstm.py
contains de LAS class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier
from encoders import encoder_factory
from asr_decoders import asr_decoder_factory

class EncoderDecoder(classifier.Classifier):
    '''a general class for an encoder decoder system'''
    def __init__(self, conf, output_dim, name=None):
        '''LAS constructor

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        super(EncoderDecoder, self).__init__(conf, output_dim, name)

        #create the listener
        self.encoder = encoder_factory.factory(conf)

        #create the speller
        self.decoder = asr_decoder_factory.factory(conf, self.output_dim)

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        #add input noise
        std_input_noise = float(self.conf['std_input_noise'])
        if is_training and std_input_noise > 0:
            noisy_inputs = inputs + tf.random_normal(
                inputs.get_shape(), stddev=std_input_noise)
        else:
            noisy_inputs = inputs

        #compute the high level features
        hlfeat = self.encoder(
            inputs=noisy_inputs,
            sequence_lengths=input_seq_length,
            is_training=is_training)

        #prepend a sequence border label to the targets to get the encoder
        #inputs, the label is the last label
        batch_size = int(targets.get_shape()[0])
        s_labels = tf.constant(self.output_dim-1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])
        encoder_inputs = tf.concat(1, [s_labels, targets])

        #compute the output logits
        logits, _ = self.decoder(
            hlfeat=hlfeat,
            encoder_inputs=encoder_inputs,
            initial_state=self.decoder.zero_state(batch_size),
            first_step=True,
            is_training=is_training)

        return logits, target_seq_length + 1
