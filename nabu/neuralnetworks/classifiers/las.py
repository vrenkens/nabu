'''@file dblstm.py
contains de LAS class'''

import tensorflow as tf
from classifier import Classifier
import encoders
import decoders

class LAS(Classifier):
    '''a listen attend and spell classifier'''
    def __init__(self, conf, output_dim):
        '''LAS constructor

        Args:
            conf: the classifier config
            output_dim: the classifier output dimension'''

        #create the listener
        self.encoder = encoders.listener.Listener(
            numlayers=int(conf['listener_layers']),
            numunits=int(conf['listener_units']),
            dropout=float(conf['listener_dropout']))

        #create the speller
        self.decoder = decoders.speller.Speller(
            numlayers=int(conf['speller_layers']),
            numunits=int(conf['speller_units']),
            dropout=float(conf['speller_dropout']),
            sample_prob=float(conf['sample_prob']))

        super(LAS, self).__init__(conf, output_dim)

    def __call__(self, inputs, input_seq_length, targets=None,
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
            numlabels=self.output_dim,
            initial_state=None,
            is_training=is_training)

        return logits, target_seq_length + 1
