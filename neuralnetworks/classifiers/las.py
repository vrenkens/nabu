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
                 target_seq_length=None, is_training=False, scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] dimansional vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length x 1] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] dimansional vector
            is_training: whether or not the network is in training mode
            scope: the name scope

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        with tf.variable_scope(scope or type(self).__name__):

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

            #shift the targets to encoder inputs by prepending a start of
            #sequence label and taking of the end of sequence label
            batch_size = int(targets.get_shape()[0])
            sos_labels = tf.ones([batch_size, 1, 1], dtype=tf.int32)
            encoder_inputs = tf.concat(1, [sos_labels, targets])
            encoder_inputs = encoder_inputs[:, :-1, :]

            #compute the output logits
            logits, _ = self.decoder(
                hlfeat=hlfeat,
                encoder_inputs=encoder_inputs,
                numlabels=self.output_dim,
                initial_state=None,
                is_training=is_training)

            return logits, target_seq_length
