'''@file dblstm.py
contains de LAS class'''

import tensorflow as tf
from classifier import Classifier
import las_elements

class LAS(Classifier):
    '''a listen attend and spell classifier'''
    def __init__(self, conf, output_dim):
        '''LAS constructor

        Args:
            conf: the classifier config
            output_dim: the classifier output dimension'''

        #create the listener
        self.encoder = las_elements.listener.Listener(
            numlayers=int(conf['listener_layers']),
            numunits=int(conf['listener_units']),
            dropout=float(conf['listener_dropout']))

        #create the speller
        self.decoder = las_elements.speller.Speller(
            numlayers=int(conf['speller_layers']),
            numunits=int(conf['speller_units']),
            dropout=float(conf['speller_dropout']))

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
            #compute the high level features
            hlfeat = self.encoder(inputs, input_seq_length, is_training)

            #compute the output logits
            logits, _ = self.decoder(hlfeat, targets, self.output_dim, None,
                                     is_training)

            return logits, target_seq_length
