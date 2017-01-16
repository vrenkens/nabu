'''@file dblstm.py
contains de LAS class'''

import tensorflow as tf
from classifier import Classifier
import las_elements

class LAS(Classifier):
    '''a listen attend and spell classifier'''

    def __call__(self, inputs, input_seq_length, targets=None,
                 target_seq_length=None, is_training=False, reuse=False,
                 scope=None):
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
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A quadruple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - an empty dictionary of control operations
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #create the listener
            listener = las_elements.listener.Listener(
                numlayers=int(self.conf['listener_layers']),
                numunits=int(self.conf['listener_units']),
                dropout=float(self.conf['listener_dropout']))

            #create the speller
            speller = las_elements.speller.Speller(
                numlayers=int(self.conf['speller_layers']),
                numunits=int(self.conf['speller_units']),
                dropout=float(self.conf['speller_dropout']))

            #compute the high level features
            hlfeat = listener(inputs, input_seq_length, is_training, reuse)

            #compute the output logits
            logits = speller(hlfeat, targets, self.output_dim, is_training,
                             reuse)

            #create a saver object
            saver = tf.train.Saver()

            return logits, target_seq_length, saver, None
