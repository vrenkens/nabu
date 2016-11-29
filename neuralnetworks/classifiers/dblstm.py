'''@file dblstm.py
contains de DBLSTM class'''

import tensorflow as tf
from classifier import Classifier
from layer import FFLayer, BLSTMLayer
from activation import TfActivation
import seq_convertors


class DBLSTM(Classifier):
    '''A deep bidirectional LSTM classifier'''

    def __init__(self, output_dim, num_layers, num_units, activation):
        '''
        DBLSTM constructor

        Args:
            output_dim: the output dimension
            num_layers: number of BLSTM layers
            num_units: number of units in the LSTM cells
            activation: the activation function that will be used between
                the layers
        '''

        super(DBLSTM, self).__init__(output_dim)
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation

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
                - a dictionary of control operations (may be empty)
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #the blstm layer
            blstm = BLSTMLayer(self.num_units)

            #the linear output layer
            outlayer = FFLayer(self.output_dim,
                               TfActivation(None, lambda(x): x), 0)

            #do the forward computation

            #add gaussian noise to the inputs
            if is_training:
                logits = inputs + tf.random_normal(inputs.get_shape(), stddev=0.6)
            else:
                logits = inputs

            for layer in range(self.num_layers):
                logits = blstm(logits, input_seq_length,
                               is_training, reuse, 'layer' + str(layer))

                logits = self.activation(logits, is_training, reuse)

            logits = seq_convertors.seq2nonseq(logits, input_seq_length)

            logits = outlayer(logits, is_training, reuse, 'outlayer')

            logits = seq_convertors.nonseq2seq(logits, input_seq_length,
                                               int(inputs.get_shape()[1]))

            #create a saver
            saver = tf.train.Saver()

        return logits, input_seq_length, saver, None
