'''@file dblstm.py
contains de DBLSTM class'''

import tensorflow as tf
from classifier import Classifier
from layer import FFLayer, BLSTMLayer
from activation import TfActivation


class DBLSTM(Classifier):
    '''A deep bidirectional LSTM classifier'''

    def __init__(self, output_dim, num_layers, num_units):
        '''
        DBLSTM constructor

        Args:
            output_dim: the output dimension
            num_layers: number of BLSTM layers
            num_units: number of units in the LSTM cells
        '''

        super(DBLSTM, self).__init__(output_dim)
        self.num_layers = num_layers
        self.num_units = num_units

    def __call__(self, inputs, seq_length, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a list containing
                a [batch_size, input_dim] tensor for each time step
            seq_length: The sequence lengths of the input utterances
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A triple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - a dictionary of control operations (empty)
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #the blstm layer
            blstm = BLSTMLayer(self.num_units)

            #the linear output layer
            outlayer = FFLayer(self.output_dim,
                               TfActivation(None, lambda(x): x), 0)

            #do the forward computation
            logits = inputs

            for layer in self.num_layers:
                logits = blstm(logits, is_training, reuse, 'layer' + str(layer))

            logits = outlayer(logits, is_training, reuse, 'outlayer')

            #create a saver
            saver = tf.train.Saver()

        return logits, seq_length, saver, None
