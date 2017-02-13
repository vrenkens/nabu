'''@file listener.py
contains the listener code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer

class Listener(object):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, numlayers, numunits, dropout=1, name=None):
        '''Listener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate
            name: the name of the Listener'''



        #save the parameters
        self.numlayers = numlayers
        self.dropout = dropout

        #create the pblstm layer
        self.pblstm = layer.PBLSTMLayer(numunits)

        #create the blstm layer
        self.blstm = layer.BLSTMLayer(numunits)

        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, sequence_lengths, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''

        with tf.variable_scope(self.scope):

            outputs = inputs
            output_seq_lengths = sequence_lengths
            for l in range(self.numlayers):
                outputs, output_seq_lengths = self.pblstm(
                    outputs, output_seq_lengths, 'layer%d' % l)

                if self.dropout < 1 and is_training:
                    outputs = tf.nn.dropout(outputs, self.dropout)

            outputs = self.blstm(
                outputs, output_seq_lengths, 'layer%d' % self.numlayers)

            if self.dropout < 1 and is_training:
                outputs = tf.nn.dropout(outputs, self.dropout)

        self.scope.reuse_variables()

        return outputs
