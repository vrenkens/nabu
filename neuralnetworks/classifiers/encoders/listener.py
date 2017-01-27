'''@file listener.py
contains the listener code'''

import tensorflow as tf
import neuralnetworks

class Listener(object):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, numlayers, numunits, dropout=1):
        '''Listener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate'''

        #save the parameters
        self.numlayers = numlayers
        self.dropout = dropout

        #create the pblstm layer
        self.pblstm = neuralnetworks.classifiers.layer.PBLSTMLayer(numunits)

    def __call__(self, inputs, sequence_lengths, is_training=False, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        """

        with tf.variable_scope(scope or type(self).__name__):

            outputs = inputs
            output_seq_lengths = sequence_lengths
            for l in range(self.numlayers):
                outputs, output_seq_lengths = self.pblstm(
                    outputs, output_seq_lengths, 'layer%d' % l)

                if self.dropout < 1 and is_training:
                    outputs = tf.nn.dropout(outputs, self.dropout)

        return outputs
