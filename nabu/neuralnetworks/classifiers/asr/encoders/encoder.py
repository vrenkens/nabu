'''@file listener.py
contains the Encoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Encoder(object):
    '''a general encoder object

    transforms input features into a high level representation'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, name=None):
        '''Listener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate
            name: the name of the Listener'''



        #save the parameters
        self.conf = conf

        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, sequence_lengths, is_training):
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

            outputs = self.encode(inputs, sequence_lengths, is_training)

        self.scope.reuse_variables()

        return outputs

    @abstractmethod
    def encode(self, inputs, sequence_lengths, is_training):
        '''
        get the high level feature representation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''
