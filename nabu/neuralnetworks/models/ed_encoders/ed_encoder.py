'''@file ed_encoder.py
contains the EDEncoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class EDEncoder(object):
    '''a general encoder for an encoder decoder system

    transforms input features into a high level representation'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, name=None):
        '''EDEncoder constructor

        Args:
            conf: the encoder configuration
            name: the encoder name'''

        #save the configuration
        self.conf = conf

        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a list of [bath_size x time x ...]
                tensors
            - the sequence lengths of the outputs as a list of [batch_size]
                tensors
        '''

        with tf.variable_scope(self.scope):

            outputs, output_seq_length = self.encode(inputs, input_seq_length,
                                                     is_training)

        self.scope.reuse_variables()

        return outputs, output_seq_length

    @abstractmethod
    def encode(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a list of [bath_size x time x ...]
                tensors
            - the sequence lengths of the outputs as a list of [batch_size]
                tensors
        '''

    @property
    def variables(self):
        '''get a list of the models's variables'''

        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope.name)
