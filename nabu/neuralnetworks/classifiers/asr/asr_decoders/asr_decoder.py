'''@file asr_decoder.py
contains the AsrDecoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class AsrDecoder(object):
    '''a general asr decoder object

    converts the high level features into output logits'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim, name=None):
        '''speller constructor

        Args:
            conf: the classifier config as a dictionary
            output_dim: the classifier output dimension
            name: the speller name'''


        #save the parameters
        self.conf = conf
        self.output_dim = output_dim

        self.scope = tf.VariableScope(False, name or type(self).__name__)


    def __call__(self, hlfeat, encoder_inputs, initial_state, first_step,
                 is_training):
        '''
        Create the variables and do the forward computation

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        '''

        with tf.variable_scope(self.scope):

            logits, state = self.decode(hlfeat, encoder_inputs, initial_state,
                                        first_step, is_training)

        self.scope.reuse_variables()

        return logits, state

    @abstractmethod
    def decode(self, hlfeat, encoder_inputs, initial_state, first_step,
               is_training):
        '''
        Get the logits and the output state

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        '''

    @abstractmethod
    def zero_state(self, batch_size):
        '''get the decoder zero state

        Returns:
            an rnn_cell zero state'''
