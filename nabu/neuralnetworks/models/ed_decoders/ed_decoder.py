'''@file asr_decoder.py
contains the EDDecoder class'''

import os
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.tools.default_conf import apply_defaults

class EDDecoder(object):
    '''a general decoder for an encoder decoder system

    converts the high level features into output logits'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dims, constraint, name=None):
        '''EDDecoder constructor

        Args:
            conf: the decoder configuration as a ConfigParser
            output_dims: a dictionary containing the output dimensions for each
                output
            constraint: the constraint for the variables
        '''


        #save the parameters
        self.conf = dict(conf.items('decoder'))

        #apply default configuration
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')
        apply_defaults(self.conf, default)

        self.outputs = output_dims.keys()

        self.output_dims = output_dims

        self.scope = tf.VariableScope(
            tf.AUTO_REUSE,
            name or type(self).__name__,
            constraint=constraint)


    def __call__(self, encoded, encoded_seq_length, targets, target_seq_length,
                 is_training):

        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a dictionary of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a dictionary of [batch_size] vectors
            targets: the targets used as decoder inputs as a dictionary of
                [batch_size x time x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

        with tf.variable_scope(self.scope):

            logits, logit_sequence_length, state = self._decode(
                encoded,
                encoded_seq_length,
                targets,
                target_seq_length,
                is_training)

        return logits, logit_sequence_length, state

    @abstractmethod
    def _decode(self, encoded, encoded_seq_length, targets, target_seq_length,
                is_training):

        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a dictionary of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a dictionary of [batch_size] vectors
            targets: the targets used as decoder inputs as a dictionary of
                [batch_size x time x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

    @abstractmethod
    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

    @property
    def variables(self):
        '''get a list of the models's variables'''

        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables
