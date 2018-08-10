'''@file rnn_decoder.py
contains the general recurrent decoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.neuralnetworks.models.ed_decoders import ed_decoder

class RNNDecoder(ed_decoder.EDDecoder):
    '''a speller decoder for the LAS architecture'''

    __metaclass__ = ABCMeta

    def _decode(self, encoded, encoded_seq_length, targets, target_seq_length,
                is_training):

        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors
            targets: the targets used as decoder inputs as a list of
                [batch_size x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a list of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

        #get the batch size
        batch_size = tf.shape(targets.values()[0])[0]
        output_dim = self.output_dims.values()[0]
        output_name = self.output_dims.keys()[0]

        #prepend a sequence border label to the targets to get the encoder
        #inputs
        expanded_targets = tf.pad(targets.values()[0], [[0, 0], [1, 0]],
                                  constant_values=output_dim-1)

        #create the rnn cell
        rnn_cell = self.create_cell(encoded, encoded_seq_length, is_training)

        #create the embedding
        embedding = lambda ids: tf.one_hot(
            ids,
            output_dim,
            dtype=tf.float32)

        #create the decoder helper
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=embedding(expanded_targets),
            sequence_length=target_seq_length.values()[0],
            embedding=embedding,
            sampling_probability=float(self.conf['sample_prob'])
        )

        #create the decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=rnn_cell,
            helper=helper,
            initial_state=rnn_cell.zero_state(batch_size, tf.float32)
        )

        #use the decoder
        logits, state, logit_seq_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True)
        logits = logits.rnn_output

        return (
            {output_name: logits},
            {output_name: logit_seq_length},
            state)

    @abstractmethod
    def create_cell(self, encoded, encoded_seq_length, is_training):
        '''create the rnn cell

        Args:
            encoded: the encoded sequences as a [batch_size x max_time x dim]
                tensor that will be queried with attention
                set to None if the rnn_cell should be created without the
                attention part (for zero_state)
            encoded_seq_length: the encoded sequence lengths as a [batch_size]
                vector
            is_training: bool whether or not the network is in training mode

        Returns:
            an RNNCell object'''

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded dict of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''



        encoded = {name:tf.zeros([batch_size, 0, encoded_dim[name]])
                   for name in encoded_dim}

        rnn_cell = self.create_cell(
            encoded,
            tf.zeros([batch_size]),
            False)

        return rnn_cell.zero_state(batch_size, tf.float32)

    def __getstate__(self):
        '''getstate'''

        return self.__dict__
