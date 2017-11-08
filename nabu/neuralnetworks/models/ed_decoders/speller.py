'''@file speller.py
contains the speller functionality'''

import tensorflow as tf
from nabu.neuralnetworks.models.ed_decoders import ed_decoder

class Speller(ed_decoder.EDDecoder):
    '''a speller decoder for the LAS architecture'''

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
        #inputs, the label is the last label
        s_labels = tf.tile([[tf.constant(output_dim-1,
                                         dtype=tf.int32)]],
                           [batch_size, 1])
        expanded_targets = tf.concat([s_labels, targets.values()[0]], 1)

        #one hot encode the targets
        one_hot_targets = tf.one_hot(expanded_targets, output_dim,
                                     dtype=tf.float32)

        #create the rnn cell
        rnn_cell = self.create_cell(encoded, encoded_seq_length, is_training)

        #create the decoder helper
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=one_hot_targets,
            sequence_length=target_seq_length.values()[0]+1,
            embedding=lambda ids: tf.one_hot(ids, output_dim,
                                             dtype=tf.float32),
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

    def create_cell(self, encoded, encoded_seq_length, _):
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

        rnn_cells = []

        for _ in range(int(self.conf['num_layers'])):

            #create the multilayered rnn cell
            rnn_cell = tf.contrib.rnn.LSTMCell(
                num_units=int(self.conf['num_units']),
                reuse=tf.get_variable_scope().reuse)

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        if encoded is not None:

            #create the attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=rnn_cell.output_size,
                memory=encoded.values()[0],
                memory_sequence_length=encoded_seq_length.values()[0]
            )

            #add attention to the rnn cell
            rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=rnn_cell,
                attention_mechanism=attention_mechanism
            )

        #add an output layer to the rnn cell
        rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(
            rnn_cell,
            output_size=self.output_dims.values()[0],
            reuse=tf.get_variable_scope().reuse)

        return rnn_cell

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

    def get_output_dims(self, trainlabels):
        '''get the decoder output dimensions

        args:
            trainlabels: the number of extra labels the trainer needs

        returns:
            a dictionary containing the output dimensions'''

        #get the dimensions of all the targets
        output_dims = {}
        for i, d in enumerate(self.conf['output_dims'].split(' ')):
            output_dims[self.outputs[i]] = int(d) + trainlabels

        return output_dims
