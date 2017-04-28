'''@file speller.py
contains the speller functionality'''

import tensorflow as tf
from nabu.neuralnetworks.models.ed_decoders import ed_decoder
from nabu.neuralnetworks.components import rnn_cell as rnn_cell_impl

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
        batch_size = tf.shape(targets[0])[0]

        #prepend a sequence border label to the targets to get the encoder
        #inputs, the label is the last label
        s_labels = tf.tile([[tf.constant(self.output_dims[0]-1,
                                         dtype=tf.int32)]],
                           [batch_size, 1])
        expanded_targets = tf.concat([s_labels, targets[0]], 1)

        #one hot encode the targets
        one_hot_targets = tf.one_hot(expanded_targets, self.output_dims[0],
                                     dtype=tf.float32)

        #create the rnn cell
        rnn_cell = self.create_cell(encoded, encoded_seq_length, is_training)

        #create the decoder helper
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=one_hot_targets,
            sequence_length=target_seq_length[0],
            embedding=lambda ids: tf.one_hot(ids, self.output_dims[0],
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
        logits, state = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = logits.rnn_output

        return [logits], [target_seq_length[0] + 1], state

    def _step(self, encoded, encoded_seq_length, targets, state, is_training):
        '''take a single decoding step

        encoded: the encoded inputs, this is a list of
            [batch_size x ...] tensors
        encoded_seq_length: the sequence lengths of the encoded inputs
            as a list of [batch_size] vectors
        targets: the targets decoded in the previous step as a list of
            [batch_size] vectors
        state: the state of the previous deocding step as a possibly nested
            tupple of [batch_size x ...] vectors
        is_training: whether or not the network is in training mode.

        Returns:
            - the output logits of this decoding step as a list of
                [batch_size x ...] tensors
            - the updated state as a possibly nested tupple of
                [batch_size x ...] vectors
        '''

        #one hot encode the targets
        one_hot_targets = tf.one_hot(targets[0], self.output_dims[0],
                                     dtype=tf.float32)

        #create the rnn cell
        rnn_cell = self.create_cell(encoded, encoded_seq_length, is_training)

        #use the rnn_cell
        logits, state = rnn_cell(one_hot_targets, state)

        return [logits], state

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

        rnn_cells = []

        for _ in range(int(self.conf['num_layers'])):

            #create the multilayered rnn cell
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                int(self.conf['num_units']),
                reuse=tf.get_variable_scope().reuse)

            if float(self.conf['dropout']) < 1 and is_training:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(
                    rnn_cell,
                    output_keep_prob=float(self.conf['dropout']))

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        #rnn_cell = rnn_cell_impl.StateOutputWrapper(rnn_cell)

        if encoded is not None:

            #create the attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=rnn_cell.output_size,
                memory=encoded[0],
                memory_sequence_length=encoded_seq_length[0]
            )

            #add attention to the rnn cell
            rnn_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                cell=rnn_cell,
                attention_mechanism=attention_mechanism,
                attention_size=rnn_cell.output_size
            )

            #make an output projection to get the correct output dimension
            rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(
                rnn_cell,
                self.output_dims[0],
                reuse=tf.get_variable_scope().reuse)

            #wrap the rnn cell to make it a constant scope
            rnn_cell = rnn_cell_impl.ScopeRNNCellWrapper(
                rnn_cell, self.scope.name + '/rnn_cell')

        return rnn_cell

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        rnn_cell = self.create_cell(None, None, False)
        cell_state = rnn_cell.zero_state(batch_size, tf.float32)
        attention = tf.zeros([batch_size, rnn_cell.output_size])
        zero_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
            cell_state=cell_state,
            attention=attention)

        return zero_state
