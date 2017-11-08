'''@ file rnn_decoder.py
contains the RNNDecoder class'''

import tensorflow as tf
import ed_decoder
from nabu.neuralnetworks.components import rnn_cell as rnn_cell_impl

class RNNDecoder(ed_decoder.EDDecoder):
    '''a decoder that is used to detect phonological features'''

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

        #concatenate the vector of encoded inputs to the targets
        rnn_in = tf.tile(
            encoded.values()[0],
            [1, tf.shape(one_hot_targets)[1], 1])
        rnn_in = tf.concat([one_hot_targets, rnn_in], 2)

        #create the rnn cell
        rnn_cell = self.create_cell()

        #create the decoder helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=rnn_in,
            sequence_length=target_seq_length.values()[0]+1,
        )

        #create the decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=rnn_cell,
            helper=helper,
            initial_state=rnn_cell.zero_state(batch_size, tf.float32)
        )

        #use the decoder
        logits, state, logit_seq_length = tf.contrib.seq2seq.dynamic_decode(
            decoder)
        logits = logits.rnn_output

        return (
            {output_name: logits},
            {output_name: logit_seq_length},
            state)

    def create_cell(self):
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
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=int(self.conf['num_units']),
                layer_norm=self.conf['layer_norm'] == 'True',
                reuse=tf.get_variable_scope().reuse)

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

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

        rnn_cell = self.create_cell()
        cell_state = rnn_cell.zero_state(batch_size, tf.float32)
        attention = tf.zeros([batch_size, self.output_dims.values()[0]])
        zero_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
            cell_state=cell_state,
            attention=attention)

        return zero_state

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
