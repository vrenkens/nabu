'''@file speller.py
contains the speller functionality'''

import tensorflow as tf
from nabu.neuralnetworks.models.ed_decoders import rnn_decoder
from nabu.neuralnetworks.components.rnn_cell import StateOutputWrapper
from nabu.neuralnetworks.components import attention
from nabu.neuralnetworks.components.rnn_cell import AttentionWrapper

class Speller(rnn_decoder.RNNDecoder):
    '''a speller decoder for the LAS architecture'''

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
            rnn_cell = tf.contrib.rnn.LSTMCell(
                num_units=int(self.conf['num_units']),
                reuse=tf.get_variable_scope().reuse)

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        if self.conf['statequery'] == 'True':
            rnn_cell = StateOutputWrapper(rnn_cell)


        if encoded is not None:

            #create the attention mechanism
            attention_mechanisms = [attention.factory(
                conf=self.conf,
                num_units=rnn_cell.output_size,
                encoded=encoded[e],
                encoded_seq_length=encoded_seq_length[e]
            ) for e in encoded]

            attention_layer_size = \
                [int(self.conf['num_units'])]*len(attention_mechanisms)

            #add attention to the rnn cell
            rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=rnn_cell,
                attention_mechanism=attention_mechanisms,
                attention_layer_size=attention_layer_size,
                alignment_history=False,
                output_attention=True
            )

        #the output layer
        rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell=rnn_cell,
            output_size=self.output_dims.values()[0],
            reuse=tf.get_variable_scope().reuse
        )

        return rnn_cell
