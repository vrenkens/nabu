'''@file lstm_decoder.py
contains the LstmDecoder class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers.layer import Linear

class LstmDecoder(object):
    '''The decoder object used in a LSTM language model'''
    def __init__(self, numlayers, numunits, dropout=1, name=None):
        '''speller constructor

        Args:
            numlayers: number of layers in rge rnn
            numunits: number of units in each layer
            dropout: the dropout rate
            name: the speller name'''


        #save the parameters
        self.numlayers = numlayers
        self.numunits = numunits
        self.dropout = dropout

        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, encoder_inputs, numlabels, initial_state=None,
                 is_training=False):
        '''
        Create the variables and do the forward computation in training mode

        Args:
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].
            numlabels: number of output labels
            initial_state: the initial decoder state, could be usefull for
                decoding
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        '''

        with tf.variable_scope(self.scope):

            #get the batch size
            batch_size = encoder_inputs.get_shape()[0]

            #one hot encode the targets
            one_hot_inputs = tf.one_hot(encoder_inputs, numlabels,
                                        dtype=tf.float32)

            #put targets in time major
            time_major_inputs = tf.transpose(one_hot_inputs, [1, 0, 2])

            #convert targets to list
            input_list = tf.unstack(time_major_inputs)

            #create the rnn cell
            rnn_cell = self.create_rnn(is_training)

            #create the output layer
            outlayer = Linear(numlabels)

            if initial_state is None:
                initial_state = rnn_cell.zero_state(batch_size, tf.float32)

            #use the attention decoder
            logit_list, state = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs=input_list,
                initial_state=initial_state,
                cell=rnn_cell,
                scope='rnn_decoder')

            logits = tf.transpose(tf.stack(logit_list), [1, 0, 2])

            logits = outlayer(logits)

        self.scope.reuse_variables()

        return logits, state

    def create_rnn(self, is_training=False):
        '''created the decoder rnn cell

        Args:
            is_training: whether or not the network is in training mode

        Returns:
            an rnn cell'''

        rnn_cells = []

        for _ in range(self.numlayers):

            #create the multilayered rnn cell
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                self.numunits,
                reuse=tf.get_variable_scope().reuse)

            if self.dropout < 1 and is_training:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(
                    rnn_cell,
                    output_keep_prob=self.dropout)

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        return rnn_cell

    def zero_state(self, batch_size):
        '''get the listener zero state

        Args:
            batch_size: the batch size

        Returns:
            an rnn_cell zero state'''

        return self.create_rnn().zero_state(batch_size, tf.float32)
