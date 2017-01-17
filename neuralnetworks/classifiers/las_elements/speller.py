'''@file speller.py
contains the speller functionality'''

import tensorflow as tf


class Speller(object):
    '''a speller object

    converts the high level features into output logits'''
    def __init__(self, numlayers, numunits, dropout=1):
        '''speller constructor

        Args:
            numlayers: number of layers in rge rnn
            numunits: number of units in each layer
            dropout: the dropout rate'''

        #save the parameters
        self.numlayers = numlayers
        self.numunits = numunits
        self.dropout = dropout


    def __call__(self, hlfeat, targets, numlabels, initial_state=None,
                 is_training=False, reuse=False, scope=None):
        """
        Create the variables and do the forward computation in training mode

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            targets: the one-hot encoded training targets of shape
                [batch_size x target_seq_length x 1].
            numlabels: number of output labels
            initial_state: the initial decoder state, could be usefull for
                decoding
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                for variables with the same name in the graph and reuse
                these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output logits of the listener
            the final state of the listener
        """

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #get the batch size
            batch_size = hlfeat.get_shape()[0]

            #squeezed targets
            squeezed_targets = tf.reshape(targets, targets.get_shape()[0:2])

            #one hot encode the targets
            one_hot_targets = tf.one_hot(squeezed_targets, numlabels,
                                         dtype=tf.float32)

            #convert targets to list
            target_list = tf.unpack(tf.transpose(one_hot_targets, [1, 0, 2]))

            #create the rnn cell
            rnn_cell = self.create_rnn(is_training)

            if initial_state is None:
                initial_state = rnn_cell.zero_state(batch_size, tf.float32)
                initial_state_attention = False
            else:
                initial_state_attention = True

            #use the attention decoder
            logit_list, state = tf.nn.seq2seq.attention_decoder(
                decoder_inputs=target_list,
                initial_state=initial_state,
                attention_states=hlfeat,
                cell=rnn_cell,
                output_size=numlabels,
                scope='attention_decoder',
                initial_state_attention=initial_state_attention)

            logits = tf.transpose(tf.pack(logit_list), [1, 0, 2])

            return logits, state

    def create_rnn(self, is_training=False):
        '''created the decoder rnn cell

        Args:
            is_training: whether or not the network is in training mode

        Returns:
            an rnn cell'''

        #create the multilayered rnn cell
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.numunits)

        if self.dropout < 1 and is_training:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, self.dropout)

        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*self.numlayers)

        return rnn_cell

    def zero_state(self, batch_size):
        '''get the listener zero state

        Returns:
            an rnn_cell zero state'''

        return self.create_rnn().zero_state(batch_size, tf.float32)
