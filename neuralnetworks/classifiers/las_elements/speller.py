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


    def __call__(self, hlfeat, targets, numlabels, is_training=False,
                 reuse=False, scope=None):
        """
        Create the variables and do the forward computation in training mode

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            targets: the one-hot encoded training targets of shape
                [batch_size x target_seq_length x 1].
            numlabels: number of output labels
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                for variables with the same name in the graph and reuse
                these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the outputs of the listener
            the final state of the listener
        """

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #get the batch size
            batch_size = hlfeat.get_shape()[0]

            #create the multilayered rnn cell
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.numunits)

            if self.dropout < 1 and is_training:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, self.dropout)

            rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*self.numlayers)

            #one hot encode the targets
            one_hot_targets = tf.one_hot(tf.squeeze(targets), numlabels,
                                         dtype=tf.float32)

            #convert targets to list
            target_list = tf.unpack(tf.transpose(one_hot_targets, [1, 0, 2]))

            #use the attention decoder
            logit_list, _ = tf.nn.seq2seq.attention_decoder(
                decoder_inputs=target_list,
                initial_state=rnn_cell.zero_state(batch_size, tf.float32),
                attention_states=hlfeat,
                cell=rnn_cell,
                output_size=numlabels,
                scope='attention_decoder')

            logits = tf.transpose(tf.pack(logit_list), [1, 0, 2])

            return logits
