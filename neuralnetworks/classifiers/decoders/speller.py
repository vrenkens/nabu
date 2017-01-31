'''@file speller.py
contains the speller functionality'''

from functools import partial
import tensorflow as tf


class Speller(object):
    '''a speller object

    converts the high level features into output logits'''
    def __init__(self, numlayers, numunits, dropout=1, sample_prob=0):
        '''speller constructor

        Args:
            numlayers: number of layers in rge rnn
            numunits: number of units in each layer
            dropout: the dropout rate
            sample_prob: the probability that the network will sample from the
                output during training'''


        #save the parameters
        self.numlayers = numlayers
        self.numunits = numunits
        self.dropout = dropout
        self.sample_prob = sample_prob


    def __call__(self, hlfeat, encoder_inputs, numlabels, initial_state=None,
                 initial_state_attention=False, is_training=False,
                 scope=None):
        """
        Create the variables and do the forward computation in training mode

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length (x 1)].
            numlabels: number of output labels
            initial_state: the initial decoder state, could be usefull for
                decoding
            initial_state_attention: whether attention has to be applied
                to the initital state to ge an initial context
            is_training: whether or not the network is in training mode
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        """

        with tf.variable_scope(scope or type(self).__name__):

            #get the batch size
            batch_size = hlfeat.get_shape()[0]

            #squeezed targets
            squeezed_inputs = tf.reshape(
                encoder_inputs,
                encoder_inputs.get_shape()[0:2])

            #one hot encode the targets
            one_hot_inputs = tf.one_hot(squeezed_inputs, numlabels,
                                        dtype=tf.float32)

            #put targets in time major
            time_major_inputs = tf.transpose(one_hot_inputs, [1, 0, 2])

            #convert targets to list
            input_list = tf.unpack(time_major_inputs)

            #create the rnn cell
            rnn_cell = self.create_rnn(is_training)

            if initial_state is None:
                initial_state = rnn_cell.zero_state(batch_size, tf.float32)

            #create the loop functions
            lf = partial(loop_function, time_major_inputs, self.sample_prob)

            #use the attention decoder
            logit_list, state = tf.nn.seq2seq.attention_decoder(
                decoder_inputs=input_list,
                initial_state=initial_state,
                attention_states=hlfeat,
                cell=rnn_cell,
                output_size=numlabels,
                loop_function=lf,
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
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                rnn_cell, output_keep_prob=self.dropout)

        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*self.numlayers)

        return rnn_cell

    def zero_state(self, batch_size):
        '''get the listener zero state

        Returns:
            an rnn_cell zero state'''

        return self.create_rnn().zero_state(batch_size, tf.float32)

def loop_function(decoder_inputs, sample_prob, prev, i):
    '''the loop function used in the attention decoder_inputs, used for
    scheduled sampling

    Args:
        decoder_inputs: the ground truth labels as a tensor of shape
            [seq_length, batch_size, numlabels] (time_major)
        sample_prob: the probability that the network will sample the output
        prev: the outputs of the previous steps
        i: the current decoding step

    returns:
        the input for the nect time step
    '''

    batch_size = int(decoder_inputs.get_shape()[1])
    numlabels = decoder_inputs.get_shape()[2]

    #get the most likely characters as the sampled output
    next_input_sampled = tf.one_hot(tf.argmax(prev, 1), numlabels)

    #get the current ground truth labels
    next_input_truth = tf.gather(decoder_inputs, i)

    #creat a boolean vector of where to sample
    sample = tf.less(tf.random_uniform([batch_size]), sample_prob)

    next_input = tf.select(sample, next_input_sampled, next_input_truth)

    return next_input
