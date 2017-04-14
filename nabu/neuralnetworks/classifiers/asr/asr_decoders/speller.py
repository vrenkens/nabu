'''@file speller.py
contains the speller functionality'''

from functools import partial
import tensorflow as tf
from nabu.neuralnetworks.classifiers.asr.asr_decoders import asr_decoder


class Speller(asr_decoder.AsrDecoder):
    '''a speller decoder for the LAS architecture'''

    def decode(self, hlfeat, encoder_inputs, initial_state, first_step,
               is_training):
        '''
        Get the logits and the output state

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        '''

        #one hot encode the targets
        one_hot_inputs = tf.one_hot(encoder_inputs, self.output_dim,
                                    dtype=tf.float32)

        #put targets in time major
        time_major_inputs = tf.transpose(one_hot_inputs, [1, 0, 2])

        #convert targets to list
        input_list = tf.unstack(time_major_inputs)

        #create the rnn cell
        rnn_cell = self.create_rnn(is_training)

        #create the loop functions
        lf = partial(loop_function, time_major_inputs,
                     float(self.conf['speller_sample_prob']))

        #use the attention decoder
        logit_list, state = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=input_list,
            initial_state=initial_state,
            attention_states=hlfeat,
            cell=rnn_cell,
            output_size=self.output_dim,
            loop_function=lf,
            scope='attention_decoder',
            initial_state_attention=not first_step)

        logits = tf.transpose(tf.stack(logit_list), [1, 0, 2])

        return logits, state

    def create_rnn(self, is_training=False):
        '''created the decoder rnn cell

        Args:
            is_training: whether or not the network is in training mode

        Returns:
            an rnn cell'''

        rnn_cells = []

        for _ in range(int(self.conf['speller_numlayers'])):

            #create the multilayered rnn cell
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                int(self.conf['speller_numunits']),
                reuse=tf.get_variable_scope().reuse)

            if float(self.conf['speller_dropout']) < 1 and is_training:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(
                    rnn_cell,
                    output_keep_prob=float(self.conf['speller_dropout']))

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

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

    next_input = tf.where(sample, next_input_sampled, next_input_truth)

    return next_input
