'''@file lstm_decoder.py
contains the LstmDecoder class'''

import tensorflow as tf
import ed_decoder

class LstmDecoder(ed_decoder.EDDecoder):
    '''The LSTM decoder, that predicts the next labels based on the history'''

    def decode(self,
               encoded,
               encoded_seq_length,
               targets,
               target_seq_length,
               initial_state,
               continued,
               is_training):
        '''
        Create the variables and do the forward computation

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encded inputs
                as a list of [batch_size] vectors
            targets: during training these are the targets during decoding,
                this is the previously decoded target
            target_seq_length: the sequence lengths of the targets
                as a list of [batch_size] vectors
            initial_state: the initial decoder state, could be usefull for
                decoding
            continued: bool that determines if the decoder is continuing to
                decode a sequence or not
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

        #get the batch size
        batch_size = tf.shape(targets[0])[0]

        #prepend a sequence border label to the targets to get the encoder
        #inputs, the label is the last label
        if not continued:
            s_labels = tf.tile([[tf.constant(self.output_dims[0]-1,
                                             dtype=tf.int32)]],
                               [batch_size, 1])
            expanded_targets = tf.concat([s_labels, targets[0]], 1)
        else:
            expanded_targets = targets[0]

        #one hot encode the targets
        one_hot_targets = tf.one_hot(expanded_targets, self.output_dims[0],
                                     dtype=tf.float32)

        #put targets in time major
        time_major_targets = tf.transpose(one_hot_targets, [1, 0, 2])

        #convert targets to list
        target_list = tf.unstack(time_major_targets)

        #create the rnn cell
        rnn_cell = self.create_rnn(is_training)

        if initial_state is None:
            initial_state = rnn_cell.zero_state(batch_size, tf.float32)

        #use the attention decoder
        logit_list, state = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=target_list,
            initial_state=initial_state,
            cell=rnn_cell,
            scope='rnn_decoder')

        logits = tf.transpose(tf.stack(logit_list), [1, 0, 2])

        logits = tf.contrib.layers.linear(logits, self.output_dims[0])

        return [logits], [target_seq_length[0] + 1], state

    def create_rnn(self, is_training=False):
        '''created the decoder rnn cell

        Args:
            is_training: whether or not the network is in training mode

        Returns:
            an rnn cell'''

        #create the multilayered rnn cell
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(
            int(self.conf['num_units']))

        if float(self.conf['dropout']) < 1 and is_training:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=float(self.conf['dropout']))

        rnn_cell = tf.contrib.rnn.MultiRNNCell(
            [rnn_cell]*int(self.conf['num_layers']))

        return rnn_cell

    def zero_state(self, batch_size):
        '''get the decoder zero state

        Args:
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        return self.create_rnn().zero_state(batch_size, tf.float32)
