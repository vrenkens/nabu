'''@ file linear_decoder.py
contains the LinearDecoder class'''

import tensorflow as tf
import ed_decoder

class LinearDecoder(ed_decoder.EDDecoder):
    '''a linear decoder that maps the encoded input sequences to outputs

    Uses a single linear layer'''

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

        outputs = []
        for i, e in enumerate(encoded):

            #apply the linear layer
            outputs.append(tf.contrib.layers.linear(
                e, self.output_dims[i], scope='outlayer%d' % i))

        return outputs, encoded_seq_length, ()

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

        return self(encoded, encoded_seq_length, targets, None,
                    is_training)

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        return ()
