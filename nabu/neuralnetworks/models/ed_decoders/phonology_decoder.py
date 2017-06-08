'''@ file phonology_decoder.py
contains the PhonologyDecoder class'''

import numpy as np
import tensorflow as tf
import ed_decoder

class PhonologyDecoder(ed_decoder.EDDecoder):
    '''a decoder that is used to detect phonological features'''

    def _decode(self, encoded, encoded_seq_length, targets, target_seq_length,
                is_training):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors
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

        #splice the features
        with tf.name_scope('splice'):
            splicedlist = [encoded.values()[0]]
            for i in range(1, int(self.conf['context_window'])):
                left = tf.pad(
                    tensor=encoded.values()[0][:, :-i, :],
                    paddings=[[0, 0], [i, 0], [0, 0]])
                right = tf.pad(
                    tensor=encoded.values()[0][:, :-i, :],
                    paddings=[[0, 0], [0, i], [0, 0]])
                splicedlist += [left, right]

            spliced = tf.concat(splicedlist, axis=2)

        ph_dims = [int(d) for d in self.conf['phonology_dims'].split(' ')]

        #apply for each phonological feature
        detector_outputs = [
            detector(
                inputs=spliced,
                num_layers=int(self.conf['num_layers']),
                num_units=int(self.conf['num_units']),
                name='detector_%d' % i)
            for i, _ in enumerate(ph_dims)]

        #apply an output layer to each detector output
        features = [
            tf.contrib.layers.fully_connected(
                inputs=o,
                num_outputs=ph_dims[i],
                scope='outlayer_%d' % i)
            for i, o in enumerate(detector_outputs)]

        #read the mapping
        npmapping = np.load(self.conf['mapping']).transpose()
        mapping = tf.constant(npmapping, dtype=tf.float32)

        #map the features to the output
        output_name = self.output_dims.keys()[0]
        outputs = tf.tensordot(tf.concat(features, 2), mapping, axes=1)
        outputs = {output_name:outputs}

        output_seq_length = {output_name: encoded_seq_length.values()[0]}

        return outputs, output_seq_length, ()

    def _step(self, encoded, encoded_seq_length, targets, state, is_training):
        '''take a single decoding step

        encoded: the encoded inputs, this is a list of
            [batch_size x time x ...] tensors
        encoded_seq_length: the sequence lengths of the encoded inputs
            as a list of [batch_size] vectors
        targets: the targets decoded in the previous step as a dictionary of
            [batch_size] vectors
        state: the state of the previous deocding step as a possibly nested
            tupple of [batch_size x ...] vectors
        is_training: whether or not the network is in training mode.

        Returns:
            - the output logits of this decoding step as a dictionary of
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

    def get_output_dims(self, targetconfs, trainlabels):
        '''get the decoder output dimensions

        args:
            targetconfs: the target data confs
            trainlabels: the number of extra labels the trainer needs

        returns:
            a dictionary containing the output dimensions'''

        #get the dimensions of all the targets
        output_dims = {}
        for i, d in enumerate(self.conf['output_dims'].split(' ')):
            output_dims[self.outputs[i]] = d + trainlabels

        return output_dims

def detector(inputs, num_layers, num_units, name=None):
    '''add a feature detecting network

    args:
        inputs: a [batch_size x ...] tensor containing the inputs
        num_layers: the number of layers
        num_units: the number of units in each layer

    returns:
        a [batch_size x ...] tensor'''

    with tf.variable_scope(name or 'detector'):
        outputs = inputs
        for l in range(num_layers):
            outputs = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=num_units,
                scope='layer%d' % l
            )
    return outputs
