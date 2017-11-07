'''@file hotstart_decoder.py
contains the HotstartDecoder'''

import tensorflow as tf
import ed_decoder
import ed_decoder_factory

class HotstartDecoder(ed_decoder.EDDecoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, trainlabels, outputs, name=None):
        '''EDDecoder constructor

        Args:
            conf: the decoder configuration as a configparser
            trainlabels: the number of extra labels required by the trainer
            outputs: the name of the outputs of the model
        '''

        #super constructor
        super(HotstartDecoder, self).__init__(conf, trainlabels, outputs, name)

        #set the wrapped section as the decoder section
        conf.remove_section('decoder')
        conf.add_section('decoder')
        for option, value in conf.items(self.conf['wrapped']):
            conf.set('decoder', option, value)
        conf.remove_section(self.conf['wrapped'])

        #create the wrapped decoder
        self.wrapped = ed_decoder_factory.factory(
            conf.get('decoder', 'decoder'))(
                conf, trainlabels, outputs, self.conf['wrapped'])



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

        #call the wrapped decoder
        logits, lengths, state = self.wrapped(
            encoded,
            encoded_seq_length,
            targets,
            target_seq_length,
            is_training)

        for var in self.wrapped.variables:
            value = tf.contrib.framework.load_variable(
                self.conf['modeldir'],
                var.name)

            if self.conf['trainable'] == 'False':
                tf.add_to_collection('untrainable', var)

            #pylint: disable=W0212
            var._initializer_op = var.assign(value).op


        return logits, lengths, state

    def _step(self, encoded, encoded_seq_length, targets, state, is_training):
        '''take a single decoding step

        encoded: the encoded inputs, this is a dictionary of
            [batch_size x time x ...] tensors
        encoded_seq_length: the sequence lengths of the encoded inputs
            as a dictionary of [batch_size] vectors
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

        #pylint: disable=W0212
        return self.wrapped._step(
            encoded, encoded_seq_length, targets, state, is_training)

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        return self.wrapped.zero_state(encoded_dim, batch_size)

    def get_output_dims(self, trainlabels):
        '''get the decoder output dimensions

        args:
            trainlabels: the number of extra labels the trainer needs

        returns:
            a dictionary containing the output dimensions'''

        return self.wrapped.get_output_dims(trainlabels)
