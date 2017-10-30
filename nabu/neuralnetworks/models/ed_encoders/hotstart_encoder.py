'''@file hotstart_encoder.py
contains the HotstartEncoder'''

import tensorflow as tf
import ed_encoder
import ed_encoder_factory

class HotstartEncoder(ed_encoder.EDEncoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor

        Args:
            conf: the encoder configuration
            name: the encoder name'''


        #wrapped ecoder
        self.wrapped = ed_encoder_factory.factory(conf['wrapped'])(
            conf)

        super(HotstartEncoder, self).__init__(conf, name)

    def encode(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a dictionary of
                [bath_size x time x ...] tensors
            - the sequence lengths of the outputs as a dictionary of
                [batch_size] tensors
        '''

        #call the wrapped encoder
        encoded, encoded_seq_length = self.wrapped(
            inputs, input_seq_length, is_training)

        for var in self.wrapped.variables:
            value = tf.contrib.framework.load_variable(
                self.conf['modeldir'],
                var.name)

            if self.conf['trainable'] == 'False':
                tf.add_to_collection('untrainable', var)

            #pylint: disable=W0212
            var._initializer_op = var.assign(value).op


        return encoded, encoded_seq_length
