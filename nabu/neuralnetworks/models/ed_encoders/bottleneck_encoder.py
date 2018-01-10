'''@file bottleneck_encoder.py
contains the BottleneckEncoder'''

import tensorflow as tf
import ed_encoder
import ed_encoder_factory

class BottleneckEncoder(ed_encoder.EDEncoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, constraint, name=None):
        '''constructor

        Args:
            conf: the encoder configuration
            name: the encoder name
            constraint: the constraint for the variables
        '''


        super(BottleneckEncoder, self).__init__(conf, constraint, name)

        #set the wrapped section as the encoder section
        conf.remove_section('encoder')
        conf.add_section('encoder')
        for option, value in conf.items(self.conf['wrapped']):
            conf.set('encoder', option, value)
        conf.remove_section(self.conf['wrapped'])

        #wrapped ecoder
        self.wrapped = ed_encoder_factory.factory(
            conf.get('encoder', 'encoder'))(
                conf, constraint, self.conf['wrapped'])




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

        #apply a bottleneck layer to all encoded
        for e in encoded:
            encoded[e] = tf.contrib.layers.linear(
                inputs=encoded[e],
                num_outputs=int(self.conf['output_dim']),
                scope='%s_bottleneck' % e
            )

        return encoded, encoded_seq_length
