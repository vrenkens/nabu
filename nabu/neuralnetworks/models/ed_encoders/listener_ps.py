'''@file listener_ps.py
contains the ListenerPS class'''

import tensorflow as tf
import ed_encoder
from nabu.neuralnetworks.components import layer

class ListenerPS(ed_encoder.EDEncoder):
    '''a listener with projected subsampling

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''ListenerPS constructor

        Args:
            conf: the encoder configuration
            name: the encoder name'''

        #create the blstm layer
        self.blstm = layer.BLSTMLayer(int(conf['num_units']))

        super(ListenerPS, self).__init__(conf, name)

    def encode(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a list of [bath_size x ...]
                tensors
            - the sequence lengths of the outputs as a list of [batch_size]
                tensors
        '''


        #add input noise
        std_input_noise = float(self.conf['input_noise'])
        if is_training and std_input_noise > 0:
            noisy_inputs = inputs[0] + tf.random_normal(
                tf.shape(inputs[0]), stddev=std_input_noise)
        else:
            noisy_inputs = inputs[0]

        outputs = noisy_inputs
        output_seq_lengths = input_seq_length[0]
        for l in range(int(self.conf['num_layers'])):
            with tf.variable_scope('layer%d' % l):
                outputs = self.blstm(outputs, output_seq_lengths)

                #apply projected subsampling
                outputs, output_seq_lengths = layer.projected_subsampling(
                    inputs=outputs,
                    input_seq_lengths=output_seq_lengths,
                    num_steps=int(self.conf['pyramid_steps'])
                )

                #apply batch normalization
                outputs = tf.layers.batch_normalization(outputs,
                                                        training=is_training)

                outputs = tf.nn.relu(outputs)

        #apply final blstm layer
        outputs = self.blstm(
            outputs, output_seq_lengths,
            'layer%d' % int(self.conf['num_layers']))

        return [outputs], [output_seq_lengths]
