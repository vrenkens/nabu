'''@file listener.py
contains the listener code'''

import tensorflow as tf
import ed_encoder
from nabu.neuralnetworks.components import layer
from nabu.neuralnetworks.components import ops

class Listener(ed_encoder.EDEncoder):
    '''a listener object

    transforms input features into a high level representation'''

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



        encoded = {}
        encoded_seq_length = {}

        for inp in inputs:
            with tf.variable_scope(inp):
                #add input noise
                std_input_noise = float(self.conf['input_noise'])
                if is_training and std_input_noise > 0:
                    noisy_inputs = inputs[inp] + tf.random_normal(
                        tf.shape(inputs[inp]), stddev=std_input_noise)
                else:
                    noisy_inputs = inputs[inp]

                outputs = noisy_inputs
                output_seq_lengths = input_seq_length[inp]
                for l in range(int(self.conf['num_layers'])):
                    outputs, output_seq_lengths = layer.pblstm(
                        inputs=outputs,
                        sequence_length=output_seq_lengths,
                        num_units=int(self.conf['num_units']),
                        num_steps=int(self.conf['pyramid_steps']),
                        scope='layer%d' % l)

                    if float(self.conf['dropout']) < 1 and is_training:
                        outputs = tf.nn.dropout(
                            outputs, float(self.conf['dropout']))

                outputs = layer.blstm(
                    inputs=outputs,
                    sequence_length=output_seq_lengths,
                    num_units=int(self.conf['num_units']),
                    scope='layer%d' % int(self.conf['num_layers']))

                if float(self.conf['dropout']) < 1 and is_training:
                    outputs = tf.nn.dropout(outputs,
                                            float(self.conf['dropout']))

                encoded[inp] = outputs
                encoded_seq_length[inp] = output_seq_lengths

        return encoded, encoded_seq_length
