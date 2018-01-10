'''@file dblstm.py
contains de DBLSTM class'''

import tensorflow as tf
import ed_encoder
from nabu.neuralnetworks.components import layer

class DBLSTM(ed_encoder.EDEncoder):
    '''A deep bidirectional LSTM classifier'''

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

        #do the forward computation

        encoded = {}
        encoded_seq_length = {}

        for inp in inputs:
            with tf.variable_scope(inp):
                #add gaussian noise to the inputs
                if is_training and float(self.conf['input_noise']) > 0:
                    logits = inputs[inp] + tf.random_normal(
                        tf.shape(inputs[inp]),
                        stddev=float(self.conf['input_noise']))
                else:
                    logits = inputs[inp]

                for l in range(int(self.conf['num_layers'])):

                    logits = layer.blstm(
                        inputs=logits,
                        sequence_length=input_seq_length[inp],
                        num_units=int(self.conf['num_units']),
                        scope='layer' + str(l))

                if is_training and float(self.conf['dropout']) < 1:
                    logits = tf.nn.dropout(logits, float(self.conf['dropout']))

                encoded[inp] = logits
                encoded_seq_length[inp] = input_seq_length[inp]

        return encoded, encoded_seq_length
