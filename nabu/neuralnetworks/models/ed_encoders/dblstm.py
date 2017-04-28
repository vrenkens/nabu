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

        #the blstm layer
        blstm = layer.BLSTMLayer(int(self.conf['num_units']))

        #do the forward computation

        #add gaussian noise to the inputs
        if is_training and float(self.conf['input_noise']) > 0:
            logits = inputs[0] + tf.random_normal(
                tf.shape(inputs[0]),
                stddev=float(self.conf['input_noise']))
        else:
            logits = inputs[0]

        for l in range(int(self.conf['num_layers'])):
            logits = blstm(logits, input_seq_length[0], 'layer' + str(l))

            logits = tf.nn.dropout(logits, float(self.conf['dropout']))

        return [logits], input_seq_length
