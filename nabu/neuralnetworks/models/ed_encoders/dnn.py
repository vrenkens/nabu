'''@file dnn.py
contains the DNN class'''

import tensorflow as tf
import ed_encoder

class DNN(ed_encoder.EDEncoder):
    '''a DNN encoder'''

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
            - the outputs of the encoder as a list of [bath_size x time x ...]
                tensors
            - the sequence lengths of the outputs as a list of [batch_size]
                tensors
        '''

        #create the activation function
        if self.conf['activation'] == 'relu':
            activation_fn = tf.nn.relu
        elif self.conf['activation'] == 'tanh':
            activation_fn = tf.nn.tanh
        else:
            raise Exception('unexpected activation function %s' %
                            self.conf['activation'])

        #do the forward computation
        logits = {}
        for inp in inputs:
            with tf.variable_scope(inp):
                logits[inp] = inputs[inp]
                for i in range(int(self.conf['num_layers'])):
                    logits[inp] = tf.contrib.layers.linear(
                        inputs=logits[inp],
                        num_outputs=int(self.conf['num_units']),
                        activation_fn=activation_fn,
                        scope='layer%d' % i)
                    if float(self.conf['dropout']) < 1 and is_training:
                        logits[inp] = tf.nn.dropout(logits[inp],
                                                    float(self.conf['dropout']))

        return logits, input_seq_length
