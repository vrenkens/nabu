'''@file dnn.py
contains the RDNN class'''

import tensorflow as tf
import ed_encoder
from nabu.neuralnetworks.components import layer

class BLDNN(ed_encoder.EDEncoder):
    '''a RDNN encoder (blstm folowed by dnn)'''

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

        #splice the features
        spliced = {}
        for inp in inputs:
            times = [inputs[inp]]
            for i in range(1, int(self.conf['context'])):
                times.append(tf.pad(
                    tensor=inputs[inp][:, i:, :],
                    paddings=[[0, 0], [0, i], [0, 0]]))
                times.append(tf.pad(
                    tensor=inputs[inp][:, :-i, :],
                    paddings=[[0, 0], [i, 0], [0, 0]]))
            spliced[inp] = tf.concat(times, 2)


        #do the forward computation
        logits = {}
        for inp in spliced:
            with tf.variable_scope(inp):
                logits[inp] = spliced[inp]

                for i in range(int(self.conf['blstm_layers'])):

                    logits[inp] = layer.blstm(
                        inputs=logits[inp],
                        sequence_length=input_seq_length[inp],
                        num_units=int(self.conf['blstm_units']),
                        scope='blstm_layer' + str(i))

                    if float(self.conf['blstm_dropout']) < 1 and is_training:
                        logits[inp] = tf.nn.dropout(
                            logits[inp],
                            float(self.conf['blstm_dropout']))

                for i in range(int(self.conf['ff_layers'])):

                    logits[inp] = tf.contrib.layers.fully_connected(
                        inputs=logits[inp],
                        num_outputs=int(self.conf['ff_units']),
                        scope='ff_layer%d' % i)

                    if self.conf['layer_norm'] == 'True':
                        logits[inp] = tf.contrib.layers.layer_norm(logits[inp])

                    if float(self.conf['ff_dropout']) < 1 and is_training:
                        logits[inp] = tf.nn.dropout(
                            logits[inp],
                            float(self.conf['ff_dropout']))

        return logits, input_seq_length
