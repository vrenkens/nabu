'''@file dnn.py
contains the DNN class'''

import tensorflow as tf
import ed_encoder
from nabu.neuralnetworks.components import ops

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

                #stack the sequences for effecicience reasons
                logits[inp] = ops.stack_seq(logits[inp], input_seq_length[inp])
                for i in range(int(self.conf['num_layers'])):
                    logits[inp] = tf.contrib.layers.fully_connected(
                        inputs=logits[inp],
                        num_outputs=int(self.conf['num_units']),
                        scope='layer%d' % i)
                    if self.conf['layer_norm'] == 'True':
                        logits[inp] = tf.contrib.layers.layer_norm(logits[inp])
                    if float(self.conf['dropout']) < 1 and is_training:
                        logits[inp] = tf.nn.dropout(logits[inp],
                                                    float(self.conf['dropout']))
                #unstack the sequences again
                logits[inp] = ops.unstack_seq(
                    logits[inp], input_seq_length[inp])

        return logits, input_seq_length
