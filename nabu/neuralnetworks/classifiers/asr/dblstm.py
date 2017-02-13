'''@file dblstm.py
contains de DBLSTM class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier, layer, activation
from nabu.neuralnetworks import ops


class DBLSTM(classifier.Classifier):
    '''A deep bidirectional LSTM classifier'''

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        #the blstm layer
        blstm = layer.BLSTMLayer(int(self.conf['num_units']))

        #the linear output layer
        outlayer = layer.FFLayer(self.output_dim,
                                 activation.TfActivation(None, lambda(x): x), 0)

        #do the forward computation

        #add gaussian noise to the inputs
        if is_training:
            logits = inputs + tf.random_normal(inputs.get_shape(),
                                               stddev=0.6)
        else:
            logits = inputs

        for l in range(int(self.conf['num_layers'])):
            logits = blstm(logits, input_seq_length, 'layer' + str(l))

        logits = ops.seq2nonseq(logits, input_seq_length)

        logits = outlayer(logits, is_training, 'outlayer')

        logits = ops.nonseq2seq(logits, input_seq_length,
                                int(inputs.get_shape()[1]))

        return logits, input_seq_length
