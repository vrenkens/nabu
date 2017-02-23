'''@file ff_listener.py
contains the FfListener class'''

import tensorflow as tf
import encoder
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks.ops import pyramid_stack

class FfListener(encoder.Encoder):
    '''a feedforward listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''FfListener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate
            name: the name of the Listener'''


        #create the linear layer
        self.layer = layer.Linear(int(conf['listener_numunits']))

        super(FfListener, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        '''
        get the high level feature representation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''


        outputs = inputs
        output_seq_lengths = sequence_lengths
        for l in range(int(self.conf['listener_numlayers'])):

            #apply the linear layer
            outputs = self.layer(outputs, 'layer%d' % l)

            #apply the nonlinearity
            outputs = tf.nn.relu(outputs)

            #apply the pyramid stack
            outputs, output_seq_lengths = pyramid_stack(outputs,
                                                        output_seq_lengths)

            if float(self.conf['listener_dropout']) < 1 and is_training:
                outputs = tf.nn.dropout(
                    outputs, float(self.conf['listener_dropout']))

        outputs = self.layer(outputs,
                             'layer%d' % int(self.conf['listener_numlayers']))

        if float(self.conf['listener_dropout']) < 1 and is_training:
            outputs = tf.nn.dropout(outputs,
                                    float(self.conf['listener_dropout']))

        return outputs
