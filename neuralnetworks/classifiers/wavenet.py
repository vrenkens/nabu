'''@file wavenet.py
a wavenet classifier'''

import tensorflow as tf
from classifier import Classifier
import seq_convertors
from layer import FFLayer, GatedDilatedConvolution, Conv1dlayer
import activation

class Wavenet(Classifier):
    ''''a wavenet classifier'''

    def __init__(self, output_dim, num_layers, num_blocks, num_units,
                 kernel_size):
        '''
        Wavenet constructor

        Args:
            output_dim: the output dimension
            num_layers: number of dilated convolution layers per block
            num_blocks: number of dilated convolution blocks
            num_units: number of filters
        '''

        super(Wavenet, self).__init__(output_dim)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.kernel_size = kernel_size

    def __call__(self, inputs, input_seq_length, targets=None,
                 target_seq_length=None, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] dimansional vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length x 1] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] dimansional vector
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A quadruple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - a dictionary of control operations (may be empty)
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #create the input layer
            inlayer = Conv1dlayer(self.num_units, self.kernel_size, 1)

            #create the gated convolutional layers
            dconv = GatedDilatedConvolution(self.kernel_size)

            #create the fully connected layer
            act = activation.TfActivation(None, tf.nn.relu)
            fflayer = FFLayer(self.num_units, act)

            #create the output layer
            act = activation.TfActivation(None, lambda x: x)
            outlayer = FFLayer(self.output_dim, act)

            #apply the input layer
            logits = 0
            forward = inlayer(inputs, is_training, reuse, 'inlayer')

            #apply the the blocks of dilated convolutions layers
            for b in range(self.num_blocks):
                for l in range(self.num_layers):
                    forward, highway = dconv(forward, 2**l, is_training, reuse,
                                             'dconv%d-%d' % (b,l))
                    logits += highway

            #go to nonsequential data
            logits = seq_convertors.seq2nonseq(logits, input_seq_length)

            #apply the relu
            logits = tf.nn.relu(logits)

            #apply the fully connected layer
            logits = fflayer(logits, is_training, reuse, scope='FFlayer')

            #apply the output layer
            logits = outlayer(logits, is_training, reuse, scope='outlayer')

            #go back to sequential data
            logits = seq_convertors.nonseq2seq(logits, input_seq_length,
                                               int(inputs.get_shape()[1]))

            #create a saver
            saver = tf.train.Saver()

        return logits, input_seq_length, saver, None
