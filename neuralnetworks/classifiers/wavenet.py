'''@file wavenet.py
a wavenet classifier'''

import tensorflow as tf
from classifier import Classifier
from layer import GatedAConv1d, Conv1dLayer

class Wavenet(Classifier):
    ''''a wavenet classifier'''

    def __init__(self, output_dim, num_layers, num_blocks, num_units,
                 kernel_size, causal):
        '''
        Wavenet constructor

        Args:
            output_dim: the output dimension
            num_layers: number of dilated convolution layers per block
            num_blocks: number of dilated convolution blocks
            num_units: number of filters
            causal: flag for causality, if true every output will only be
                affected by previous inputs
        '''

        super(Wavenet, self).__init__(output_dim)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.kernel_size = kernel_size
        self.causal = causal

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

            #create the gated convolutional layers
            dconv = GatedAConv1d(self.kernel_size)

            #create the one by one convolution layer
            onebyone = Conv1dLayer(self.num_units, 1, 1)

            #create the output layer
            outlayer = Conv1dLayer(self.output_dim, 1, 1)

            #add gaussian noise to the inputs
            if is_training:
                forward = inputs + tf.random_normal(inputs.get_shape(), stddev=0.6)
            else:
                forward = inputs

            #apply the input layer
            logits = 0
            forward = onebyone(forward, input_seq_length, is_training, reuse,
                               'inlayer')
            forward = tf.nn.tanh(forward)

            #apply the the blocks of dilated convolutions layers
            for b in range(self.num_blocks):
                for l in range(self.num_layers):
                    forward, highway = dconv(
                        forward, input_seq_length, self.causal, 2**l,
                        is_training, reuse, 'dconv%d-%d' % (b, l))
                    logits += highway

            #apply the relu
            logits = tf.nn.relu(logits)

            #apply the one by one convloution
            logits = onebyone(logits, input_seq_length, is_training, reuse,
                              '1x1')
            logits = tf.nn.relu(logits)

            #apply the output layer
            logits = outlayer(logits, input_seq_length, is_training, reuse,
                              'outlayer')

            #create a saver
            saver = tf.train.Saver()

        return logits, input_seq_length, saver, None
