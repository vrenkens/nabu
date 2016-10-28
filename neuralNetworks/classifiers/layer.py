'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn

class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation, weights_std=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.weights_std = weights_std

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer\

        Returns:
            The output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                stddev = (self.weights_std if self.weights_std is not None
                          else 1/int(inputs.get_shape()[1])**0.5)

                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=tf.random_normal_initializer(stddev=stddev))

                biases = tf.get_variable(
                    'biases', [self.output_dim],
                    initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)

        return outputs

class BLSTMLayer(object):
    """This class allows enables blstm layer creation as well as computing
       their output. The output is found by linearly combining the forward
       and backward pass as described in:
       Graves et al., Speech recognition with deep recurrent neural networks,
       page 6646.
    """
    def __init__(self, num_units):
        """
        BlstmLayer constructor

        Args:
            num_units: The number of units in the LSTM
        """

        self.num_units = num_units

    def __call__(self, inputs, sequence_length, is_training=False,
                 reuse=None, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #create the forward and backward cells
            with tf.variable_scope('forward'):
                forward_lstm_block = rnn_cell.LSTMCell(self.num_units,
                                                       use_peepholes=True)
            with tf.variable_scope('backward'):
                backward_lstm_block = rnn_cell.LSTMCell(self.num_units,
                                                        use_peepholes=True)

            #do the forward computation
            outputs, _, _ = bidirectional_rnn(forward_lstm_block,
                                              backward_lstm_block,
                                              inputs,
                                              sequence_length=sequence_length)

            return outputs
