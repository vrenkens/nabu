'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from neuralnetworks import ops

class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, act, weights_std=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            act: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = act
        self.weights_std = weights_std

    def __call__(self, inputs, is_training=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            scope: the variable scope of the layer

        Returns:
            The output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('parameters'):

                stddev = (self.weights_std if self.weights_std is not None
                          else 1/int(inputs.get_shape()[1])**0.5)

                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=tf.random_normal_initializer(stddev=stddev))

                biases = tf.get_variable(
                    'biases', [self.output_dim],
                    initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear'):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation'):
                outputs = self.activation(linear, is_training)

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

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell = rnn_cell.BasicLSTMCell(self.num_units)

            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(2, outputs_tupple)

            return outputs

class PBLSTMLayer(object):
    ''' a pyramidal bidirectional LSTM layer'''

    def __init__(self, num_units):
        """
        BlstmLayer constructor
        Args:
            num_units: The number of units in the LSTM
            pyramidal: indicates if a pyramidal BLSTM is desired.
        """

        #create BLSTM layer
        self.blstm = BLSTMLayer(num_units)

    def __call__(self, inputs, sequence_lengths, scope=None):
        """
        Create the variables and do the forward computation
        Args:
            inputs: A time minor tensor of shape [batch_size, time,
                input_size],
            sequence_lengths: the length of the input sequences
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.
        Returns:
            the output of the layer, the concatenated outputs of the
            forward and backward pass shape [batch_size, time/2, input_size*2].
        """


        with tf.variable_scope(scope or type(self).__name__):

            #apply blstm layer
            outputs = self.blstm(inputs, sequence_lengths)
            stacked_outputs, output_seq_lengths = ops.pyramid_stack(
                outputs,
                sequence_lengths)


        return stacked_outputs, output_seq_lengths

class GatedAConv1d(object):
    '''A gated atrous convolution block'''
    def __init__(self, kernel_size):
        '''constructor

        Args:
            kernel_size: size of the filters
        '''

        self.kernel_size = kernel_size

    def __call__(self, inputs, seq_length, causal=False, dilation_rate=1,
                 is_training=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            seq_length: the length of the input sequences
            causal: flag for causality, if true every output will only be
                affected by previous inputs
            dilation_rate: the rate of dilation
            is_training: whether or not the network is in training mode
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            a pair containing:
                - The residual output
                - the skip connections
        '''

        with tf.variable_scope(scope or type(self).__name__):

            num_units = int(inputs.get_shape()[2])

            #the dilated convolution layer
            dconv = AConv1dLayer(num_units, self.kernel_size,
                                 dilation_rate)

            #the one by one convolution
            onebyone = Conv1dLayer(num_units, 1, 1)

            #compute the data
            data = dconv(inputs, seq_length, causal, is_training, 'data_dconv')
            data = tf.nn.tanh(data)

            #compute the gate
            gate = dconv(inputs, seq_length, causal, is_training, 'gate_dconv')
            gate = tf.nn.sigmoid(gate)

            #compute the gated output
            gated = data*gate

            #compute the final output
            out = onebyone(gated, seq_length, is_training, '1x1_res')
            out = tf.nn.tanh(out)

            #compute the skip
            skip = onebyone(gated, seq_length, is_training, '1x1_skip')

            #return the residual and the skip
            return inputs + out, skip

class Conv1dLayer(object):
    '''a 1-D convolutional layer'''

    def __init__(self, num_units, kernel_size, stride):
        '''constructor

        Args:
            num_units: the number of filters
            kernel_size: the size of the filters
            stride: the stride of the convolution
        '''

        self.num_units = num_units
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, inputs, seq_length, is_training=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            seq_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the outputs which is a [batch_size, max_length/stride, num_units]
        '''

        with tf.variable_scope(scope or type(self).__name__):

            input_dim = int(inputs.get_shape()[2])
            stddev = 1/input_dim**0.5

            #the filte parameters
            w = tf.get_variable(
                'filter', [self.kernel_size, input_dim, self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #the bias parameters
            b = tf.get_variable(
                'bias', [self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #do the convolution
            out = tf.nn.conv1d(inputs, w, self.stride, padding='SAME')

            #add the bias
            out = ops.seq2nonseq(out, seq_length)
            out += b
            out = ops.nonseq2seq(out, seq_length, int(inputs.get_shape()[1]))

        return out

class AConv1dLayer(object):
    '''a 1-D atrous convolutional layer'''

    def __init__(self, num_units, kernel_size, dilation_rate):
        '''constructor

        Args:
            num_units: the number of filters
            kernel_size: the size of the filters
            dilation_rate: the rate of dilation
        '''

        self.num_units = num_units
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def __call__(self, inputs, seq_length, causal=False,
                 is_training=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            seq_length: the length of the input sequences
            causal: flag for causality, if true every output will only be
                affected by previous inputs
            is_training: whether or not the network is in training mode
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the outputs which is a [batch_size, max_length/stride, num_units]
        '''

        with tf.variable_scope(scope or type(self).__name__):

            input_dim = int(inputs.get_shape()[2])
            stddev = 1/input_dim**0.5

            #the filter parameters
            w = tf.get_variable(
                'filter', [self.kernel_size, input_dim, self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #the bias parameters
            b = tf.get_variable(
                'bias', [self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #do the arous convolution
            if causal:
                out = ops.causal_aconv1d(inputs, w, self.dilation_rate)
            else:
                out = ops.aconv1d(inputs, w, self.dilation_rate)


            #add the bias
            out = ops.seq2nonseq(out, seq_length)
            out += b
            out = ops.nonseq2seq(out, seq_length, int(inputs.get_shape()[1]))

        return out
