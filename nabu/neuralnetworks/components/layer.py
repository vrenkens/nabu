'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from nabu.neuralnetworks.components import ops

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
            lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)

            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs

class PBLSTMLayer(object):
    ''' a pyramidal bidirectional LSTM layer'''

    def __init__(self, num_units, num_steps):
        """
        BlstmLayer constructor
        Args:
            num_units: The number of units in the LSTM
            num_steps: the number of time steps to concatenate
        """

        #create BLSTM layer
        self.blstm = BLSTMLayer(num_units)
        self.num_steps = num_steps

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
                sequence_lengths,
                self.num_steps)


        return stacked_outputs, output_seq_lengths

class BLSTMNormLayer(object):
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
            lstm_cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)

            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs

class PBLSTMNormLayer(object):
    ''' a pyramidal bidirectional LSTM layer'''

    def __init__(self, num_units, num_steps):
        """
        BlstmLayer constructor
        Args:
            num_units: The number of units in the LSTM
            num_steps: the number of time steps to concatenate
        """

        #create BLSTM layer
        self.blstm = BLSTMNormLayer(num_units)
        self.num_steps = num_steps

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
                sequence_lengths,
                self.num_steps)


        return stacked_outputs, output_seq_lengths
