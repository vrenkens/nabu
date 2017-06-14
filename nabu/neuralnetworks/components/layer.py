'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from nabu.neuralnetworks.components import ops

class BLSTMLayer(object):
    '''a BLSTM layer'''

    def __init__(self, num_units, layer_norm=False):
        '''
        BLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                layer_norm=self.layer_norm,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                layer_norm=self.layer_norm,
                reuse=tf.get_variable_scope().reuse)

            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs

class PBLSTMLayer(object):
    ''' a pyramidal bidirectional LSTM layer'''

    def __init__(self, num_units, num_steps=2, layer_norm=False):
        """
        PBLSTMLayer constructor

        Args:
            num_units: The number of units in the LSTM
            num_steps: the number of time steps to concatenate
            layer_norm: whether layer normalization should be applied
        """

        #create BLSTM layer
        self.blstm = BLSTMLayer(num_units, layer_norm)
        self.num_steps = num_steps

    def __call__(self, inputs, sequence_lengths, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer and the sequence lengths of the outputs
        '''


        with tf.variable_scope(scope or type(self).__name__):

            #apply blstm layer
            outputs = self.blstm(inputs, sequence_lengths)
            stacked_outputs, output_seq_lengths = ops.pyramid_stack(
                outputs,
                sequence_lengths,
                self.num_steps)


        return stacked_outputs, output_seq_lengths

def projected_subsampling(inputs, input_seq_lengths, num_steps, name=None):
    '''
    apply projected subsampling, this is concatenating 2 timesteps,
    projecting to a lower dimensionality, applying batch_normalization
    and a relu layer

    args:
        inputs: a [batch_size x max_length x dim] input tensorflow
        input_seq_lengths: the input sequence lengths as a [batch_size] vector
        num_steps: the number of steps to concatenate
        is_training: bool training mode
        name: the name of the operation

    returns:
        - a [batch_size x ceil(max_length/2) x dim] output tensor
        - the output sequence lengths as a [batch_size] vector
    '''

    with tf.variable_scope(name or 'subsampling'):
        input_dim = int(inputs.get_shape()[2])

        #concatenate 2 timesteps
        stacked_inputs, output_seq_lengths = ops.pyramid_stack(
            inputs,
            input_seq_lengths,
            num_steps)

        #project back to the input dimension
        outputs = tf.contrib.layers.linear(stacked_inputs, input_dim)

        return outputs, output_seq_lengths
