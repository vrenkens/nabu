'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from nabu.neuralnetworks.components import ops

def blstm(
    inputs,
    sequence_length,
    num_units,
    layer_norm=False,
    scope=None):
    '''
    a BLSTM layer

    args:
        inputs: the input to the layer as a
            [batch_size, max_length, dim] tensor
        sequence_length: the length of the input sequences as a
            [batch_size] tensor
        num_units: The number of units in the one directon
        layer_norm: whether layer normalization should be applied
        scope: The variable scope sets the namespace under which
            the variables created during this call will be stored.

    returns:
        the blstm outputs
    '''

    with tf.variable_scope(scope or 'BLSTM'):

        #create the lstm cell that will be used for the forward and backward
        #pass
        lstm_cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units=num_units,
            layer_norm=layer_norm,
            reuse=tf.get_variable_scope().reuse)
        lstm_cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            layer_norm=layer_norm,
            reuse=tf.get_variable_scope().reuse)

        #do the forward computation
        outputs_tupple, _ = bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
            sequence_length=sequence_length)

        outputs = tf.concat(outputs_tupple, 2)

        return outputs

def pblstm(
    inputs,
    sequence_length,
    num_units,
    num_steps=2,
    layer_norm=False,
    scope=None):
    '''
    a Pyramidal BLSTM layer

    args:
        inputs: the input to the layer as a
            [batch_size, max_length, dim] tensor
        sequence_length: the length of the input sequences as a
            [batch_size] tensor
        num_units: The number of units in the one directon
        num_steps: the number of time steps to concatenate
        layer_norm: whether layer normalization should be applied
        scope: The variable scope sets the namespace under which
            the variables created during this call will be stored.

    returns:
        - the PBLSTM outputs
        - the new sequence lengths
    '''

    with tf.variable_scope(scope or 'PBLSTM'):
        #apply blstm layer
        outputs = blstm(
            inputs=inputs,
            sequence_length=sequence_length,
            num_units=num_units,
            layer_norm=layer_norm
        )

        #stack the outputs
        outputs, output_seq_lengths = ops.pyramid_stack(
            outputs,
            sequence_length,
            num_steps)

        return outputs, output_seq_lengths

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
