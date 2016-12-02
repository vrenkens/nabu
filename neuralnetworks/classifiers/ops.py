'''@file ops.py
some operations'''

import tensorflow as tf

def aconv1d(inputs, filt, dilation_rate, scope=None):
    '''a 1 dimensional causal (diluted) convolution

    Args:
        inputs: a [batch_size, max_seq_length, dim] input tensorflow
        filt: the filter of shape [kernel_size, dim_in, dim_out]
        dilation rate: the rate of dilation (integer)
        scope: the name of the operations

    Returns:
        the output of the 1D atrous convolution
    '''

    with tf.name_scope(scope or 'aconv1d'):
        #add the dimension (height = 1) to make 2d convolution possible
        exp_inputs = tf.expand_dims(inputs, 1)
        exp_filter = tf.expand_dims(filt, 0)

        #do the convolution
        out = tf.nn.atrous_conv2d(exp_inputs, exp_filter, dilation_rate,
                                  padding='SAME')

        #remove the added dimension and extra outputs at the end
        out = out[:, 0, :, :]

    return out

def causal_aconv1d(inputs, filt, dilation_rate, scope=None):
    '''a 1 dimensional causal atrous (diluted) convolution

    Args:
        inputs: a [batch_size, max_seq_length, dim] input tensorflow
        filt: the filter of shape [kernel_size, dim_in, dim_out]
        dilation rate: the rate of dilation (integer)
        scope: the name of the operations

    Returns:
        the output of the 1D causal atrous convolution
    '''

    with tf.name_scope(scope or 'causal_aconv1d'):
        filter_size = int(filt.get_shape()[0])
        inputs_shape = inputs.get_shape().as_list()

        #pad zeros to the input to make the convolution causal
        padding_shape = inputs_shape
        padding_shape[1] = dilation_rate*(filter_size-1)
        padded = tf.concat(1, [tf.zeros(padding_shape), inputs])

        #do the convolution
        out = aconv1d(padded, filt, dilation_rate)

        #remove the extra outputs at the end
        out = out[:, :inputs_shape[1], :]

    return out

def mu_law_encode(inputs, num_levels, scope=None):
    '''do mu-law encoding

    Args:
        inputs: the inputs to quantize
        num_levels: number of quantization lavels

    Returns:
        te one-hot encoded inputs'''

    with tf.name_scope(scope or 'mu_law'):
        mu = num_levels - 1
        transformed = tf.sign(inputs)*tf.log(1+mu*tf.abs(inputs))/tf.log(1+mu)
        quantized = tf.cast((transformed+1)*num_levels/2+0.5, tf.int32)
        encoded = tf.one_hot(quantized, num_levels)

    return encoded
