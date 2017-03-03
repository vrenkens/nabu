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
        padded = tf.concat([tf.zeros(padding_shape), inputs], 1)

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

def pyramid_stack(inputs, sequence_lengths, scope=None):
    '''
    concatenate each two consecutive elements

    Args:
        inputs: A time minor tensor [batch_size, time, input_size]
        sequence_lengths: the length of the input sequences
        scope: the current scope

    Returns:
        inputs: Concatenated inputs [batch_size, time/2, input_size*2]
        sequence_lengths: the lengths of the inputs sequences [batch_size]
    '''

    with tf.name_scope(scope or 'pyramid_stack'):

        input_shape = tf.Tensor.get_shape(inputs)

        #pad with zeros if odd number of inputs
        if int(input_shape[1]) % 2 == 1:
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 0]])
            length = int(input_shape[1]) + 1
        else:
            padded_inputs = inputs
            length = int(input_shape[1])

        #convert imputs to time major
        time_major_input = tf.transpose(padded_inputs, [1, 0, 2])

        #seperate odd and even inputs
        odd_inputs = tf.gather(time_major_input, range(1, length, 2))
        even_inputs = tf.gather(time_major_input, range(0, length, 2))

        #concatenate odd and even inputs
        time_major_outputs = tf.concat([even_inputs, odd_inputs], 2)

        #convert back to time minor
        outputs = tf.transpose(time_major_outputs, [1, 0, 2])

        #compute the new sequence length
        output_sequence_lengths = tf.cast(tf.ceil(tf.cast(sequence_lengths,
                                                          tf.float32)/2),
                                          tf.int32)

    return outputs, output_sequence_lengths

def seq2nonseq(sequential, seq_length, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        sequential: the sequential data which is a [batch_size, max_length, dim]
            tensor
        seq_length: a vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq'):
        #convert the list for each time step to a list for each sequence
        sequences = tf.unstack(sequential)

        #remove the padding from sequences
        sequences = [tf.gather(sequences[s], tf.range(seq_length[s]))
                     for s in range(len(sequences))]

        #concatenate the sequences
        tensor = tf.concat(sequences, 0)

    return tensor

def nonseq2seq(tensor, seq_length, length, name=None):
    '''
    Convert non sequential data to sequential data

    Args:
        tensor: non sequential data, which is a TxF tensor where T is the sum of
            all sequence lengths
        seq_length: a vector containing the sequence lengths
        length: the constant length of the output sequences
        name: [optional] the name of the operation

    Returns:
        sequential data, which is a [batch_size, max_length, dim]
        tensor
    '''

    with tf.name_scope(name or'nonseq2seq'):
        #get the cumulated sequence lengths to specify the positions in tensor
        cum_seq_length = tf.concat([tf.constant([0]), tf.cumsum(seq_length)], 0)

        #get the indices in the tensor for each sequence
        indices = [tf.range(cum_seq_length[l], cum_seq_length[l+1])
                   for l in range(int(seq_length.get_shape()[0]))]

        #create the non-padded sequences
        sequences = [tf.gather(tensor, i) for i in indices]

        #pad the sequences with zeros
        sequences = [tf.pad(sequences[s], [[0, length-seq_length[s]], [0, 0]])
                     for s in range(len(sequences))]

        #specify that the sequences have been padded to the constant length
        for seq in sequences:
            seq.set_shape([length, int(tensor.get_shape()[1])])

        #stack the sequences into a tensor
        sequential = tf.stack(sequences)

    return sequential
