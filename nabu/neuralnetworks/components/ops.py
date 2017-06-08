'''@file ops.py
some operations'''

import tensorflow as tf

def pyramid_stack(inputs, sequence_lengths, numsteps, axis=2, scope=None):
    '''
    concatenate each two consecutive elements

    Args:
        inputs: A time minor tensor [batch_size, time, input_size]
        sequence_lengths: the length of the input sequences
        numsteps: number of time steps to concatenate
        axis: the axis where the inputs should be stacked
        scope: the current scope

    Returns:
        inputs: Concatenated inputs
            [batch_size, time/numsteps, input_size*numsteps]
        sequence_lengths: the lengths of the inputs sequences [batch_size]
    '''

    with tf.name_scope(scope or 'pyramid_stack'):

        numdims = len(inputs.shape)

        #convert imputs to time major
        time_major_input = tf.transpose(inputs, [1, 0] + range(2, numdims))


        #pad the inputs to an appropriate length length
        length = tf.cast(tf.shape(time_major_input)[0], tf.float32)
        pad_length = tf.ceil(length/numsteps)*numsteps - length
        pad_length = tf.cast(pad_length, tf.int32)
        pad_shape = tf.concat([[pad_length],
                               tf.shape(time_major_input)[1:]], 0)
        padding = tf.zeros(pad_shape, dtype=inputs.dtype)
        padded_inputs = tf.concat([time_major_input, padding], 0)

        #get the new length
        length = tf.shape(padded_inputs)[0]

        #seperate the inputs for every concatenated timestep
        seperated = []
        for i in range(numsteps):
            seperated.append(tf.gather(
                padded_inputs, tf.range(i, length, numsteps)))

        #concatenate odd and even inputs
        time_major_outputs = tf.concat(seperated, axis)

        #convert back to time minor
        outputs = tf.transpose(time_major_outputs, [1, 0] + range(2, numdims))

        #compute the new sequence length
        output_sequence_lengths = tf.cast(tf.ceil(tf.cast(sequence_lengths,
                                                          tf.float32)/numsteps),
                                          tf.int32)

    return outputs, output_sequence_lengths

def seq2nonseq(sequential, sequence_lengths, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        sequential: the sequential data which is a [batch_size, max_length, dim]
            tensor
        sequence_lengths: a [batch_size] vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq'):

        indices = get_indices(sequence_lengths)

        #create the values
        tensor = tf.gather_nd(sequential, indices)


    return tensor

def dense_sequence_to_sparse(sequences, sequence_lengths):
    '''convert sequence dense representations to sparse representations

    Args:
        sequences: the dense sequences as a [batch_size x max_length] tensor
        sequence_lengths: the sequence lengths as a [batch_size] vector

    Returns:
        the sparse tensor representation of the sequences
    '''

    with tf.name_scope('dense_sequence_to_sparse'):

        #get all the non padding sequences
        indices = tf.cast(get_indices(sequence_lengths), tf.int64)

        #create the values
        values = tf.gather_nd(sequences, indices)

        #the shape
        shape = tf.cast(tf.shape(sequences), tf.int64)

        sparse = tf.SparseTensor(indices, values, shape)

    return sparse

def cross_entropy_loss_eos(targets, logits, logit_seq_length,
                           target_seq_length):
    '''
    Compute the cross_entropy loss with an added end of sequence label

    Args:
        targets: a [batch_size x time] tensor containing the targets
        logits: a [batch_size x time x num_classes] tensor containing the logits
        logit_seq_length: a [batch_size] vector containing the
            logit sequence lengths
        target_seq_length: a [batch_size] vector containing the
            target sequence lengths

    Returns:
        a scalar value containing the loss
    '''

    batch_size = tf.shape(targets)[0]

    with tf.name_scope('cross_entropy_loss'):

        output_dim = tf.shape(logits)[2]

        #get the logits for the final timestep
        indices = tf.stack([tf.range(batch_size),
                            logit_seq_length-1],
                           axis=1)
        final_logits = tf.gather_nd(logits, indices)

        #stack all the logits except the final logits
        stacked_logits = seq2nonseq(logits,
                                    logit_seq_length - 1)

        #create the stacked targets
        stacked_targets = seq2nonseq(targets,
                                     target_seq_length)

        #create the targets for the end of sequence labels
        final_targets = tf.tile([output_dim-1], [batch_size])

        #add the final logits and targets
        stacked_logits = tf.concat([stacked_logits, final_logits], 0)
        stacked_targets = tf.concat([stacked_targets, final_targets], 0)

        #compute the cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_logits,
            labels=stacked_targets)

        loss = tf.reduce_mean(losses)

    return loss

def get_indices(sequence_length):
    '''get the indices corresponding to sequences (and not padding)

    Args:
        sequence_length: the sequence_lengths as a N-D tensor

    Returns:
        A [sum(sequence_length) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(sequence_length.shape)

        #get th emaximal length
        max_length = tf.reduce_max(sequence_length)

        sizes = tf.shape(sequence_length)

        range_tensor = tf.range(max_length-1)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(sequence_length, numdims)))

    return indices
