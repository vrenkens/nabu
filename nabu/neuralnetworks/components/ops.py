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

def stack_seq(sequential, sequence_lengths, name=None):
    '''
    remove padding and stack sequences

    Args:
        sequential: the sequential data which is a [batch_size, max_length, dim]
            tensor
        sequence_lengths: a [batch_size] vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'stack_seq'):

        indices = get_indices(sequence_lengths)

        #create the values
        tensor = tf.gather_nd(sequential, indices)
        tensor.set_shape([None] + sequential.shape.as_list()[2:])


    return tensor

def unstack_seq(nonseq, sequence_lengths, name=None):
    '''
    unstack sequences and add padding

    Args:
        nonseq: the non sequential data which is a
            [sum(sequence_lengths) x dim] tensor
        sequence_lengths: a [batch_size] vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        sequential data, which is a  [batch_size, max_length, dim] tensor
    '''

    with tf.name_scope(name or 'unstack_seq'):
        max_length = tf.reduce_max(sequence_lengths)
        batch_size = tf.size(sequence_lengths)
        unstacked = tf.TensorArray(
            dtype=nonseq.dtype,
            size=batch_size,
            element_shape=tf.TensorShape([None]).concatenate(nonseq.shape[1:]),
            infer_shape=False)
        unstacked = unstacked.split(nonseq, sequence_lengths)
        unstacked = map_ta(
            lambda x: pad_to(x, max_length),
            unstacked
        )
        unstacked = unstacked.stack()
        unstacked.set_shape([None, None] + nonseq.shape.as_list()[1:])

    return unstacked


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

def get_indices(sequence_length):
    '''get the indices corresponding to sequences (and not padding)

    Args:
        sequence_length: the sequence_lengths as a N-D tensor

    Returns:
        A [sum(sequence_length) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(sequence_length.shape)

        #get the maximal length
        max_length = tf.reduce_max(sequence_length)

        sizes = tf.shape(sequence_length)

        range_tensor = tf.range(max_length)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(sequence_length, numdims)))

    return indices

def pad_to(tensor, length, axis=0, name=None):
    '''pad the tensor to a certain length

    args:
        - tensor: the tensor to pad
        - length: the length to pad to, has to be larger than tensor.shape[axis]
        - axis: the axis to pad
        - name: the name of the operation

    returns:
        the padded tensor
    '''

    with tf.name_scope(name or 'pad_to'):
        rank = tensor.shape.ndims
        orig_length = tf.shape(tensor)[axis]
        assert_op = tf.assert_less(axis, rank,
                                   message='axis has to be less than rank')
        with tf.control_dependencies([assert_op]):
            assert_op = tf.assert_less_equal(
                orig_length, length,
                message='target length less than original length')
        with tf.control_dependencies([assert_op]):
            paddings = tf.SparseTensor(
                indices=[[axis, 1]],
                values=tf.expand_dims(length-orig_length, 0),
                dense_shape=[rank, 2])

        padded = tf.pad(tensor, tf.sparse_tensor_to_dense(paddings))

    return padded

def map_ta(fn, ta):
    '''
    apply fn to each element in tensorarray

    args:
        fn: the function to apply
        ta: the tensorarray

    returns:
        the resulting tensorarray
    '''

    def body(index, ta_out):
        '''the body of the while loop'''
        ta_out = ta_out.write(index, fn(ta.read(index)))
        newindex = index + 1

        return newindex, ta_out

    def condition(index, ta_out):
        '''loop condition'''
        return tf.not_equal(index, ta_out.size())

    ta_init = tf.TensorArray(
        dtype=ta.dtype,
        size=ta.size()
    )
    index_init = 0

    _, mapped = tf.while_loop(
        cond=condition,
        body=body,
        loop_vars=[index_init, ta_init]
    )

    return mapped
