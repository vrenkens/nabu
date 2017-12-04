'''@file attention.py
contain attention mechanisms'''

import tensorflow as tf

class BahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    '''attention mechanism that is location aware'''

    def __call__(self, query, previous_alignments):
        '''Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            previous_alignments: Tensor of dtype matching `self.values` and
                shape `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''

        with tf.variable_scope(None, 'bahdanau_attention',
                               [query, previous_alignments]):

            #query = tf.Print(query, [query])

            processed_query = \
                self.query_layer(query) if self.query_layer else query

            score = _bahdanau_score(processed_query, self._keys,
                                    self._normalize)

            alignments = self._probability_fn(score, previous_alignments)

            return alignments

class LocationAwareAttention(tf.contrib.seq2seq.BahdanauAttention):
    '''attention mechanism that is location aware'''

    def __init__(self,
                 num_units,
                 numfilt,
                 filtersize,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name='LocationAwareAttention'):
        '''Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            numfilt: the number of filters used for the convolutinonal features
            filtersize; te size
            memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options
                include @{tf.contrib.seq2seq.hardmax} and
                @{tf.contrib.sparsemax.sparsemax}. Its signature should be:
                `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before
                passing into `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        '''

        super(LocationAwareAttention, self).__init__(
            num_units,
            memory,
            memory_sequence_length,
            normalize,
            probability_fn,
            score_mask_value,
            name
        )

        self._numfilt = numfilt
        self._filtersize = filtersize

    def __call__(self, query, previous_alignments):
        '''Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            previous_alignments: Tensor of dtype matching `self.values` and
                shape `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''

        with tf.variable_scope(None, 'location_aware_attention',
                               [query, previous_alignments]):
            processed_query = \
                self.query_layer(query) if self.query_layer else query

            conv_features = tf.layers.conv1d(
                inputs=tf.expand_dims(previous_alignments, 2),
                filters=self._numfilt,
                kernel_size=self._filtersize,
                padding='same',
                use_bias=False
            )

            processed_conv_features = tf.layers.dense(
                inputs=conv_features,
                units=self._num_units,
                use_bias=False,
                name='process_conv_features'
            )

            score = _bahdanau_location_score(
                processed_query, self._keys,
                processed_conv_features, self._normalize)

            alignments = self._probability_fn(score, previous_alignments)

            return alignments

class MonotonicAttention(tf.contrib.seq2seq.BahdanauAttention):
    '''attention mechanism that is location aware'''

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name='MonotonicAttention'):
        '''Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options
                include @{tf.contrib.seq2seq.hardmax} and
                @{tf.contrib.sparsemax.sparsemax}. Its signature should be:
                `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before
                passing into `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        '''

        super(MonotonicAttention, self).__init__(
            num_units,
            memory,
            memory_sequence_length,
            normalize,
            probability_fn,
            score_mask_value,
            name
        )
        _curent_prob_fn = self._probability_fn
        self._probability_fn = lambda s, p: _monotonit_probability_wrapper(
            _curent_prob_fn(s, p), p)

def _bahdanau_location_score(processed_query, keys, processed_convolutional_features,
                             normalize):
    '''
    Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    'Neural Machine Translation by Jointly Learning to Align and Translate.'
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    'Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks.'
    https://arxiv.org/abs/1602.07868
    To enable the second form, set `normalize=True`.

    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to
            keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        processed_convolutional_features: the processed convolutional features
            as shape `[batch_size, max_time, num_units]`
        normalize: Whether to normalize the score function.

    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    '''

    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)

    v = tf.get_variable('attention_v', [num_units], dtype=dtype)

    summed = processed_query + keys + processed_convolutional_features

    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable(
            'attention_g', dtype=dtype,
            initializer=tf.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = tf.get_variable(
            'attention_b', [num_units], dtype=dtype,
            initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))

        return tf.reduce_sum(normed_v*tf.tanh(summed + b), [2])
    else:
        return tf.reduce_sum(v*tf.tanh(summed), [2])

def _bahdanau_score(processed_query, keys, normalize):
    '''
    Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    'Neural Machine Translation by Jointly Learning to Align and Translate.'
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    'Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks.'
    https://arxiv.org/abs/1602.07868
    To enable the second form, set `normalize=True`.

    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to
            keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        normalize: Whether to normalize the score function.

    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    '''

    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)

    v = tf.get_variable('attention_v', [num_units], dtype=dtype)

    summed = processed_query + keys

    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable(
            'attention_g', dtype=dtype,
            initializer=tf.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = tf.get_variable(
            'attention_b', [num_units], dtype=dtype,
            initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))

        return tf.reduce_sum(normed_v*tf.tanh(summed + b), [2])
    else:
        return tf.reduce_sum(v*tf.tanh(summed), [2])

def _monotonit_probability_wrapper(alignments, previous_alignments):
    '''
    this function will make sure that the alignments are monotonic

    args:
        alignments: the current alignments shape [batch_size x max_time]
        previous_alignments: the previous alignments shape
            [batch_size x max_time]

    returns:
        the alignments adjusted for monotonicity
    '''

    alignments_cumsum = tf.cumsum(alignments, axis=1)
    previous_alignments_cumsum = tf.cumsum(previous_alignments, axis=1)
    target_cumsum = tf.minimum(alignments_cumsum, previous_alignments_cumsum)
    shifted_target_cumsum = tf.pad(target_cumsum[:, :-1], [[0, 0], [1, 0]])
    monotonic_alignments = target_cumsum - shifted_target_cumsum

    #the initial alignment is zero everywhere. If it is the alignment, keep
    #the input alignment
    monotonic_alignments = tf.where(
        tf.equal(tf.reduce_sum(previous_alignments, 1), 0),
        alignments,
        monotonic_alignments)

    return monotonic_alignments
