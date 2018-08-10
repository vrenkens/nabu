'''@file attention.py
contain attention mechanisms'''

import tensorflow as tf

def factory(conf, num_units, encoded, encoded_seq_length):
    '''create the attention mechanism'''

    prob_fn = {
        'softmax': None,
        'normalized_sigmoid': normalized_sigmoid,
        'sigmoid':tf.sigmoid
    }

    if conf['attention'] == 'location_aware':
        return LocationAwareAttention(
            num_units=num_units,
            numfilt=int(conf['numfilt']),
            filtersize=int(conf['filtersize']),
            memory=encoded,
            memory_sequence_length=encoded_seq_length,
            probability_fn=prob_fn[conf['probability_fn']]
        )
    elif conf['attention'] == 'vanilla':
        return tf.contrib.seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=encoded,
            memory_sequence_length=encoded_seq_length,
            probability_fn=prob_fn[conf['probability_fn']]
        )
    elif conf['attention'] == 'windowed':
        return WindowedAttention(
            num_units=num_units,
            left_window_width=int(conf['left_window_width']),
            right_window_width=int(conf['right_window_width']),
            memory=encoded,
            memory_sequence_length=encoded_seq_length,
            probability_fn=prob_fn[conf['probability_fn']]
        )

def normalized_sigmoid(x, axis=-1):
    '''
    the normalized sigmoid probability function

    args:
        x: the input tensor
        axis: the axis to normalize

    returns:
        the normalize sigmoid output
    '''

    sig = tf.sigmoid(x)

    return sig/tf.reduce_sum(sig, axis, keep_dims=True)


class BahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    '''normal Bahdanau Style attention'''

    def __call__(self, query, state):
        '''Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and
                shape `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''

        with tf.variable_scope(None, 'bahdanau_attention',
                               [query, state]):

            processed_query = \
                self.query_layer(query) if self.query_layer else query

            score = _bahdanau_score(processed_query, self._keys,
                                    self._normalize)

            alignments = self._probability_fn(score, state)

            return alignments, alignments

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
                 dtype=None,
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
            dtype,
            name
        )

        self._numfilt = numfilt
        self._filtersize = filtersize

    def __call__(self, query, state):
        '''Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and
                shape `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''

        with tf.variable_scope(None, 'location_aware_attention',
                               [query, state]):
            processed_query = \
                self.query_layer(query) if self.query_layer else query

            conv_features = tf.layers.conv1d(
                inputs=tf.expand_dims(state, 2),
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

            alignments = self._probability_fn(score, state)

            return alignments, alignments

def _bahdanau_location_score(processed_query, keys,
                             processed_convolutional_features,
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

class WindowedAttention(tf.contrib.seq2seq.BahdanauAttention):
    '''attention mechanism that is location aware'''

    def __init__(self,
                 num_units,
                 left_window_width,
                 right_window_width,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 dtype=None,
                 name='LocationAwareAttention'):
        '''Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            window_width: the width of the attention window
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

        super(WindowedAttention, self).__init__(
            num_units,
            memory,
            memory_sequence_length,
            normalize,
            probability_fn,
            score_mask_value,
            dtype,
            name
        )

        self._left_window_width = left_window_width
        self._right_window_width = right_window_width

    def initial_alignments(self, batch_size, dtype):
        '''get the initial alignments'''

        max_time = self._alignments_size
        return tf.concat([
            tf.ones([batch_size, 1], dtype),
            tf.zeros([batch_size, max_time-1])], 1)

    def __call__(self, query, state):
        '''Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and
                shape `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''

        with tf.variable_scope(None, 'windowed_attention',
                               [query, state]):
            #process the query
            processed_query = \
                self.query_layer(query) if self.query_layer else query

            #determine the attention window
            cum_alignment = tf.cumsum(state, 1)
            half_step = cum_alignment > 0.5
            shifted_left = tf.pad(
                half_step[:, self._left_window_width+1:],
                [[0, 0], [0, self._left_window_width+1]],
                constant_values=True)
            shifted_right = tf.pad(
                half_step[:, :-self._right_window_width],
                [[0, 0], [self._right_window_width, 0]],
                constant_values=False)
            window = tf.logical_xor(shifted_left, shifted_right)

            score = _bahdanau_score(
                processed_query, self._keys, self._normalize)

            #mask the score using the window
            score = tf.where(window, score, -tf.ones_like(score)*float('inf'))

            alignments = self._probability_fn(score, state)

            return alignments, alignments
