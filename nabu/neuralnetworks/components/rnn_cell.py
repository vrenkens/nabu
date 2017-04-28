'''@file rnn_cell.py
contains some customized rnn cells'''

import tensorflow as tf
from tensorflow.python.util import nest

class ScopeRNNCellWrapper(tf.contrib.rnn.RNNCell):
    '''this wraps an RNN cell to make sure it uses the same scope every time
    it's called'''

    def __init__(self, cell, name):
        '''ScopeRNNCellWrapper constructor'''

        self._cell = cell
        self.scope = tf.VariableScope(None, name)

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.output_size

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        with tf.variable_scope(self.scope):
            output, new_state = self._cell(inputs, state, scope)

        return output, new_state

class StateOutputWrapper(tf.contrib.rnn.RNNCell):
    '''this wraps an RNN cell to make it output its concatenated state instead
        of the output'''

    def __init__(self, cell):
        '''StateOutputWrapper constructor'''

        self._cell = cell

    @property
    def output_size(self):
        '''return cell output size'''

        return sum([int(x) for x in nest.flatten(self._cell.state_size)])

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        _, new_state = self._cell(inputs, state, scope)
        output = tf.concat(nest.flatten(new_state), axis=1)

        return output, new_state

class DynamicAttentionWrapper(tf.contrib.rnn.RNNCell):
    '''a slightly modified verion of tensorflows DynamicAttentionWrapper

    instead of the rnn_cell output the rnn_cell state is used to querry the
    attention mechanism. Also no linear layer is applied to the context, but
    instead the context is diectly used as output. A linear layer IS applied
    to the concatenated input and attention before using it as the input to the
    cell.
    '''

    def __init__(self,
                 cell,
                 attention_mechanism,
                 probability_fn=None):

        self._probability_fn = probability_fn
        if self._probability_fn is None:
            self._probability_fn = tf.nn.softmax


        self._attention_mechanism = attention_mechanism
        self._attention_size = \
            int(self._attention_mechanism.values.get_shape()[2])
        self._cell = cell

    @property
    def state_size(self):
        '''returns the state size'''
        return (self._cell.state_size,
                self._attention_size)

    @property
    def output_size(self):
        '''the cell output size'''

        return self._attention_size + self._cell.output_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return tf.contrib.seq2seq.DynamicAttentionWrapperState(
            cell_state=self._cell.zero_state(batch_size, dtype),
            attention=tf.zeros([batch_size, self._attention_size]))

    def __call__(self, inputs, state, scope=None):
        '''Perform a step of attention-wrapped RNN.'''

        attention = state.attention
        cell_state = state.cell_state
        cell_inputs = tf.contrib.layers.linear(
            tf.concat([inputs, attention], 1), int(inputs.get_shape()[1]))

        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        if isinstance(next_cell_state, tuple):
            query = tf.concat(nest.flatten(next_cell_state), 1)
        else:
            query = next_cell_state

        score = self._attention_mechanism(query)
        alignments = self._probability_fn(score)
        alignments = tf.expand_dims(alignments, 1)
        context = tf.matmul(alignments, self._attention_mechanism.values)
        context = tf.squeeze(context, [1])

        next_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
            cell_state=next_cell_state,
            attention=context)

        output = tf.concat([cell_output, context], 1)

        return output, next_state
