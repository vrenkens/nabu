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

class BypassWrapper(tf.contrib.rnn.RNNCell):
    '''this will concatenate the input with the output'''

    def __init__(self, cell):
        '''ScopeRNNCellWrapper constructor'''

        self._cell = cell

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.output_size + self._cell.input_shape[-1]

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        output, new_state = self._cell(inputs, state, scope)
        output = tf.concat([output, inputs], -1)

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

class AttentionProjectionWrapper(tf.contrib.rnn.RNNCell):
    '''this wraps an RNN cell to take the context and state and maps it to
        a fixed diension'''

    def __init__(self, cell, output_dim, activation=None):
        '''AttentionProjectionWrapper constructor takes the context and the
        state and maps to output_dim units

        Args:
            cell: the wrapped cell
            output_dim: the number of output units
            activation: a nonlinearity to be applied after the mapping
                (default: linear)
        '''

        self._cell = cell
        self._output_dim = output_dim
        self._activation = activation

    @property
    def output_size(self):
        '''return cell output size'''

        return self._output_dim

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        cell_output, new_state = self._cell(inputs, state, scope)
        context = new_state.attention
        output = tf.layers.dense(
            tf.concat([cell_output, context], axis=1),
            self._output_dim,
            self._activation)

        return output, new_state
