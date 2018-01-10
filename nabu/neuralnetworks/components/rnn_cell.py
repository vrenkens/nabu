'''@file rnn_cell.py
contains some customized rnn cells'''

import collections
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

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

class DynamicRoutingAttentionWrapperState(
        collections.namedtuple('DynamicRoutingAttentionWrapperState',
                               ('cell_state',
                                'alignments',
                                'capsule_activities',
                                'reconstruction',
                                'context'))):
    '''
    Tupple for DynamicRoutingAttentionWrapperState

    args:
        cell_state: the state of the wrapped cell
        alignments: the current alignments
        context: the sum of the activities of the capsules
    '''

    pass

class DynamicRoutingAttentionWrapper(tf.contrib.rnn.RNNCell):
    '''A dynamic routing rnn cell'''

    def __init__(self, cell, num_capsules, capsule_dim, numiters,
                 attention_mechanism, input_context, input_activation,
                 input_inputs):
        '''
        Constructor
        '''

        self._cell = cell
        self._numiters = numiters
        self._num_capsules = num_capsules
        self._capsule_dim = capsule_dim
        self._attention = attention_mechanism
        self._input_context = input_context
        self._input_activation = input_activation
        self._input_inputs = input_inputs
        norm = tf.norm(
            self._attention.values,
            axis=-1,
            keep_dims=True) + np.finfo(np.float32).eps
        norm = tf.Print(norm, [tf.reduce_min(norm)])
        self._features = self._attention.values/norm
        norm = tf.Print(norm, [tf.reduce_any(tf.is_nan(self._features))])


    @property
    def output_size(self):
        '''return cell output size'''

        return tf.TensorShape([self._num_capsules, self._capsule_dim])

    @property
    def batch_size(self):
        '''batch size'''
        return self._attention.batch_size

    @property
    def state_size(self):
        '''return cell state size'''

        return DynamicRoutingAttentionWrapperState(
            cell_state=self._cell.state_size,
            alignments=(self._attention.alignments_size,),
            capsule_activities=self._capsule_dim*self._num_capsules,
            reconstruction=tf.shape(self._attention.keys)[1]*self._capsule_dim,
            context=self._attention.values.shape[-1])

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return DynamicRoutingAttentionWrapperState(
            cell_state=self._cell.zero_state(batch_size, dtype),
            alignments=(self._attention.initial_alignments(batch_size, dtype),),
            capsule_activities=tf.zeros(
                [batch_size, self._capsule_dim*self._num_capsules],
                dtype),
            reconstruction=tf.zeros(
                [batch_size,
                 tf.shape(self._attention.keys)[1]*self._capsule_dim],
                dtype),
            context=tf.zeros(
                [batch_size, self._attention.values.shape[-1]],
                dtype))


    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell'''

        capsule_activities = tf.reshape(
            state.capsule_activities,
            [self.batch_size, self._num_capsules, self._capsule_dim])

        #get the capsule activities from the chosen output
        chosen_capsule_activities = tf.squeeze(tf.matmul(
            tf.expand_dims(inputs, 1),
            capsule_activities
        ), [1])

        cell_inputs = []
        if self._input_inputs:
            cell_inputs.append(inputs)
        if self._input_context:
            cell_inputs.append(state.context)
        if self._input_activation:
            cell_inputs.append(chosen_capsule_activities)
        cell_inputs = tf.concat(cell_inputs, -1)

        reconstruction = tf.reshape(
            state.reconstruction, [self.batch_size, -1, self._capsule_dim]
        )

        reconstruction = reconstruction + tf.multiply(
            tf.expand_dims(state.alignments[0], 2),
            tf.expand_dims(chosen_capsule_activities, 1)
        )

        #use the wrapped cell
        cell_output, next_cell_state = self._cell(cell_inputs, state.cell_state)

        #get the alignments using attention
        alignments = self._attention(cell_output, state.alignments[0])
        context = tf.squeeze(tf.matmul(
            tf.expand_dims(alignments, 1),
            self._attention.values
        ), [1])


        predictions, routing_logits = self._predict(
            cell_output, alignments)

        #iteratively determine the routing weights
        for j in range(self._numiters + 1):
            with tf.name_scope('iter%d' % j):

                #compute the routing weights
                routing_weights = tf.nn.softmax(routing_logits, 1)

                #compute the capsule activities
                capsule_activities = _softmax_squash(tf.squeeze(tf.matmul(
                    tf.expand_dims(routing_weights, 2),
                    predictions,
                    name='capsule_activities'), [2]))

                if j == self._numiters:
                    break

                #compute the new routing logits by comparing the predicted
                #output to the actual output
                logits_update = tf.reduce_sum(
                    tf.expand_dims(capsule_activities, 2)*predictions, -1,
                    name='routing_logits_update')

                routing_logits += logits_update

        next_state = DynamicRoutingAttentionWrapperState(
            cell_state=next_cell_state,
            alignments=(alignments,),
            capsule_activities=tf.reshape(capsule_activities,
                                          [self.batch_size, -1]),
            reconstruction=tf.reshape(reconstruction, [self.batch_size, -1]),
            context=context
        )

        #return the lengths of the capsule inputs as the output
        return capsule_activities, next_state

    def _predict(self, cell_output, alignments):
        '''make a prediction'''

        '''inputs = tf.concat([
            self._attention.values,
            tf.tile(
                tf.expand_dims(cell_output, 1),
                [1, tf.shape(self._attention.values)[1], 1])
        ], 2)'''
        inputs = self._features
        inputs = tf.multiply(inputs, tf.expand_dims(alignments, 2))

        prediction_weights = tf.get_variable(
            name='prediction_weights',
            shape=[self._num_capsules, inputs.shape[-1],
                   self._capsule_dim],
            dtype=tf.float32,
            initializer=_initializer(
                int(inputs.shape[-1]), self._capsule_dim)
        )

        '''init_logits_weights = tf.get_variable(
            name='init_logits_weights',
            shape=[self._num_capsules, inputs.shape[-1]],
            dtype=tf.float32,
            initializer=_initializer(int(inputs.shape[-1]), 1)
        )'''

        predictions = tf.tensordot(
            inputs,
            prediction_weights,
            [[2], [1]]
        )
        predictions = tf.transpose(predictions, [0, 2, 1, 3])

        '''logits = tf.tensordot(
            inputs,
            init_logits_weights,
            [[2], [1]]
        )
        logits = tf.transpose(logits, [0, 2, 1])'''

        logits = tf.layers.dense(cell_output, self._num_capsules)
        logits = tf.tile(
            tf.expand_dims(logits, 2),
            [1, 1, tf.shape(inputs)[1]]
        )

        return predictions, logits

def _softmax_squash(tensor):
    '''the nonlinearity for the capsule inputs

    args:
        tensor: [batch_size, num_capsules, dim]'''

    with tf.name_scope('softmax_squash'):
        norm = tf.norm(tensor, axis=-1, keep_dims=True)
        sqnorm = tf.square(norm)
        out = norm/tf.reduce_sum(sqnorm, 1, keep_dims=True)*tensor

    return out

def _sigmoid_squash(tensor):
    '''the nonlinearity for predictions

    args:
        tensor: [batch_size, num_capsules, time, dim]'''

    with tf.name_scope('sigmoid_squash'):
        norm = tf.norm(tensor, axis=-1, keep_dims=True)
        sqnorm = tf.square(norm)
        out = norm/(1+sqnorm)*tensor

    return out

def _mask_logits(logits, seq_length):
    '''mask the logits'''

    with tf.name_scope('mask_logits'):

        indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(logits)[1]), 0),
            [tf.shape(logits)[0], 1])

        condition = tf.less(
            indices, tf.expand_dims(seq_length, 1))

        masked = tf.where(condition, logits, -float('inf')*tf.ones_like(logits))

        return masked

def _initializer(fan_in, fan_out):
    '''create an initializer'''

    limit = (6.0/(fan_in + fan_out))**0.5

    return tf.random_uniform_initializer(
        minval=-limit,
        maxval=limit
    )

class NormOutputWrapper(tf.contrib.rnn.RNNCell):
    '''output the log-norm of the capsules'''

    def __init__(self, cell):
        '''
        constructor

        args:
            cell: the wrapped cell
        '''

        self._cell = cell

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.output_size[0]

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

        return tf.log(tf.norm(output, axis=-1)), new_state

class DynamicRoutingWrapper(tf.contrib.rnn.RNNCell):
    '''wrapper with dynamic routing'''

    def __init__(self, cell, num_capsules, capsule_dim, numiters):
        '''constructor'''

        self._cell = cell
        self._numiters = numiters
        self._num_capsules = num_capsules
        self._capsule_dim = capsule_dim

    @property
    def output_size(self):
        '''return cell output size'''

        return tf.TensorShape([self._num_capsules, self._capsule_dim])

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        #compute the output of the wrapped cell
        output, new_state = self._cell(inputs, state, scope)

        #predict the capsule activities
        predictions = self._predict(output)

        #the initial logits
        init_logits = tf.get_variable(
            name='init_logits',
            shape=[self._num_capsules, self._call.output_size[0]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )
        routing_logits = tf.tile(
            tf.expand_dims(init_logits, 0),
            [tf.shape(output)[0], 1, 1])

        for i in range(self._numiters + 1):
            routing_weights = tf.nn.softmax(routing_logits, 1)
            capsule_activities = _softmax_squash(tf.squeeze(tf.matmul(
                tf.expand_dims(routing_weights, 2),
                predictions
            ), [2]))
            if i == self._numiters:
                break
            logits_update = tf.squeeze(tf.matmul(
                predictions,
                tf.expand_dims(capsule_activities, 3)
            ), [3])
            routing_logits += logits_update

        return capsule_activities, new_state


    def _predict(self, output):
        '''predict the activities of the capsules'''

        weights = tf.get_variable(
            name='weights',
            shape=[self._num_capsules, self._call.output_size[0],
                   output.shape[-1], self._capsule_dim],
            dtype=tf.float32,
            initializer=_initializer(
                int(output.shape[-1]), self._capsule_dim)
        )

        predictions = tf.squeeze(tf.matmul(
            tf.expand_dims(tf.tile(
                tf.expand_dims(output, 1),
                [1, self._num_capsules, 1, 1]), 3),
            tf.tile(
                tf.expand_dims(weights, 0),
                [tf.shape(output)[0], 1, 1, 1, 1])
        ), [3])

        return predictions

class AttentionWrapperState(
        collections.namedtuple('DynamicRoutingAttentionWrapperState',
                               ('cell_state',
                                'alignments',
                                'context',
                                'remaining_prob'))):
    '''
    Tupple for DynamicRoutingAttentionWrapperState

    args:
        cell_state: the state of the wrapped cell
        alignments: the current alignments
        context: the sum of the activities of the capsules
        remaining_prob: the ramaining probability
    '''

    pass

class AttentionWrapper(tf.contrib.rnn.RNNCell):
    '''A dynamic routing rnn cell'''

    def __init__(self, cell, attention_mechanism):
        '''
        Constructor
        '''

        self._cell = cell
        self._attention = attention_mechanism

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.output_size + int(self._attention.values.shape[-1])

    @property
    def state_size(self):
        '''return cell state size'''

        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            alignments=(self._attention.alignments_size,),
            context=self._attention.values.shape[-1],
            remaining_prob=self._attention.alignments_size)

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return AttentionWrapperState(
            cell_state=self._cell.zero_state(batch_size, dtype),
            alignments=(self._attention.initial_alignments(batch_size, dtype),),
            context=tf.zeros(
                [batch_size, self._attention.values.shape[-1]],
                dtype),
            remaining_prob=tf.ones_like(
                self._attention.initial_alignments(batch_size, dtype)))


    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell'''

        cell_inputs = tf.concat([inputs, state.context], -1)

        #use the wrapped cell
        cell_output, next_cell_state = self._cell(cell_inputs, state.cell_state)

        #get the alignments using attention
        alignments = self._attention(cell_output, state.alignments[0])
        alignments = alignments*state.remaining_prob
        context = tf.squeeze(tf.matmul(
            tf.expand_dims(alignments, 1),
            self._attention.values
        ), [1])
        output = tf.concat([context, cell_output], 1)
        remaining_prob = state.remaining_prob - alignments

        next_state = AttentionWrapperState(
            cell_state=next_cell_state,
            alignments=(alignments,),
            context=context,
            remaining_prob=remaining_prob
        )

        return output, next_state
