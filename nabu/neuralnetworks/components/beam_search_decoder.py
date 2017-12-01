'''@file beam_search_decoder
contain the BeamSearchDecoder'''

import collections
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest
from ops import map_ta


class BeamSearchState(
        collections.namedtuple('BeamSearchState',
                               ('cell_states',
                                'alignment_history',
                                'logprobs',
                                'lengths',
                                'finished'))):
    '''
    class for the beam search state

    - cell_states: the cell states as a tuple of [batch_size x beam_width x s]
        tensor
    - alignment_history: the alignments history as a TensorArray
    - predicted_ids: a [batch_size x beam_width] tensorArray containing the
        predicted ids
    - logprobs: a [batch_size x beam_width] tensor containing, the log
        probability of the beam elements
    - lengths: a [batch_size x beam_width] tensor containing the lengths
    - finished: a [batch_size x beam_width] tensor containing which elements
        have finished
    '''

    pass

class BeamSearchDecoderOutput(
        collections.namedtuple('BeamSearchDecoderOutput',
                               ('predicted_ids',
                                'parent_ids'))):
    '''
    class for the output of the BeamSearchDecoder

    - predicted_ids: a [batch_size x beam_width] tensor
        containing the predicted ids
    - parent_ids: a [batch_size x beam_width] tensor containing the id of
        the parent beam element
    '''

class BeamSearchDecoderFinalOutput(
        collections.namedtuple('BeamSearchDecoderFinalOutput',
                               ('predicted_ids',
                                'lengths',
                                'scores',
                                'alignments'))):
    '''
    class for the final output of the BeamSearchDecoder

    - predicted_ids: a [time x batch_size x beam_width] tensor
        containing the predicted ids
    - lengths: a [1 x batch_size x beam_width] tensor containing the lengths
    - scores: a [1 x batch_size x beam_width] tensor containing the scores
    - alignments: a [time x batch_size x beam_width x in_time] tensor containing
        the alignments
    '''

    pass

#pylint: disable=W0613
class BeamSearchDecoder(tf.contrib.seq2seq.Decoder):
    '''the beam search decoder'''

    @property
    def batch_size(self):
        '''the batch size'''

        return self._batch_size

    @property
    def output_size(self):
        '''the output size (empty)'''

        return BeamSearchDecoderOutput(
            predicted_ids=tf.TensorShape([self.beam_width]),
            parent_ids=tf.TensorShape([self.beam_width]))

    @property
    def output_dtype(self):
        '''the output dtype (empty)'''

        return BeamSearchDecoderOutput(
            predicted_ids=tf.int32,
            parent_ids=tf.int32)

    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 output_layer=None,
                 length_penalty_weight=0.0):
        '''constructor

        args:
            cell: An `RNNCell` instance.
            embedding: A callable that takes a vector tensor of `ids`
                (argmax ids), or the `params` argument for `embedding_lookup`.
            start_tokens: `int32` vector shaped `[batch_size]`, the start
                tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            initial_state: A (possibly nested tuple of...) tensors and
                TensorArrays.
            beam_width:  Python integer, the number of beams.
            output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
                `tf.layers.Dense`.  Optional layer to apply to the RNN output
                prior to storing the result or sampling.
            length_penalty_weight: Float weight to penalize length. Disabled
                with 0.0.
        '''

        self.cell = cell
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.initial_state = initial_state
        self.beam_width = beam_width
        self.output_layer = output_layer
        self.length_penalty_weight = length_penalty_weight
        self._batch_size = tf.shape(start_tokens)[0]

    def initialize(self, name=None):
        '''
        Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
            name: Name scope for any created operations.
        Returns:
            `(finished, initial_inputs, initial_state)`: initial values of
            'finished' flags, inputs and state.
        '''

        with tf.name_scope(name or 'initialize_beam_search'):

            finished = tf.zeros([self.batch_size, self.beam_width])
            start_tokens = tf.tile(tf.expand_dims(self.start_tokens, 1),
                                   [1, self.beam_width])
            inputs = self.embedding(start_tokens)
            cell_states = nest.map_structure(
                lambda s: _unstack(s, self.batch_size, self.beam_width),
                self.initial_state)
            logprobs = tf.concat([
                tf.zeros([self.batch_size, 1]),
                tf.fill([self.batch_size, self.beam_width - 1], -np.inf)], 1)
            lengths = tf.zeros([self.batch_size, self.beam_width],
                               dtype=tf.int32)
            finished = tf.zeros([self.batch_size, self.beam_width],
                                dtype=tf.bool)
            alignment_history = tf.TensorArray(tf.float32, size=0,
                                               dynamic_size=True)
            state = BeamSearchState(
                cell_states=cell_states,
                alignment_history=alignment_history,
                logprobs=logprobs,
                lengths=lengths,
                finished=finished)

        return finished, inputs, state

    def step(self, time, inputs, state, name=None):
        '''
        Called per step of decoding (but only once for dynamic decoding).

        Args:
            time: Scalar `int32` tensor. Current step number.
            inputs: RNNCell input (possibly nested tuple of) tensor[s] for
                this time step.
            state: RNNCell state (possibly nested tuple of) tensor[s] from
                previous time step.
            name: Name scope for any created operations.

        Returns:
            `(outputs, next_state, next_inputs, finished)`: `outputs` is an
            object containing the decoder output, `next_state` is a
            (structure of) state tensors and TensorArrays, `next_inputs` is
            the tensor that should be used as input for the next step,
            `finished` is a boolean tensor telling whether the sequence is
            complete, for each sequence in the batch.
        '''

        with tf.name_scope(name or 'beam_search'):

            with tf.name_scope('compute_outputs'):

                with tf.name_scope('stack_beams'):
                    #call the cell to get the new outputs and states
                    cell_inputs = _stack(inputs, self.batch_size,
                                         self.beam_width)
                    cell_states = nest.map_structure(
                        lambda s: _stack(s, self.batch_size, self.beam_width),
                        state.cell_states)

                outputs, cell_states = self.cell(cell_inputs, cell_states)
                if self.output_layer:
                    outputs = self.output_layer(outputs)

                with tf.name_scope('unstack_beams'):
                    outputs = _unstack(outputs, self.batch_size,
                                       self.beam_width)
                    cell_states = nest.map_structure(
                        lambda s: _unstack(s, self.batch_size, self.beam_width),
                        cell_states)


                output_dim = outputs.get_shape()[-1].value

            with tf.name_scope('expand_beam'):

                #collect all possible hypotheses to expand
                new_logprobs = tf.nn.log_softmax(outputs)
                predicted_ids = tf.tile(
                    tf.expand_dims(tf.expand_dims(tf.range(output_dim), 0), 0),
                    [self.batch_size, self.beam_width, 1])
                finished = tf.tile(
                    tf.expand_dims(state.finished, 2),
                    [1, 1, output_dim])
                new_logprobs = tf.where(
                    finished,
                    -new_logprobs.dtype.max*tf.ones_like(new_logprobs),
                    new_logprobs)
                logprobs = tf.tile(
                    tf.expand_dims(state.logprobs, 2),
                    [1, 1, output_dim]) + new_logprobs
                lengths = tf.tile(
                    tf.expand_dims(state.lengths, 2),
                    [1, 1, output_dim])
                cell_states = nest.map_structure(
                    lambda s: _tile_state(s, output_dim),
                    cell_states)

                predicted_ids = _stack_hypotheses(predicted_ids,
                                                  self.batch_size,
                                                  self.beam_width, output_dim)
                lengths = _stack_hypotheses(lengths, self.batch_size,
                                            self.beam_width, output_dim)
                lengths = tf.where(
                    tf.equal(predicted_ids, self.end_token),
                    lengths,
                    lengths + 1)
                logprobs = _stack_hypotheses(logprobs, self.batch_size,
                                             self.beam_width, output_dim)
                cell_states = nest.map_structure(
                    lambda s: _stack_hypotheses(s, self.batch_size,
                                                self.beam_width, output_dim),
                    cell_states)

                #add the hypotheses for the finished elements to stay
                stay_logprobs = tf.where(
                    state.finished,
                    state.logprobs,
                    -new_logprobs.dtype.max*tf.ones([self.batch_size,
                                                     self.beam_width]))
                stay_ids = tf.fill([self.batch_size, self.beam_width],
                                   self.end_token)
                predicted_ids = tf.concat([predicted_ids, stay_ids], 1)
                logprobs = tf.concat([logprobs, stay_logprobs], 1)
                lengths = tf.concat([lengths, state.lengths], 1)
                #concat states has to be called with the unfinished states first
                cell_states = nest.map_structure(
                    _concat_states,
                    cell_states, state.cell_states)

            with tf.name_scope('prune_beam'):

                #update the scores for all hypotheses
                scores = _score(logprobs, lengths, self.length_penalty_weight)

                #select the best hypotheses
                scores, indices = tf.nn.top_k(scores, self.beam_width)

                #get the selected ids and the selected beams from the indices
                parent_ids = tf.floor_div(indices, output_dim)
                parent_ids = tf.where(
                    tf.equal(parent_ids, self.beam_width),
                    tf.mod(indices, output_dim),
                    parent_ids, name='parent_ids')

                #prepare indices for gather_nd
                batch_indices = tf.tile(
                    tf.expand_dims(tf.range(self.batch_size), 1),
                    [1, self.beam_width])
                indices = tf.stack([batch_indices, indices], 2)

                #gather the best beam elements
                lengths = tf.gather_nd(lengths, indices, name='prune_lengths')
                predicted_ids = tf.gather_nd(predicted_ids, indices,
                                             name='prune_ids')
                logprobs = tf.gather_nd(logprobs, indices, 'prune_logprobs')
                cell_states = nest.map_structure(
                    lambda s: _gather_state(s, indices),
                    cell_states
                )


            finished = tf.equal(predicted_ids, self.end_token)

            if hasattr(cell_states, 'alignments'):
                alignment_history = state.alignment_history.write(
                    state.alignment_history.size(),
                    cell_states.alignments[0]
                )
            else:
                alignment_history = state.alignment_history.write(
                    state.alignment_history.size(),
                    tf.expand_dims(tf.zeros_like(predicted_ids), 2)
                )

            #compute the new inputs
            with tf.name_scope('next_inputs'):
                next_inputs = self.embedding(predicted_ids)

            next_state = BeamSearchState(
                cell_states=cell_states,
                alignment_history=alignment_history,
                logprobs=logprobs,
                lengths=lengths,
                finished=finished)

            outputs = BeamSearchDecoderOutput(predicted_ids, parent_ids)

        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        '''
        Finalize and return the predicted_ids.
            Args:
                outputs: An instance of BeamSearchDecoderOutput.
                final_state: An instance of BeamSearchState.
                sequence_lengths: An `int64` tensor shaped
                    `[batch_size, beam_width]`. The sequence lengths
            Returns:
                - An instance of BeamSearchDecoderFinalOutput
                - The final state
        '''

        with tf.name_scope('backwards_search'):

            #do backwards search through the predicted ids
            predicted_ids = tf.transpose(outputs.predicted_ids, [1, 2, 0])
            parent_ids = tf.transpose(outputs.parent_ids, [1, 2, 0])
            alignment_history = tf.transpose(
                final_state.alignment_history.stack(),
                [1, 2, 0, 3])

            def condition(time, sequences, beams, alignments):
                '''the condition of the while loop

                Args:
                    time: a scalar, the current time step
                    sequences: a tensorArray containing the sequences
                    beams: the current beams that are being explored
                    alignments: the tensorArray containing the alignments

                returns:
                    True if time reached 0
                '''

                return tf.not_equal(time, 0)

            def body(time, sequences, beams, alignments):
                '''the body of the while loop

                Args:
                    time: a scalar, the current time step
                    sequences: a tensorArray containing the sequences
                    beams: the current beams that are being explored shape
                        [batch_size, beam_width]
                    alignments: the tensorArray containing the alignments

                returns:
                    the updated time, sequences and beams
                '''
                new_time = tf.subtract(time, 1, name='new_time')


                with tf.name_scope('gather_indices'):

                    batch_indices = tf.tile(
                        tf.expand_dims(tf.range(self.batch_size), 1),
                        [1, self.beam_width])
                    indices = tf.stack([batch_indices, beams], 2)

                selected_ids = tf.gather_nd(predicted_ids[:, :, new_time],
                                            indices, name='selected_indices')
                new_sequences = sequences.write(new_time, selected_ids)
                new_beams = tf.gather_nd(parent_ids[:, :, new_time], indices,
                                         name='selected_beams')
                selected_alignments = tf.gather_nd(
                    alignment_history[:, :, new_time],
                    indices, name='selected_alignments')
                new_alignments = alignments.write(new_time, selected_alignments)

                return new_time, new_sequences, new_beams, new_alignments

            #create the initial loop variables
            init_time = tf.shape(predicted_ids)[-1]
            init_sequences = tf.TensorArray(
                dtype=tf.int32,
                size=init_time,
                name='init_sequences'
            )
            init_alignments = tf.TensorArray(
                dtype=tf.float32,
                size=init_time,
                name='init_alignments'
            )
            init_beams = tf.tile(
                tf.expand_dims(tf.range(self.beam_width), 0),
                [self.batch_size, 1], name='initial_beams')

            res = tf.while_loop(
                condition,
                body,
                loop_vars=[init_time, init_sequences, init_beams,
                           init_alignments])

            predicted_ids_out = res[1].stack(name='predicted_ids')
            alignments_out = res[3].stack(name='alignments')

            scores = _score(final_state.logprobs, final_state.lengths,
                            self.length_penalty_weight)
            scores = tf.expand_dims(scores, 0)
            lengths = tf.expand_dims(final_state.lengths, 0)

            outputs = BeamSearchDecoderFinalOutput(
                predicted_ids=predicted_ids_out,
                lengths=lengths,
                scores=scores,
                alignments=alignments_out
            )

        return outputs, final_state

def _score(logprobs, lengths, length_penalty_weight):
    '''
    score the beam elements

    Args:
        logprobs: The log probabilities with shape
            `[batch_size, beam_width, vocab_size]`.
        lengths: The array of sequence lengths.
        length_penalty_weight: Float weight to penalize length

        Returns:
            The scores normalized by the length_penalty.
    '''

    return tf.div(logprobs, _length_penalty(lengths, length_penalty_weight),
                  name='score')

def _length_penalty(lengths, length_penalty_weight):
    '''
    lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length

    Returns:
        The length_penalty.
    '''

    if length_penalty_weight == 0:
        return tf.constant(1.0, name='length_penalty')

    return tf.div(
        (5. + tf.to_float(lengths))**length_penalty_weight,
        (5. + 1.)**length_penalty_weight,
        name='length_penalty')

def _stack(tensor, batch_size, beam_width):
    '''
    stack the beam elements

    args:
        tensor: a [batch_size x beam_width x ...] tensor
        batch_size: the batch size
        beam_width: the beam_width

    returns: a [batch_size * beam_width x ...] tensor
    '''
    if isinstance(tensor, tf.TensorArray):
        return map_ta(partial(_stack, batch_size=batch_size,
                              beam_width=beam_width), tensor)

    if tensor.shape.ndims != 3:
        return tensor
    else:
        if tensor.shape[-1].value is not None:
            return tf.reshape(tensor, [batch_size*beam_width,
                                       tensor.get_shape()[-1]])
        else:
            return tf.reshape(tensor, [batch_size*beam_width, -1])

def _unstack(tensor, batch_size, beam_width):
    '''
    stack the beam elements

    args:
        tensor: a [batch_size * beam_width x ...] tensor
        batch_size: the batch size
        beam_width: the beam_width

    returns: a [batch_size x beam_width x ...] tensor
    '''
    if isinstance(tensor, tf.TensorArray):
        return map_ta(partial(_unstack, batch_size=batch_size,
                              beam_width=beam_width), tensor)

    if tensor.shape.ndims != 2:
        return tensor
    else:
        if tensor.shape[-1].value is not None:
            return tf.reshape(tensor, [batch_size, beam_width,
                                       tensor.get_shape()[-1]])
        else:
            return tf.reshape(tensor, [batch_size, beam_width, -1])

def _stack_hypotheses(tensor, batch_size, beam_width, output_dim):
    '''
    stack the hypotheses

    args:
        tensor: a [batch_size x beam_width x output_dim ...] tensor
        batch_size: the batch size
        beam_width: the beam_width
        output_dim: the output_dim

    returns: a [batch_size x beam_width*output_dim x ...] tensor
    '''
    if isinstance(tensor, tf.TensorArray):
        return map_ta(partial(_stack_hypotheses, batch_size=batch_size,
                              beam_width=beam_width, output_dim=output_dim),
                      tensor)

    if tensor.shape.ndims < 3:
        return tensor
    if tensor.shape.ndims == 3:
        return tf.reshape(tensor, [batch_size, beam_width*output_dim])
    if tensor.shape[-1].value is not None:

        return tf.reshape(tensor, [batch_size, beam_width*output_dim,
                                   tensor.get_shape()[-1]])

    return tf.reshape(tensor, [batch_size, beam_width*output_dim, -1])

def _tile_state(state, output_dim):
    '''
    tile the states

    args:
        state: the state to tile
        output_dim: the output dimension

    returns: the tiled state
    '''
    if isinstance(state, tf.TensorArray):
        return map_ta(partial(_tile_state, output_dim=output_dim), state)

    if state.shape.ndims != 3:
        return state

    return tf.tile(tf.expand_dims(state, 2), [1, 1, output_dim, 1])

def _concat_states(s1, s2):
    '''concatenate the states'''

    if isinstance(s1, tf.TensorArray):
        st1 = s1.stack()
        st2 = s2.stack()
        st = tf.concat([st1, st2], 2)
        sa = tf.TensorArray(s1.dtype, s1.size())
        return sa.unstack(st)

    if s1.shape.ndims == 0:
        return s1

    return tf.concat([s1, s2], 1)

def _gather_state(state, indices):
    '''do the gather on the strates'''

    if isinstance(state, tf.TensorArray):
        return map_ta(partial(_gather_state, indices=indices), state)

    if state.shape.ndims == 0:
        return state

    return tf.gather_nd(state, indices, name='prune_state')
