'''@file beam_search_decoder
contain the BeamSearchDecoder'''

import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest

class BeamSearchState(
        collections.namedtuple('BeamSearchState',
                               ('cell_states',
                                'predicted_ids',
                                'logprobs',
                                'lengths',
                                'finished'))):
    '''
    class for the beam search state

    - cell_states: the cell states as a tuple of [batch_size x beam_width x s]
        tensor
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
        collections.namedtuple('BeamSearchState',
                               ('predicted_ids',
                                'scores'))):
    '''
    class for the output of the BeamSearchDecoder

    - predicted_ids: a [batch_size x beam_width x max_length] tensor
        containing the predicted ids
    - scores: a [batch_size x beam_width] tensor containing the scores
    '''

    pass

class BeamSearchDecoder(tf.contrib.seq2seq.Decoder):
    '''the beam search decoder'''

    @property
    def batch_size(self):
        '''the batch size'''

        return self._batch_size

    @property
    def output_size(self):
        '''the output size (empty)'''

        return ()

    @property
    def output_dtype(self):
        '''the output dtype (empty)'''

        return ()

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
                lambda s: tf.reshape(s, [self.batch_size, self.beam_width, -1]),
                self.initial_state)
            predicted_ids = tf.TensorArray(
                dtype=tf.int32,
                size=0
            )
            scores = tf.concat([
                tf.zeros([self.batch_size, 1]),
                tf.fill([self.batch_size, self.beam_width - 1], -np.inf)], 1)
            lengths = tf.zeros([self.batch_size, self.beam_width],
                               dtype=tf.int32)
            finished = tf.zeros([self.batch_size, self.beam_width],
                                dtype=tf.bool)
            state = BeamSearchState(cell_states, predicted_ids, scores, lengths,
                                    finished)

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

        with tf.name_scope(name or 'beam_search_decoder'):

            #call the cell to get the new outputs and states
            cell_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
            cell_states = nest.map_structure(
                lambda s: tf.reshape(s, [-1, tf.shape(s)[-1]]),
                state.cell_states)
            outputs, cell_states = self.cell(cell_inputs, cell_states)
            if self.output_layer:
                outputs = self.output_layer(outputs)
            outputs = tf.reshape(
                outputs, [self.batch_size, self.beam_width, -1])
            cell_states = nest.map_structure(
                lambda s: tf.reshape(s, [self.batch_size, self.beam_width, -1]),
                cell_states)

            output_dim = tf.shape(outputs)[-1]

            #collect all possible hypotheses to expand
            new_logprobs = tf.nn.log_softmax(outputs)
            old_predicted_ids = state.predicted_ids.stack()
            predicted_ids = tf.tile(
                tf.expand_dims(old_predicted_ids, 2),
                [1, 1, output_dim, 1])
            new_ids = tf.tile(
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
                tf.expand_dims(state.lengths + 1, 2),
                [1, 1, output_dim])
            cell_states = nest.map_structure(
                lambda s: tf.tile(tf.expand_dims(s, 2),
                                  [1, 1, output_dim, tf.shape(s)[-1]]),
                cell_states)

            new_ids = tf.reshape(new_ids, [self.batch_size, -1])
            lengths = tf.reshape(lengths, [self.batch_size, -1])
            logprobs = tf.reshape(logprobs, [self.batch_size, -1])
            cell_states = nest.map_structure(
                lambda s: tf.reshape(s, [self.batch_size, -1, s.shape[-1]]),
                cell_states)
            predicted_ids = tf.reshape(
                predicted_ids, [self.batch_size, -1, time])


            #add the hypotheses for the finished elements to stay
            stay_logprobs = tf.where(
                state.finsished,
                state.logprobs,
                -new_logprobs.dtype.max*tf.ones([self.batch_size,
                                                 self.beam_width]))
            stay_ids = tf.fill([self.batch_size, self.beam_width],
                               self.end_token)
            new_ids = tf.concat([new_ids, stay_ids], 1)
            logprobs = tf.concat([logprobs, stay_logprobs], 1)
            lengths = tf.concat([lengths, state.lengths], 1)
            cell_states = nest.map_structure(
                lambda s1, s2: tf.concat([s1, s2], 1),
                cell_states, state.cell_states)
            predicted_ids = tf.concat([predicted_ids, old_predicted_ids], 1)

            #update the scores for all hypotheses
            scores = _score(logprobs, lengths, self.length_penalty_weight)

            #select the best hypotheses
            _, indices = tf.top_k(scores, self.beam_width)
            lengths = tf.gather_nd(lengths, indices)
            new_ids = tf.gather_nd(new_ids, indices)
            logprobs = tf.gather_nd(logprobs, indices)
            cell_states = nest.map_structure(
                lambda s: tf.gather_nd(s, indices),
                cell_states
            )
            finished = tf.equal(new_ids, self.end_token)
            predicted_ids = tf.gather_nd(predicted_ids, indices)
            predicted_ids = tf.concat([predicted_ids, new_ids], 2)
            predicted_ids_array = predicted_ids = tf.TensorArray(
                dtype=tf.int32,
                size=time+1
            )
            predicted_ids_array.unstack(predicted_ids)

            #compute the new inputs
            next_inputs = self.embedding(new_ids)

            next_state = BeamSearchState(cell_states, predicted_ids_array,
                                         logprobs, lengths, finished)
            outputs = ()

        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        '''
        Finalize and return the predicted_ids.
            Args:
                outputs: An empty tuple.
                final_state: An instance of BeamSearchState.
                sequence_lengths: An `int64` tensor shaped
                    `[batch_size, beam_width]`. The sequence lengths
            Returns:
                - An instance of BeamSearchDecoderOutput
                - The final state
        '''

        #put negative one for the elements exeeding the sequence length
        predicted_ids = final_state.predicted_ids.stack()
        time = tf.tile(
            tf.expand_dims(tf.range(tf.shape(predicted_ids)[-1]), 0),
            [self.batch_size, 1])
        lengths = tf.tile(
            tf.expand_dims(sequence_lengths, 1),
            [1, tf.shape(predicted_ids)[-1]])
        predicted_ids = tf.where(
            tf.less(time, lengths),
            predicted_ids,
            -tf.ones_like(predicted_ids))

        #compute the final scores
        scores = _score(final_state.logprobs, sequence_lengths,
                        self.length_penalty_weight)

        outputs = BeamSearchDecoderOutput(predicted_ids, scores)

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

    return logprobs/_length_penalty(lengths, length_penalty_weight)

def _length_penalty(lengths, length_penalty_weight):
    '''
    lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length

    Returns:
        The length_penalty.
    '''

    if length_penalty_weight == 0:
        return 1.0

    return tf.div(
        (5. + tf.to_float(lengths))**length_penalty_weight,
        (5. + 1.)**length_penalty_weight)
