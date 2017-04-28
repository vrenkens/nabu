'''@file beam_search_decoder.py
contains the BeamSearchDecoder'''

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
import decoder
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

class BeamSearchDecoder(decoder.Decoder):
    '''Beam search decoder'''

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a list of [batch_size x ...] tensors
            input_seq_length: the input sequence lengths as a list of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a list of length beam_width
                containing [batch_size x ...] sparse tensors, the beam elements
                are sorted from best to worst
            - the sequence scores as a [batch_size x beam_width] tensor
        '''

        with tf.name_scope('beam_search_decoder'):
            max_output_length = int(self.conf['max_steps'])
            batch_size = tf.shape(inputs[0])[0]
            beam_width = int(self.conf['beam_width'])

            #encode the inputs [batch_size x output_length x output_dim]
            encoded, encoded_seq_length = self.model.encoder(
                inputs=inputs,
                input_seq_length=input_seq_length,
                is_training=False)

            encoded_dim = [int(e.get_shape()[2]) for e in encoded]

            #repeat the encoded inputs for all beam elements
            encoded = [_stack_beam(tf.stack(
                [e]*beam_width, axis=1)) for e in encoded]
            encoded_seq_length = [_stack_beam(tf.stack(
                [e]*beam_width, axis=1))  for e in encoded_seq_length]


            def body(step, beam):
                '''the body of the decoding while loop

                Args:
                    beam: a Beam object containing the current beam

                returns:
                    the loop vars'''

                with tf.variable_scope('body'):

                    #get the previous output
                    prev_output = _stack_beam(beam.sequences[:, :, step])
                    states = [_stack_beam(s)
                              for s in nest.flatten(beam.states)]
                    states = nest.pack_sequence_as(beam.states, states)

                    #compute the next state and logits
                    logits, states = self.model.decoder.step(
                        encoded=encoded,
                        encoded_seq_length=encoded_seq_length,
                        targets=[prev_output],
                        state=states,
                        is_training=False)

                    #put the states and logits in the format for the beam
                    states = [_unstack_beam(s, beam_width)
                              for s in nest.flatten(states)]
                    states = nest.pack_sequence_as(beam.states, states)
                    logits = _unstack_beam(logits[0], beam_width)

                    #update the beam
                    beam = beam.update(logits, states, step)

                    step = step + 1

                return step, beam

            def cb_cond(step, beam):
                '''the condition of the decoding while loop

                Args:
                    step: the decoding step
                    beam: a Beam object containing the current beam

                returns:
                    a boolean that evaluates to True if the loop should
                    continue'''

                with tf.variable_scope('cond'):

                    #check if all beam elements have terminated
                    cont = tf.logical_and(
                        tf.logical_not(beam.all_terminated(
                            step, self.model.decoder.output_dims[0] - 1)),
                        tf.less(step, max_output_length))

                return cont


            #initialise the loop variables
            negmax = tf.tile(
                [[-tf.float32.max]],
                [batch_size, beam_width-1])
            scores = tf.concat(
                [tf.zeros([batch_size, 1]), negmax], 1)
            lengths = tf.zeros(
                [batch_size, beam_width],
                dtype=tf.int32)
            sequences = tf.zeros(
                shape=[batch_size, beam_width,
                       max_output_length],
                dtype=tf.int32)
            sequences += self.model.decoder.output_dims[0] - 1
            states = self.model.decoder.zero_state(
                encoded_dim, beam_width*batch_size)
            states = nest.pack_sequence_as(
                states,
                [_unstack_beam(s, beam_width) for s in nest.flatten(states)])
            beam = Beam(sequences, lengths, states, scores)
            step = tf.constant(0)

            #run the while loop
            _, beam = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, beam],
                parallel_iterations=1,
                back_prop=False)

        #convert the output sequences to the sparse sequences
        with tf.name_scope('sparse_output'):
            sequences = tf.concat(tf.unstack(
                beam.sequences[:, :, 1:], axis=1), axis=0)
            lengths = tf.concat(tf.unstack(beam.lengths - 1, axis=1), axis=0)

            sparse_sequences = dense_sequence_to_sparse(sequences, lengths)
            sparse_sequence_list = tf.sparse_split(
                sp_input=sparse_sequences,
                num_split=beam_width,
                axis=0)


        return sparse_sequence_list, beam.scores

    @staticmethod
    def get_output_dims(output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

        return [output_dim + 1 for output_dim in output_dims]

class Beam(namedtuple('Beam', ['sequences', 'lengths', 'states', 'scores'])):
    '''a named tuple class for a beam

    the tuple fields are:
        - sequences: the sequences as a [batch_size x beam_width x max_steps]
            tensor
        - lengths: the length of the sequences as a [batch_size x beam_width]
            tensor
        - states: the state of the decoder as a possibly nested tuple of
            [batch_size x beam_width x state_dim] tensors
        - scores: the score of the beam element as a [batch_size x beam_width]
            tensor
        '''

    def update(self, logits, states, step):
        '''update the beam by expanding the beam with new hypothesis
        and selecting the best ones. Use as beam = beam.update(...)

        Args:
            logits: the decoder output logits as a
                [batch_size x beam_width x numlabels] tensor
            states: the decoder output states as a possibly nested tuple of
                [batch_size x beam_width x state_dim] tensor
            step: the current step

        Returns:
            a new updated Beam as a Beam object'''

        with tf.variable_scope('update'):

            numlabels = int(logits.get_shape()[2])
            max_steps = int(self.sequences.get_shape()[2])
            beam_width = self.beam_width
            batch_size = tf.shape(self.sequences)[0]

            #get flags for finished beam elements: [batch_size x beam_width]
            finished = tf.logical_and(
                tf.equal(self.sequences[:, :, step], numlabels-1),
                tf.not_equal(step, 0))

            with tf.variable_scope('scores'):

                #compute the log probabilities and vectorise
                #[batch_size x beam_width*numlabels]
                logprobs = tf.reshape(tf.log(tf.nn.softmax(logits)),
                                      [batch_size, -1])

                #put the old scores in the same format as the logrobs
                #[batch_size x beam_width*numlabels]
                oldscores = tf.reshape(tf.tile(tf.expand_dims(self.scores, 2),
                                               [1, 1, numlabels]),
                                       [batch_size, -1])

                #compute the new scores
                newscores = oldscores + logprobs

                #only update the scores of the unfinished beam elements
                #[batch_size x beam_width*numlabels]
                full_finished = tf.reshape(
                    tf.tile(tf.expand_dims(finished, 2), [1, 1, numlabels]),
                    [batch_size, -1])
                scores = tf.where(full_finished, oldscores, newscores)

                #set the scores of expanded beams from finished elements to
                #negative maximum [batch_size x beam_width*numlabels]
                expanded_finished = tf.reshape(tf.concat(
                    [tf.tile([[[False]]], [batch_size, beam_width, 1]),
                     tf.tile(tf.expand_dims(finished, 2),
                             [1, 1, numlabels-1])], 2)
                                               , [batch_size, -1])

                scores = tf.where(
                    expanded_finished,
                    tf.tile([[-scores.dtype.max]],
                            [batch_size, numlabels*beam_width]),
                    scores)


            with tf.variable_scope('lengths'):
                #update the sequence lengths for the unfinshed beam elements
                #[batch_size x beam_width]
                lengths = tf.where(finished, self.lengths, self.lengths+1)

                #repeat the lengths for all expanded elements
                #[batch_size x beam_width*numlabels]
                lengths = tf.reshape(
                    tf.tile(tf.expand_dims(lengths, 2), [1, 1, numlabels]),
                    [batch_size, -1])



            with tf.variable_scope('prune'):

                #select the best beam elements according to normalised scores
                float_lengths = tf.cast(lengths, tf.float32)
                _, indices = tf.nn.top_k(scores/float_lengths, beam_width)

                #from the indices, compute the expanded beam elements and the
                #selected labels
                expanded_elements = tf.floordiv(indices, numlabels)
                labels = tf.mod(indices, numlabels)

                #transform indices and expanded_elements for gather ops
                offset = tf.expand_dims(
                    tf.range(batch_size)*beam_width*numlabels,
                    1)
                indices = indices + offset
                offset = tf.expand_dims(
                    tf.range(batch_size)*beam_width,
                    1)
                expanded_elements = expanded_elements + offset

                #put the selected label for the finished elements to zero
                finished = tf.gather(tf.reshape(finished, [-1]),
                                     expanded_elements)
                labels = tf.where(
                    finished,
                    tf.tile([[numlabels-1]], [batch_size, beam_width]),
                    labels)

                #select the best lengths and scores
                lengths = tf.gather(tf.reshape(lengths, [-1]), indices)
                scores = tf.gather(tf.reshape(scores, [-1]), indices)

            #get the states for the expanded beam elements
            with tf.variable_scope('states'):
                #flatten the states
                flat_states = nest.flatten(states)

                #select the states
                flat_states = [tf.gather(
                    tf.reshape(s, [-1, int(s.get_shape()[2])]),
                    expanded_elements)
                               for s in flat_states]

                #pack them in the original format
                states = nest.pack_sequence_as(states, flat_states)

            with tf.variable_scope('sequences'):
                #get the sequences for the expanded elements
                sequences = tf.gather(
                    tf.reshape(self.sequences, [-1, max_steps]),
                    expanded_elements)

                #seperate the real sequence from the padding
                real = sequences[:, :, :step+1]
                padding = sequences[:, :, step+2:]

                #construct the new sequences by adding the selected labels
                labels = tf.expand_dims(labels, 2)
                sequences = tf.concat([real, labels, padding], 2)

                #specify the shape of the sequences
                sequences.set_shape([None, beam_width, max_steps])

            return Beam(sequences, lengths, states, scores)

    def all_terminated(self, step, s_label):
        '''check if all elements in the beam have terminated
        Args:
            step: the current step
            s_label: the value of the sentence border label
                (sould be last label)

        Returns:
            a bool tensor'''
        with tf.variable_scope('all_terminated'):
            return tf.equal(tf.reduce_sum(self.sequences[:, :, step]),
                            s_label*self.beam_width)

    @property
    def beam_width(self):
        '''get the size of the beam'''
        return int(self.lengths.get_shape()[1])

    @property
    def batch_size(self):
        '''get the size of the beam'''
        return tf.shape(self.lengths)[0]


def _stack_beam(tensor):
    '''converts a [Batch_size x beam_width x ...] Tensor into a
    [Batch_size * beam_width x ...] Tensor

    Args:
        The [Batch_size x beam_width x ...] Tensor to be converted

    Returns:
        The converted [Batch_size * beam_width x ...] Tensor'''

    return tf.concat(tf.unstack(tensor, axis=1), axis=0)

def _unstack_beam(tensor, beam_width):
    '''converts a [Batch_size * beam_width x ...] Tenor into a
    [Batch_size x beam_width x ...] Tensor

    Args:
        The [Batch_size * beam_width x ...] Tensor to be converted
        batch_size: the batch size

    Returns:
        The converted [Batch_size x beam_width x ...] Tensor'''

    return tf.stack(tf.split(tensor, beam_width), axis=1)
