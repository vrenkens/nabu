'''@file beam_search_decoder.py
contains the BeamSearchDecoder'''

import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
import decoder
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

class BeamSearchDecoder(decoder.Decoder):
    '''Beam search decoder'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        #get the alphabet
        self.alphabet = conf['alphabet'].split(' ')

        super(BeamSearchDecoder, self).__init__(conf, model)

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a list of [batch_size x ...] tensors
            input_seq_length: the input sequence lengths as a list of
                [batch_size] vectors

        Returns:
            a dictionary with outputs containing:
                - the decoded sequences as a
                [beam_width batch_size x length x num_labels] tensor
                - the sequence lengths as a [beam_width x batch_size] tensor
                - the sequence scores as a [beam_width x batch_size] tensor
        '''

        with tf.name_scope('beam_search_decoder'):
            max_output_length = int(self.conf['max_steps'])
            batch_size = tf.shape(inputs.values()[0])[0]
            beam_width = int(self.conf['beam_width'])

            #encode the inputs [batch_size x output_length x output_dim]
            encoded, encoded_seq_length = self.model.encoder(
                inputs=inputs,
                input_seq_length=input_seq_length,
                is_training=False)

            encoded_dim = {e:int(encoded[e].get_shape()[2]) for e in encoded}

            #repeat the encoded inputs for all beam elements
            encoded = {e:_stack_beam(tf.stack(
                [encoded[e]]*beam_width, axis=1)) for e in encoded}
            encoded_seq_length = {e:_stack_beam(tf.stack(
                [encoded_seq_length[e]]*beam_width, axis=1))
                                  for e in encoded_seq_length}

            output_name = self.model.output_dims.keys()[0]


            def body(step, beam, hypotheses):
                '''the body of the decoding while loop

                Args:
                    step: the current step
                    beam: a Beam object containing the current beam
                    hypotheses: the finished hypotheses

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
                        targets={output_name:prev_output},
                        state=states,
                        is_training=False)

                    #put the states and logits in the format for the beam
                    states = [_unstack_beam(s, beam_width)
                              for s in nest.flatten(states)]
                    states = nest.pack_sequence_as(beam.states, states)
                    logits = _unstack_beam(logits.values()[0], beam_width)

                    #update the hypotheses
                    hypotheses = hypotheses.update(logits, step, beam)

                    #update the beam
                    beam = beam.update(logits, states, step)

                    step = step + 1

                return step, beam, hypotheses

            def cb_cond(step, beam, hypotheses):
                '''the condition of the decoding while loop

                Args:
                    step: the decoding step
                    beam: a Beam object containing the current beam
                    hypotheses: the finished hypotheses

                returns:
                    a boolean that evaluates to True if the loop should
                    continue'''

                with tf.variable_scope('cond'):

                    #check if maximum number of steps have been taken
                    cont = tf.less(step, max_output_length)

                return cont


            #initialise the loop variables
            negmax = tf.tile(
                [[-tf.float32.max]],
                [batch_size, beam_width-1])
            scores = tf.concat(
                [tf.zeros([batch_size, 1]), negmax], 1)
            sequences = tf.zeros(
                shape=[batch_size, beam_width,
                       max_output_length],
                dtype=tf.int32)
            sequences += self.model.decoder.output_dims.values()[0] - 1
            states = self.model.decoder.zero_state(
                encoded_dim, beam_width*batch_size)
            states = nest.pack_sequence_as(
                states,
                [_unstack_beam(s, beam_width) for s in nest.flatten(states)])
            beam = Beam(sequences, states, scores)
            step = tf.constant(0)

            sequences = tf.TensorArray(
                dtype=tf.int32,
                size=max_output_length,
                infer_shape=True,
                name='sequences'
            )
            scores = tf.TensorArray(
                dtype=tf.float32,
                size=max_output_length,
                infer_shape=True,
                name='scores'
            )
            hypotheses = Hypotheses(sequences, scores)

            #run the while loop
            _, _, hypotheses = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, beam, hypotheses],
                parallel_iterations=1,
                back_prop=False)

            with tf.name_scope('gather_results'):
                sequences = hypotheses.sequences.concat()
                lengths = tf.range(max_output_length)
                lengths = tf.stack([lengths]*beam_width, axis=1)
                lengths = tf.tile(tf.expand_dims(lengths, 2),
                                  [1, 1, batch_size])
                lengths = tf.concat(tf.unstack(lengths), 0)
                scores = hypotheses.scores.concat()

            if self.conf['num_keep'] != 'None':
                num_keep = int(self.conf['num_keep'])
                if num_keep < max_output_length*beam_width:
                    with tf.name_scope('prune_results'):
                        #only keep the top num_keep sequences
                        float_lengths = tf.cast(tf.transpose(lengths),
                                                tf.float32)
                        _, hypidx = tf.nn.top_k(
                            tf.transpose(scores)/float_lengths, num_keep)
                        hypidx = tf.transpose(hypidx)
                        batchidx = tf.stack([tf.range(batch_size)]*num_keep)
                        gatheridx = tf.stack([hypidx, batchidx], 2)
                        sequences = tf.gather_nd(sequences, gatheridx)
                        lengths = tf.gather_nd(lengths, gatheridx)
                        scores = tf.gather_nd(scores, gatheridx)

        return {output_name: (sequences, lengths, scores)}

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder as a dictionary
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''

        sequences = outputs.values()[0][0]
        lengths = outputs.values()[0][1]
        scores = outputs.values()[0][2]

        for i, name in enumerate(names):
            with open(os.path.join(directory, name), 'w') as fid:
                for b in range(sequences.shape[0]):
                    sequence = sequences[b, i][:lengths[b, i]]
                    score = scores[b, i]
                    text = ' '.join([self.alphabet[s] for s in sequence])
                    fid.write('%f %s\n' % (score, text))

    def evaluate(self, outputs, references, reference_seq_length):
        '''evaluate the output of the decoder

        args:
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            the error of the outputs
        '''

        sequences = outputs.values()[0][0]
        lengths = outputs.values()[0][1]
        scores = tf.transpose(outputs.values()[0][2])


        #select the best sequences
        float_lengths = tf.cast(tf.transpose(lengths), tf.float32)
        _, hypidx = tf.nn.top_k(scores/float_lengths)
        batchidx = tf.range(tf.shape(sequences)[1])
        gatheridx = tf.stack([hypidx[:, 0], batchidx], 1)
        sequences = tf.gather_nd(sequences, gatheridx)
        lengths = tf.gather_nd(lengths, gatheridx)
        lengths = tf.transpose(lengths)

        #convert the references to sparse representations
        sparse_targets = dense_sequence_to_sparse(
            references.values()[0], reference_seq_length.values()[0])

        #convert the best sequences to sparse representations
        sparse_sequences = dense_sequence_to_sparse(
            sequences, lengths)

        #compute the edit distance
        loss = tf.reduce_mean(
            tf.edit_distance(sparse_sequences, sparse_targets))

        return loss


class Hypotheses(namedtuple('Hypotheses', ['sequences', 'scores'])):
    '''a class to hold all the comleted hypotheses

    the tuple fields are:
        - sequences: the completed sequences as a tensorarray
            of [beam_width x batch_size x max_steps] tensors
        - scores: the scores of the completed sequences as a tensorarray
            of [beam_width x batch_size] tensors
    '''

    def update(self, logits, step, beam):
        '''update the current hypotheses

        Args:
            logits: the decoder output logits as a
                [batch_size x beam_width x numlabels] tensor
            step: the current step
            beam: the current beam
        '''

        with tf.name_scope('update_hypotheses'):
            #cut off the beams sos label
            cut_sequences = beam.sequences[:, :, 1:]
            cut_sequences = tf.transpose(cut_sequences, [1, 0, 2])

            #get the scores for the end of sequence token for each sequence
            end_logprob = tf.log(tf.nn.softmax(logits))[:, :, -1]

            #get the scores for all new hypotheses
            new_scores = beam.scores + end_logprob
            new_scores = tf.transpose(new_scores)

            #add the scores to the finished hypotheses
            sequences = self.sequences.write(step, cut_sequences)
            scores = self.scores.write(step, new_scores)

        return Hypotheses(sequences, scores)

class Beam(namedtuple('Beam', ['sequences', 'states', 'scores'])):
    '''a named tuple class for a beam

    the tuple fields are:
        - sequences: the sequences in the beam as a
            [batch_size x beam_width x max_steps] tensor
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

            numlabels = int(logits.get_shape()[2]) - 1
            max_steps = int(self.sequences.get_shape()[2])
            beam_width = self.beam_width
            batch_size = tf.shape(logits)[0]

            with tf.variable_scope('scores'):

                #compute the log probabilities of all labels except eos
                logprobs = tf.log(tf.nn.softmax(logits)[:, :, :-1])

                #vectorize logprobs
                logprobs = tf.reshape(logprobs, [batch_size, -1])

                #put the old scores in the same format as the logrobs
                #[batch_size x beam_width*numlabels]
                oldscores = tf.reshape(tf.tile(tf.expand_dims(self.scores, 2),
                                               [1, 1, numlabels]),
                                       [batch_size, -1])

                #compute the new scores
                scores = oldscores + logprobs

            with tf.variable_scope('prune'):
                #select the best beam elements
                scores, indices = tf.nn.top_k(scores, beam_width)

                #from the indices, compute the expanded beam elements and the
                #selected labels
                beamiddx = tf.floordiv(indices, numlabels)
                labels = tf.mod(indices, numlabels)

                #transform indices and expanded_elements for gather ops
                batchidx = tf.stack([tf.range(batch_size)]*beam_width, 1)
                gatheridx = tf.stack([batchidx, beamiddx], 2)

            #get the states for the expanded beam elements
            with tf.variable_scope('states'):
                #flatten the states
                flat_states = nest.flatten(states)

                #select the states
                flat_states = [tf.gather_nd(s, gatheridx) for s in flat_states]

                #pack them in the original format
                states = nest.pack_sequence_as(states, flat_states)

            with tf.variable_scope('sequences'):
                #get the sequences for the expanded elements
                sequences = tf.gather_nd(self.sequences, gatheridx)

                #seperate the real sequence from the padding
                real = sequences[:, :, :step+1]
                padding = sequences[:, :, step+2:]

                #construct the new sequences by adding the selected labels
                labels = tf.expand_dims(labels, 2)
                sequences = tf.concat([real, labels, padding], 2)

                #specify the shape of the sequences
                sequences.set_shape([None, beam_width, max_steps])

            return Beam(sequences, states, scores)


    @property
    def beam_width(self):
        '''get the size of the beam'''
        return int(self.scores.get_shape()[1])


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
