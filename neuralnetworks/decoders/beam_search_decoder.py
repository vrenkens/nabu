'''@file beam_search_decoder.py
contains the BeamSearchDecoder'''

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
import decoder
import processing

class BeamSearchDecoder(decoder.Decoder):
    '''greedy decoder'''

    def __init__(self, conf, classifier, classifier_scope, input_dim,
                 max_input_length, coder, expdir):
        '''
        Decoder constructor, creates the decoding graph

        Args:
            conf: the decoder config
            classifier: the classifier that will be used for decoding
            classifier_scope: the scope where the classier should be
                created/loaded from
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximum length of the inputs
            batch_size: the decoder batch size
            coder: a TargetCoder object
            expdir: the location where the models were saved and the results
                will be written
        '''

        self.conf = conf
        self.max_input_length = max_input_length
        self.expdir = expdir
        self.coder = coder
        self.batch_size = int(conf['batch_size'])

        with tf.variable_scope('BeamSearchDecoder'):

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name='seq_length')

            #encode the inputs [batch_size x output_length x output_dim]
            with tf.variable_scope(classifier_scope):
                hlfeat = classifier.encoder(self.inputs, self.input_seq_length,
                                            False, False)

            #repeat the high level features for all beam elements
            hlfeat = tf.reshape(tf.tile(tf.expand_dims(hlfeat, 1),
                                        [1, int(conf['beam_width']), 1, 1]),
                                [int(conf['beam_width'])*self.batch_size,
                                 int(hlfeat.get_shape()[1]),
                                 int(hlfeat.get_shape()[2])])


            def body(step, beam, reuse=True, initial_state_attention=True):
                '''the body of the decoding while loop

                Args:
                    beam: a Beam object containing the current beam
                    reuse: set to True to reuse the classifier
                    initial_state_attention: whether attention has to be applied
                        to the initital state to ge an initial context

                returns:
                    the loop vars'''

                with tf.variable_scope('body'):

                    #put the last output in the correct format
                    # [batch_size x beam_width]
                    prev_output = beam.sequences[:, :, step]

                    #put the prev_output and state in the correct shape so all
                    #beam elements from all batches are processed in parallel
                    #[batch_size*beam_width x 1]
                    prev_output = tf.expand_dims(
                        tf.reshape(prev_output, [-1]), 1)

                    states = [tf.reshape(s, [-1, int(s.get_shape()[2])])
                              for s in nest.flatten(beam.states)]
                    states = nest.pack_sequence_as(beam.states, states)

                    #compute the next state and logits
                    with tf.variable_scope(classifier_scope) as scope:
                        if reuse:
                            scope.reuse_variables()
                        logits, states = classifier.decoder(
                            hlfeat=hlfeat,
                            encoder_inputs=prev_output,
                            numlabels=classifier.output_dim,
                            initial_state=states,
                            initial_state_attention=initial_state_attention,
                            is_training=False)

                    #put the states and logits in the format for the beam
                    states = [tf.reshape(s,
                                         [self.batch_size,
                                          int(conf['beam_width']),
                                          int(s.get_shape()[1])])
                              for s in nest.flatten(states)]
                    states = nest.pack_sequence_as(beam.states, states)
                    logits = tf.reshape(logits,
                                        [self.batch_size,
                                         int(conf['beam_width']),
                                         int(logits.get_shape()[2])])

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
                        tf.logical_not(beam.all_terminated(step)),
                        tf.less(step, int(conf['max_steps'])))

                return cont


            #initialise the loop variables
            negmax = tf.tile(
                [[-tf.float32.max]],
                [self.batch_size, int(conf['beam_width'])-1])
            scores = tf.concat(
                1, [tf.zeros([self.batch_size, 1]), negmax])
            lengths = tf.ones(
                [self.batch_size, int(conf['beam_width'])],
                dtype=tf.int32)
            sequences = tf.ones(
                [self.batch_size, int(conf['beam_width']),
                 int(conf['max_steps'])],
                dtype=tf.int32)
            states = classifier.decoder.zero_state(
                int(conf['beam_width'])*self.batch_size)
            flat_states = [tf.reshape(s,
                                      [self.batch_size,
                                       int(conf['beam_width']),
                                       int(s.get_shape()[1])])
                           for s in nest.flatten(states)]
            states = nest.pack_sequence_as(states, flat_states)

            beam = Beam(sequences, lengths, states, scores)
            step = tf.constant(0)

            #do the first step because the initial state should not be used
            #to compute a context and reuse should be False
            step, beam = body(step, beam, False, False)

            #run the rest of the decoding loop
            _, beam = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, beam],
                parallel_iterations=1,
                back_prop=False)

            with tf.variable_scope('cut_sequences'):
                #get the beam scores
                scores = [tf.unpack(s) for s in tf.unpack(beam.scores)]

                #cut the beam sequences to the correct length and take of
                #the start of sequence token
                sequences = [tf.unpack(s) for s in tf.unpack(beam.sequences)]
                lengths = [tf.unpack(l) for l in tf.unpack(beam.lengths)]
                sequences = [[sequences[i][j][1:lengths[i][j]]
                              for j in range(len(lengths[i]))]
                             for i in range(len(lengths))]

            self.outputs = [[(scores[i][j], sequences[i][j])
                             for j in range(len(sequences[i]))]
                            for i in range(len(sequences))]

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)


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
            batch_size = self.batch_size

            #get flags for finished beam elements: [batch_size x beam_width]
            finished = tf.equal(self.sequences[:, :, step], 0)

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
                scores = tf.select(full_finished, oldscores, newscores)

                #set the scores of expanded beams from finished elements to
                #negative maximum [batch_size x beam_width*numlabels]
                expanded_finished = tf.reshape(tf.concat(
                    2, [tf.tile([[[False]]], [batch_size, beam_width, 1]),
                        tf.tile(tf.expand_dims(finished, 2),
                                [1, 1, numlabels-1])])
                                               , [batch_size, -1])

                scores = tf.select(
                    expanded_finished,
                    tf.tile([[-scores.dtype.max]],
                            [batch_size, numlabels*beam_width]),
                    scores)


            with tf.variable_scope('lengths'):
                #update the sequence lengths for the unfinshed beam elements
                #[batch_size x beam_width]
                lengths = tf.select(finished, self.lengths, self.lengths+1)

                #repeat the lengths for all expanded elements
                #[batch_size x beam_width*numlabels]
                lengths = tf.reshape(
                    tf.tile(tf.expand_dims(lengths, 2), [1, 1, numlabels]),
                    [batch_size, -1])



            with tf.variable_scope('prune'):

                #select the best beam elements according to normalised scores
                float_lengths = tf.cast(lengths, tf.float32)
                _, indices = tf.nn.top_k(scores/float_lengths, beam_width,
                                         sorted=False)

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
                labels = tf.select(
                    finished,
                    tf.tile([[0]], [batch_size, beam_width]),
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

                #seperate the real sequence from the adding
                real = sequences[:, :, :step+1]
                padding = sequences[:, :, step+2:]

                #construct the new sequences by adding the selected labels
                labels = tf.expand_dims(labels, 2)
                sequences = tf.concat(2, [real, labels, padding])

                #specify the shape of the sequences
                sequences.set_shape([batch_size, beam_width, max_steps])

            return Beam(sequences, lengths, states, scores)

    def all_terminated(self, step):
        '''check if all elements in the beam have terminated
        Args:
            step: the current step

        Returns:
            a bool tensor'''
        with tf.variable_scope('all_terminated'):
            return tf.equal(tf.reduce_sum(self.sequences[:, :, step]), 0)

    @property
    def beam_width(self):
        '''get the size of the beam'''
        return int(self.lengths.get_shape()[1])

    @property
    def batch_size(self):
        '''get the size of the beam'''
        return int(self.lengths.get_shape()[0])
