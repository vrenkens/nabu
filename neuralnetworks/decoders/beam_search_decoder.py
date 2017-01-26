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
            coder: a TargetCoder object
            expdir: the location where the models were saved and the results
                will be written
        '''

        self.conf = conf
        self.max_input_length = max_input_length
        self.expdir = expdir
        self.coder = coder

        with tf.variable_scope('BeamSearchDecoder'):

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[1, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[1], name='seq_length')

            #encode the inputs
            with tf.variable_scope(classifier_scope):
                hlfeat = classifier.encoder(self.inputs, self.input_seq_length,
                                            False, False)

            #repeat the high level features for all beam elements
            hlfeat = tf.tile(hlfeat, [int(conf['beam_width']), 1, 1])


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
                    prev_output = tf.expand_dims(beam.sequences[:, step], 1)

                    #get a callable for the output logits
                    with tf.variable_scope(classifier_scope) as scope:
                        if reuse:
                            scope.reuse_variables()
                        logits, state = classifier.decoder(
                            hlfeat=hlfeat,
                            encoder_inputs=prev_output,
                            numlabels=classifier.output_dim,
                            initial_state=beam.states,
                            initial_state_attention=initial_state_attention,
                            is_training=False)
                        logits = tf.reshape(
                            logits,
                            [int(conf['beam_width']), classifier.output_dim])

                    #update the beam
                    beam = beam.update(logits, state, step)

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
            negmax = tf.tile([-tf.float32.max], [int(conf['beam_width'])-1])
            scores = tf.concat(0, [tf.zeros([1]), negmax])
            lengths = tf.ones([int(conf['beam_width'])], dtype=tf.int32)
            sequences = tf.ones(
                [int(conf['beam_width']), int(conf['max_steps'])],
                dtype=tf.int32)
            states = classifier.decoder.zero_state(int(conf['beam_width']))
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
                scores = tf.unpack(beam.scores)

                #cut the beam sequences
                sequences = tf.unpack(beam.sequences)
                lengths = tf.unpack(beam.lengths)
                sequences = [sequences[i][:lengths[i]]
                             for i in range(len(lengths))]

            self.outputs = [scores, sequences]

    def process_decoded(self, decoded):
        '''
        create numpy arrays of decoded targets

        Args:
            decoded: the most likely label sequence

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

        return [(decoded[0][i], decoded[1][i][1:])
                for i in range(len(decoded[0]))]

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
        - sequences: the sequences as a [size x max_steps] tensor
        - lengths: the length of the sequences as a [size] tensor
        - states: the state of the decoder as a possibly nested tuple of
            [size x state_dim] tensors
        - scores: the score of the beam element as a [size] tensor
        '''

    def update(self, logits, states, step):
        '''update the beam by expanding the beam with new hypothesis
        and selecting the best ones. Use as beam = beam.update(...)

        Args:
            logits: the decoder output logits as a [size x numlabels] tensor
            states: the decoder output states as a possibly nested tuple of
                [size x state_dim] tensor
            step: the current step

        Returns:
            a new updated Beam as a Beam object'''

        with tf.variable_scope('update'):

            numlabels = int(logits.get_shape()[1])
            max_steps = int(self.sequences.get_shape()[1])
            beam_width = self.size

            #get flags for finished beam elements
            finished = tf.equal(self.sequences[:, step], 0)

            with tf.variable_scope('scores'):

                #compute the log probabilities and vectorise
                logprobs = tf.reshape(tf.log(tf.nn.softmax(logits)), [-1])

                #put the old scores in the same format as the logrobs
                oldscores = tf.reshape(tf.tile(tf.expand_dims(self.scores, 1),
                                               [1, numlabels]),
                                       [-1])

                #compute the new scores
                newscores = oldscores + logprobs

                #only update the scores of the unfinished beam elements
                full_finished = tf.reshape(
                    tf.tile(tf.expand_dims(finished, 1), [1, numlabels]),
                    [-1])
                scores = tf.select(full_finished, oldscores, newscores)

                #set the scores of expanded beams from finished elements to
                #negative maximum
                expanded_finished = tf.reshape(tf.concat(
                    1, [tf.tile([[False]], [beam_width, 1]),
                        tf.tile(tf.expand_dims(finished, 1), [1, numlabels-1])])
                                               , [-1])

                scores = tf.select(
                    expanded_finished,
                    tf.tile([-scores.dtype.max], [numlabels*beam_width]),
                    scores)


            with tf.variable_scope('lengths'):
                #update the sequence lengths for the unfinshed beam elements
                lengths = tf.select(finished, self.lengths, self.lengths+1)

                #repeat the lengths for all expanded elements
                lengths = tf.reshape(
                    tf.tile(tf.expand_dims(lengths, 1), [1, numlabels]), [-1])



            with tf.variable_scope('prune'):

                #select the best beam elements according to normalised scores
                float_lengths = tf.cast(lengths, tf.float32)
                _, indices = tf.nn.top_k(scores/float_lengths, beam_width,
                                         sorted=False)

                #from the indices, compute the expanded beam elements and the
                #selected labels
                expanded_elements = tf.floordiv(indices, numlabels)
                labels = tf.mod(indices, numlabels)

                #put the selected label for the finished elements to zero
                finished = tf.gather(finished, expanded_elements)
                labels = tf.select(finished, tf.tile([0], [beam_width]), labels)

                #select the best lengths and scores
                lengths = tf.gather(lengths, indices)
                scores = tf.gather(scores, indices)

            #get the states for the expanded beam elements
            with tf.variable_scope('states'):
                #flatten the states
                flat_states = nest.flatten(states)

                #select the states
                flat_states = [tf.gather(s, expanded_elements)
                               for s in flat_states]

                #pack them in the original format
                states = nest.pack_sequence_as(states, flat_states)

            with tf.variable_scope('sequences'):
                #get the sequences for the expanded elements
                sequences = tf.gather(self.sequences, expanded_elements)

                #seperate the real sequence from the adding
                real = sequences[:, :step+1]
                padding = sequences[:, step+2:]

                #construct the new sequences by adding the selected labels
                labels = tf.expand_dims(labels, 1)
                sequences = tf.concat(1, [real, labels, padding])

                #specify the shape of the sequences
                sequences.set_shape([beam_width, max_steps])

            return Beam(sequences, lengths, states, scores)

    def all_terminated(self, step):
        '''check if all elements in the beam have terminated
        Args:
            step: the current step

        Returns:
            a bool tensor'''
        with tf.variable_scope('all_terminated'):
            return tf.equal(tf.reduce_sum(self.sequences[:, step]), 0)

    @property
    def size(self):
        '''get the size of the beam'''
        return int(self.lengths.get_shape()[0])
