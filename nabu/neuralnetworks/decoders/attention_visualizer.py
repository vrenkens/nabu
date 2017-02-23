'''@file beam_search_decoder.py
contains the BeamSearchDecoder'''

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
import decoder
from nabu.processing import score

class AttentionVisiualizer(decoder.Decoder):
    '''Beam search decoder for attention visualization'''

    def get_outputs(self, inputs, input_seq_length, classifier):

        '''compute the outputs of the decoder

        Args:
            inputs: The inputs to the network as a
                [batch_size x max_input_length x input_dim] tensor
            input_seq_length: The sequence length of the inputs as a
                [batch_size] vector
            classifier: The classifier object that will be used in decoding

        Returns:
            A list with batch_size elements containing nbest lists with elements
            containing pairs of score and output labels
        '''

        #encode the inputs [batch_size x output_length x output_dim]
        hlfeat = classifier.encoder(self.inputs, self.input_seq_length, False)

        #repeat the high level features for all beam elements
        hlfeat = tf.reshape(tf.tile(tf.expand_dims(hlfeat, 1),
                                    [1, int(self.conf['beam_width']), 1, 1]),
                            [int(self.conf['beam_width'])*self.batch_size,
                             int(hlfeat.get_shape()[1]),
                             int(hlfeat.get_shape()[2])])


        def body(step, beam, first_step=False, check_finished=True):
            '''the body of the decoding while loop

            Args:
                beam: a Beam object containing the current beam
                first_step: whether or not this is the first step in decoding
                check_finished: finish a beam element if a sentence border
                    token is observed

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
                logits, states = classifier.decoder(
                    hlfeat=hlfeat,
                    encoder_inputs=prev_output,
                    initial_state=states,
                    first_step=first_step,
                    is_training=False)

                #get the attenion tensor
                if first_step:
                    attention_name = (
                        tf.get_default_graph()._name_stack
                        + '/' + type(classifier.decoder).__name__ +
                        '/attention_decoder/Attention_0/Softmax:0')
                else:
                    attention_name = (
                        tf.get_default_graph()._name_stack
                        + '/' + type(classifier.decoder).__name__
                        + '/attention_decoder/attention_decoder/' +
                        'Attention_0/Softmax:0')

                attention = tf.get_default_graph().get_tensor_by_name(
                    attention_name)

                #put the states and logits in the format for the beam
                states = [tf.reshape(s,
                                     [self.batch_size,
                                      int(self.conf['beam_width']),
                                      int(s.get_shape()[1])])
                          for s in nest.flatten(states)]
                states = nest.pack_sequence_as(beam.states, states)
                logits = tf.reshape(logits,
                                    [self.batch_size,
                                     int(self.conf['beam_width']),
                                     int(logits.get_shape()[2])])

                attention = tf.reshape(attention,
                                       [self.batch_size,
                                        int(self.conf['beam_width']),
                                        int(attention.get_shape()[1])])

                #update the beam
                beam = beam.update(logits, states, attention, step,
                                   check_finished)

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
                        step, classifier.output_dim - 1)),
                    tf.less(step, int(self.conf['max_steps'])))

            return cont


        #initialise the loop variables
        negmax = tf.tile(
            [[-tf.float32.max]],
            [self.batch_size, int(self.conf['beam_width'])-1])
        scores = tf.concat(
            1, [tf.zeros([self.batch_size, 1]), negmax])
        lengths = tf.ones(
            [self.batch_size, int(self.conf['beam_width'])],
            dtype=tf.int32)
        sequences = tf.constant(
            classifier.output_dim-1,
            shape=[self.batch_size, int(self.conf['beam_width']),
                   int(self.conf['max_steps'])],
            dtype=tf.int32)
        states = classifier.decoder.zero_state(
            int(self.conf['beam_width'])*self.batch_size)
        flat_states = [tf.reshape(s,
                                  [self.batch_size,
                                   int(self.conf['beam_width']),
                                   int(s.get_shape()[1])])
                       for s in nest.flatten(states)]
        states = nest.pack_sequence_as(states, flat_states)
        attention = tf.zeros([self.batch_size, int(self.conf['beam_width']),
                              int(hlfeat.get_shape()[1]),
                              int(self.conf['max_steps'])])

        beam = Beam(sequences, lengths, states, scores, attention)
        step = tf.constant(0)

        #do the first step because the initial state should not be used
        #to compute a context
        step, beam = body(step, beam, True, False)

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
            #the sequence border tokens
            sequences = [tf.unpack(s) for s in tf.unpack(beam.sequences)]
            lengths = [tf.unpack(l) for l in tf.unpack(beam.lengths)]
            attention = [tf.unpack(a) for a in tf.unpack(beam.attention)]
            hlfeat = tf.unpack(hlfeat)
            sequences = [[sequences[i][j][1:lengths[i][j]-1]
                          for j in range(len(lengths[i]))]
                         for i in range(len(lengths))]
            attention = [[attention[i][j][:, 1:lengths[i][j]]
                          for j in range(len(lengths[i]))]
                         for i in range(len(lengths))]

        outputs = [[(scores[i][j], sequences[i][j], attention[i][j], hlfeat[i])
                    for j in range(len(sequences[i]))]
                   for i in range(len(sequences))]

        return outputs

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''
        return score.cer(outputs, targets)


class Beam(namedtuple('Beam', ['sequences', 'lengths', 'states', 'scores',
                               'attention'])):
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

    def update(self, logits, states, attention, step, check_finished):
        '''update the beam by expanding the beam with new hypothesis
        and selecting the best ones. Use as beam = beam.update(...)

        Args:
            logits: the decoder output logits as a
                [batch_size x beam_width x numlabels] tensor
            states: the decoder output states as a possibly nested tuple of
                [batch_size x beam_width x state_dim] tensor
            attention: the attention for this decoding step as a
                [batch_size x beam_width x input_length] tensor
            step: the current step
            check_finished: finish a beam element if a sentence border
                token is observed

        Returns:
            a new updated Beam as a Beam object'''

        with tf.variable_scope('update'):

            numlabels = int(logits.get_shape()[2])
            max_steps = int(self.sequences.get_shape()[2])
            beam_width = self.beam_width
            batch_size = self.batch_size
            input_length = int(self.attention.get_shape()[2])

            #get flags for finished beam elements: [batch_size x beam_width]
            if check_finished:
                finished = tf.equal(self.sequences[:, :, step], numlabels-1)
            else:
                finished = tf.constant(False, dtype=tf.bool,
                                       shape=[batch_size, beam_width])

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

                #seperate the real sequence from the adding
                real = sequences[:, :, :step+1]
                padding = sequences[:, :, step+2:]

                #construct the new sequences by adding the selected labels
                labels = tf.expand_dims(labels, 2)
                sequences = tf.concat(2, [real, labels, padding])

                #specify the shape of the sequences
                sequences.set_shape([batch_size, beam_width, max_steps])

            with tf.variable_scope('attention'):
                #get the attentions for the expanded elements
                new_attention = tf.gather(
                    tf.reshape(self.attention, [-1, input_length, max_steps]),
                    expanded_elements)

                #seperate the real attention from the adding
                real = new_attention[:, :, :, :step+1]
                padding = new_attention[:, :, :, step+2:]

                #construct the new attention
                attention = tf.expand_dims(attention, 3)
                new_attention = tf.concat(3, [real, attention, padding])

                #specify the shape
                new_attention.set_shape([batch_size, beam_width, input_length,
                                         max_steps])

            return Beam(sequences, lengths, states, scores, new_attention)

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
        return int(self.lengths.get_shape()[0])
