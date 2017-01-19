'''@file beam_search_decoder.py
contains the BeamSearchDecoder'''

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
import decoder
import processing

import pdb

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
                hlfeat = classifier.encoder(self.inputs, [max_input_length],
                                            False, False)


            def body(step, beam, reuse=True):
                '''the body of the decoding while loop

                Args:
                    beam: a Beam object containing the current beam
                    reuse: set to True to reuse the classifier

                returns:
                    the loop vars'''

                with tf.variable_scope('body') as scope:

                    #start with an empty
                    expanded_beam = Beam(beam.state, beam.max_steps)

                    for b in beam:

                        #put the last output in the correct format
                        prev_output = tf.expand_dims(tf.expand_dims(
                            b.sequence[b.length], 0), 0)

                        #get a callable for the output logits
                        with tf.variable_scope(classifier_scope) as scope:
                            if reuse:
                                scope.reuse_variables()
                            logits, state = classifier.decoder(
                                hlfeat=hlfeat,
                                targets=prev_output,
                                numlabels=classifier.output_dim,
                                initial_state=b.state,
                                is_training=False)
                            logits = tf.squeeze(logits)
                        cb_logits_state = lambda l=logits, s=state: (l, s)

                        #bool for dummy or expand
                        expand = tf.logical_or(
                            tf.equal(b.sequence[b.length], 0),
                            tf.equal(b.score, -b.score.dtype.max))

                        #compute the new beam
                        new_beam = tf.cond(
                            expand,
                            lambda e=b, ls=cb_logits_state: expand_beam(ls, e),
                            lambda e=b: dummy_expand(classifier.output_dim, e))

                        #expand the beam
                        expanded_beam += new_beam

                    #prune the beam
                    expanded_beam.prune(int(conf['beam_width']))

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

                #check if all beam elements have terminated
                expand = tf.pack([tf.logical_or(
                    tf.equal(b.sequence[b.length], 0),
                    tf.equal(b.score, -b.score.dtype.max)) for b in beam])

                finished = tf.equal(tf.reduce_sum(tf.cast(expand, tf.int32)), 0)

                return tf.logical_and(tf.logical_not(finished),
                                      tf.less(step, int(conf['max_steps'])))



            #initialise the loop variables
            score = tf.constant(0, dtype=tf.float32)
            step = tf.constant(0)
            outputs = tf.constant([1])
            state = None
            beam_element = BeamElement(outputs, step, state, score)
            beam = dummy_expand(classifier.output_dim, beam_element)

            #do the first step to allow None state
            step, beam = body(step, beam, False)

            #run the rest of the decoding loop
            _, beam = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, beam],
                parallel_iterations=1,
                back_prop=False)

            #get the beam scores
            scores = [b.score for b in beam]

            #cut the beam sequences
            sequences = [b.sequence[:b.length] for b in beam]

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

        return [(d[0][i], d[1][i]) for i in range(len(decoded[0]))]

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)

class BeamElement(namedtuple('BeamElement', ['sequence', 'length', 'state',
                                             'score'])):
    '''a named tuple class for elements in the beam

    the tuple fields are:
        - sequence: the sequence as a [max_steps] vector
        - length: the length of the sequence as a scalar
        - state: the state of the decoder as a possibly nested tuple of tensors
        - score: the score of the beam element
        '''

    @property
    def vector(self):
        '''the beam element as a vector

        returns:
            - a vector representing the beam element'''

        sequence_vector = tf.cast(self.sequence, tf.float32)
        length_vector = tf.cast(tf.expand_dims(self.length), tf.float32)
        score_vector = tf.expand_dims(self.score)
        state_vector = tf.concat(0, nest.flatten(self.state))

        return tf.concat(0, [sequence_vector, length_vector, state_vector,
                             score_vector])

class Beam(list):
    '''a beam that holds beam elements'''

    def __init__(self, state, max_steps):
        '''Beam constructor, creates an ampty beam

        Args:
            state: an example of a state in the beam, this should be a possibly
                nested list of [1xS] tensors where S is the length of the state
                element
            max_steps: the maximum number of decoding steps'''

        #super constructor
        super(Beam, self).__init__()

        #save the state example
        self.state = state

        #get the dimension of the state
        if state is not None:
            self.state_dim = sum([int(s.get_size()[1])
                                 for s in nest.flatten(state)])
        else:
            self.state_dim = 0

        self.max_steps = max_steps

    def write_tensor(self, tensor):
        '''write a tensor containing vectorised beam elements to the beam
        this overwrites the beam

        Args:
            - index: the index where the beam element should be written
            - vector: the vector representation of the beam element
        '''

        #clear the beam
        del self[:]

        #get the fields from the beam element out of the vector
        sequence_tensor = tf.cast(tensor[:, :self.max_steps], tf.int32)
        length_vector = tf.cast(tensor[:, self.max_steps], tf.int32)
        state_tensor = tensor[:, self.max_steps + 1:
                              self.max_steps + 1 + self.state_dim]
        score_vector = tensor[:, -1]

        #create all the beam elements and put them in the beam
        for i in range(int(tensor.get_shape()[0])):
            #get the beam element fields
            sequence = sequence_tensor[i]
            state = nest.pack_sequence_as(self.state, state_tensor[i])
            length = length_vector[i]
            score = score_vector[i]

            #create the beamelement
            beamElement = BeamElement(sequence, length, state, score)

            #put the beam element in the beam
            self[i] = beamElement

    @property
    def tensor(self):
        '''create a tensor version of the beam

        Returns:
            a tensor of size [beam_size, beam_element_size]'''

        return tf.pack([e.vector for e in self])

    def prune(self, beam_width):
        '''prune the beam to only retain the elements with the highest score

        Args:
            beam_width: the number of retained beam elements'''

        #put all the scores of the beam into a vector
        scores = tf.pack([e.score for e in self])

        #get the indices of the highest scores
        _, indices = tf.nn.top_k(scores, beam_width)

        #get the best beam_elements from the beam as a tensor
        beam_tensor = tf.gather(self.tensor, indices)

        #write the tensor to the beam
        self.write_tensor(beam_tensor)

def expand_beam(cb_logits_state, beam_element):
    '''expand the beam_element into a beam of possibilities

    Args:
        cb_logits_state: a callable that returns a pair containing the
            logits [1, 1, numlabels] and state when called
        beam_element: the beam element that will be expanded

    Returns:
        the expanded beam'''

    #get the logits and state
    logits, state = cb_logits_state()
    logits = tf.squeeze(logits)

    #get the number of labels
    numlabels = int(logits.get_shape()[0])

    #get the logprobs
    logprobs = tf.log(tf.nn.softmax(logits))

    #compute the scores
    float_step = tf.cast(beam_element.length, tf.float32)
    scores = tf.expand_dims(
        (beam_element.score*(float_step-1) + logprobs)/float_step, 1)

    #vectorize the states an repeat for all the beams
    states = tf.concat(0, nest.flatten(state))
    states = tf.expand_dims(states, 0)
    states = tf.tile(states, [numlabels, 1])

    #add one to the length and repeat for all beams
    lengths = tf.tile([[beam_element.length + 1]], [numlabels, 1])

    sequences = tf.tile(tf.expand_dims(beam_element.sequence, 0),
                        [numlabels, 1])
    sequences[:, beam_element.length+1] = tf.range(numlabels)
    beam_tensor = tf.concat(1, [sequences, lengths, states, scores])

    #create a beam
    max_steps = int(beam_element.sequence.get_shape()[0])
    beam = Beam(beam_element.state, max_steps)
    beam.write_tensor(beam_tensor)

    return beam

def dummy_expand(numlabels, beam_element):
    '''create a dummy beam that contains the input beam elements and for
    the rest it contains dummy elements to meet the constant size constraint
    of tf.cond

    Args:
        numlabels: the number of labels
        beam_element: the beam_element that should be expanded

    Returns:
        the expanded beam'''

    #create a new beam
    max_steps = int(beam_element.sequence.get_shape()[0])
    beam = Beam(beam_element.state, max_steps)

    #set the beam element as first element
    beam.append(beam_element)

    #create a dummy element
    dummy = BeamElement(beam_element.sequence, beam_element.length,
                        beam_element.state, -beam_element.score.dtype.max)

    beam += [dummy]*(numlabels-1)

    return beam
