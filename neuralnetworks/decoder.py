'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
import processing

def decoder_factory(conf,
                    classifier,
                    classifier_scope,
                    input_dim,
                    max_input_length,
                    coder,
                    expdir,
                    decoder_type):
    '''
    creates a decoder object

    Args:
        conf: the decoder config
        classifier: the classifier that will be used for decoding
        classifier_scope: the scope where the classier should be created/loaded
            from
        input_dim: the input dimension to the nnnetgraph
        max_input_length: the maximum length of the inputs
        coder: a TargetCoder object
        expdir: the location where the models were saved and the results
            will be written
        decoder_type: the decoder type
    '''

    if decoder_type == 'ctcdecoder':
        decoder_class = CTCDecoder
    elif decoder_type == 'greedydecoder':
        decoder_class = GreedyDecoder
    else:
        raise Exception('Undefined decoder type: %s' % decoder_type)

    return decoder_class(conf,
                         classifier,
                         classifier_scope,
                         input_dim,
                         max_input_length,
                         coder,
                         expdir)

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_decoded(self, decoded):
        '''
        do some postprocessing on the output of the decoding graph

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

    @abstractmethod
    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''


class CTCDecoder(Decoder):
    '''CTC Decoder'''

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

        with tf.variable_scope('ctcdecoder'):

            self.conf = conf
            self.max_input_length = max_input_length
            self.expdir = expdir
            self.coder = coder


            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[1, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[1], name='seq_length')

            #create the decoding graph
            logits, logits_seq_length =\
                classifier(
                    self.inputs, self.input_seq_length, targets=None,
                    target_seq_length=None, is_training=False,
                    scope=classifier_scope)

            #Convert logits to time major
            logits = tf.pack(tf.unpack(logits, axis=1))

            #do the CTC beam search
            sparse_outputs, logprobs = tf.nn.ctc_greedy_decoder(
                tf.pack(logits), logits_seq_length)

            #convert the outputs to dense tensors
            dense_outputs = [
                tf.reshape(tf.sparse_tensor_to_dense(o), [-1])
                for o in sparse_outputs]

            self.outputs = dense_outputs + [tf.reshape(logprobs, [-1])]

    def process_decoded(self, decoded):
        '''
        create numpy arrays of decoded targets

        Args:
            decoded: a tupple of length beam_width + 1 where the first
                beam_width elements are vectors with label sequences and the
                last elements is a beam_width length vector containing scores

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

        target_sequences = decoded[:-1]
        logprobs = decoded[-1]

        processed = [(logprobs[b], target_sequences[b])
                     for b in range(len(target_sequences))]

        return processed

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)

class GreedyDecoder(Decoder):
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

        with tf.variable_scope('greedyDecoder'):

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

            def body(step, state, outputs, score, reuse=True):
                '''the body of the decoding while loop

                Args:
                    step: the decoding step
                    state: the current decoding state
                    outputs: the current decodin outputs
                    score: the score of the decoded sequence
                    reuse: if set to True, the variables in the classifier
                        will be reused

                returns:
                    the loop vars'''

                with tf.variable_scope('body') as scope:

                    #put the last output in the correct format
                    prev_output = tf.expand_dims(tf.expand_dims(
                        outputs[-1], 0), 0)

                    #get the output logits
                    with tf.variable_scope(classifier_scope) as scope:
                        if reuse:
                            scope.reuse_variables()
                        logits, state = classifier.decoder(
                            hlfeat=hlfeat,
                            targets=prev_output,
                            numlabels=classifier.output_dim,
                            initial_state=state,
                            is_training=False)
                        logits = tf.squeeze(logits)

                    #compute the log probabilities
                    logprobs = tf.log(tf.nn.softmax(logits))

                    #get the most likely output label
                    output = tf.cast(tf.argmax(logprobs, axis=0), tf.int32)

                    #append the outputs to the outputs
                    outputs = tf.concat(0, [outputs, [output]])

                    #update the score
                    score = score + tf.reduce_max(logprobs)

                    step = step + 1

                return step, state, outputs, score

            def cb_cond(step, state, outputs, s):
                '''the condition of the decoding while loop

                Args:
                    step: the decoding step
                    state: the current decoding state
                    outputs: the current decodin outputs
                    s: the score of the decoded sequence

                returns:
                    a boolean that evaluates to True if the loop should
                    continue'''

                return tf.logical_and(tf.logical_not(tf.equal(outputs[-1], 0)),
                                      tf.less(step, int(conf['max_steps'])))



            #initialise the loop variables
            score = tf.constant(0, dtype=tf.float32)
            step = tf.constant(0)
            outputs = tf.constant([1])
            state = None

            #do the first step to allow None state
            step, state, outputs, score = body(step, state, outputs, score,
                                               False)

            #make sure the variables are reused in the next

            state_shape = [st.get_shape() for st in nest.flatten(state)]
            state_shape = nest.pack_sequence_as(state, state_shape)

            #run the rest of the decoding loop
            _, _, outputs, score = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, state, outputs, score],
                shape_invariants=[step.get_shape(), state_shape,
                                  tf.TensorShape([None]), score.get_shape()],
                parallel_iterations=1,
                back_prop=False)

            self.outputs = [score, outputs]

    def process_decoded(self, decoded):
        '''
        take of the start of sequence token

        Args:
            decoded: the most likely label sequence

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

        return [(decoded[0], decoded[1][1:])]

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)

class BeamSearchDecoder(Decoder):
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


            def body(step, beam):
                '''the body of the decoding while loop

                Args:
                    step: the decoding step
                    beam: the current beam as a list of tupples containing:
                        - the decoded sequence
                        - the score of the sequence
                        - the state of the decoder

                returns:
                    the loop vars'''

                with tf.variable_scope('body') as scope:

                    #start with an empty
                    expanded_beam = []

                    for b in beam:

                        #put the last output in the correct format
                        prev_output = tf.expand_dims(tf.expand_dims(
                            b[0][-1], 0), 0)

                        #get the output logits
                        with tf.variable_scope(classifier_scope) as scope:
                            scope.reuse_variables()
                            logits, state = classifier.decoder(
                                hlfeat=hlfeat,
                                targets=prev_output,
                                numlabels=classifier.output_dim,
                                initial_state=b[2],
                                is_training=False)
                            logits = tf.squeeze(logits)

                        expand = expand_beam(b[0], logits, b[1], state, step)
                        dummy = dummy_beam(b, classifier.output_dim)

                        expanded_beam += tf.cond(
                            b[0][-1] == 0,
                            lambda e=dummy: e,
                            lambda e=expand: e
                        )

                    beam = prune_beam(expanded_beam, int(conf['beam_width']))

                    step = step + 1

                return step, beam

            def cb_cond(step, state, outputs, s):
                '''the condition of the decoding while loop

                Args:
                    step: the decoding step
                    beam: the current beam as a list of tupples containing:
                        - the decoded sequence
                        - the score of the sequence
                        - the state of the decoder

                returns:
                    a boolean that evaluates to True if the loop should
                    continue'''

                #check if all beam elements have terminated
                last_labels = tf.pack([b[0][-1] for b in beam])

                #of the sum of all last differs from zero at least one elements
                #not finished
                all_finished = tf.equal(last_labels.reduce_sum(), 0)

                return tf.logical_and(tf.logical_not(all_finished),
                                      tf.less(step, int(conf['max_steps'])))



            #initialise the loop variables
            score = tf.constant(0, dtype=tf.float32)
            step = tf.constant(0)
            outputs = tf.constant([1])
            state = classifier.decoder.zero_state(1)
            state_shape = [st.get_shape() for st in nest.flatten(state)]
            state_shape = nest.pack_sequence_as(state, state_shape)
            first_element = (outputs, score, None)
            beam = dummy_beam(first_element, int(conf['beam_width']))

            #do the first step to allow None state
            step, beam = body(step, beam)

            #create the shape invariants
            beam_invariants = (tf.TensorShape([None]), score.get_shape(),
                               state_shape)

            shape_invariants = [step.get_shape(), beam_invariants]

            #run the rest of the decoding loop
            _, beam = tf.while_loop(
                cond=cb_cond,
                body=body,
                loop_vars=[step, beam],
                shape_invariants=shape_invariants,
                parallel_iterations=1,
                back_prop=False)

            #get the outputs and scores from the beam
            self.outputs = [(b[1], b[0]) for b in beam]

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

        return decoded

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)

def decode(decoder, reader, sess):
    '''decode using the neural net

    Args:
        decoder: the decoder that should be used
        reader: a feauture reader object containing the testng features
        sess: a tensorflow session

    Returns:
        a dictionary containing the outputs
    '''

    #start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    decoded = dict()

    while True:

        (utt_id, inputs, looped) = reader.get_utt()

        if looped:
            reader.prev_id()
            break

        #get the sequence length
        input_seq_length = [inputs.shape[0]]

        #pad the inputs
        inputs = np.append(
            inputs, np.zeros([decoder.max_input_length-inputs.shape[0],
                              inputs.shape[1]]), 0)

        #pylint: disable=E1101
        output = sess.run(
            decoder.outputs,
            feed_dict={decoder.inputs:inputs[np.newaxis, :, :],
                       decoder.input_seq_length:input_seq_length})

        processed = decoder.process_decoded(output)

        #convert the label sequence into a sequence of characers
        decoded[utt_id] = [(p[0], decoder.coder.decode(p[1]))
                           for p in processed]


    return decoded

def expand_beam(sequence, logits, score, state, step):
    '''expand the beam with all options

    Args:
        sequence: the sequence of the expanded element of the beam
        logits: the output logits of the classifier as a numlabels dimensional
            vector
        score: the score of the expanded element of the beam
        state: the output state of the classifier
        step: the current step in the decoding process

    returns:
        the expanded beam as a list of tupples containing:
            - the decoded sequence
            - the score of the sequence
            - the state of the decoder
    '''

    with tf.variable_scope('expand_beam'):
        #the number of possible labels
        numlabels = int(logits.get_shape()[0])

        #put the logits through a softmax and a log to get the logprobs
        logprobs = tf.log(tf.nn.softmax(logits))

        #get a list of scores for the expanded beam
        scores = tf.unpack((score*(step-1) + logprobs)/step)

        #update the sequences
        tiled_sequence = tf.tile(tf.expand_dims(sequence, 1), [1, numlabels])
        new_labels = tf.expand_dims(tf.range(numlabels), 0)
        sequences = tf.unpack(tf.concat(0, [tiled_sequence, new_labels]))

        #the state is the same for all elements in the beam
        states = [state]*numlabels

    return zip(sequences, scores, states)

def dummy_beam(beam_element, numlabels):
    '''create a beam with only one real elements

    Args:
        beam_element: the real beam elements
        numlabels: the number of output labels
    '''
    with tf.variable_scope('dummy_beam'):
        dummy_element = (beam_element[0], -beam_element[1].dtype.max,
                         beam_element[2])
    return [beam_element] + [dummy_element]*(numlabels-1)

def prune_beam(beam, beam_width):
    '''prune the beam

    Args:
        beam: the beam as a list of tupples containing:
            - the decoded sequence
            - the score of the sequence
            - the state of the decoder
        beam_width: the beam width

    returns:
        the pruned beam as a list of length beam_width
    '''

    with tf.variable_scope('prune_beam'):
        #create tensorarrays for the fields in the beam elements
        arrays = []
        for i, _ in enumerate(beam[0]):
            array = tf.TensorArray(dtype=beam[0][i].dtype,
                                   size=len(beam))
            for b, e in enumerate(beam):
                array.write(b, e[i])
            arrays.append(array)

        #get the best indices in the beam
        _, indices = tf.nn.top_k(arrays[1].stack, beam_width)
        indices = tf.unpack(indices)

        #get the best indices from the arrays and put them in the beam
        pruned_beam = []
        for index in indices:
            pruned_beam.append((array.read(index) for array in arrays))

    return pruned_beam
