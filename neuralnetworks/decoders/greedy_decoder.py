'''@file greedy_decoder.py
contains the GreedyDecoder'''

import tensorflow as tf
from tensorflow.python.util import nest
import decoder
import processing

class GreedyDecoder(decoder.Decoder):
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

            def body(step, state, outputs, score, reuse=True,
                     initial_state_attention=True):
                '''the body of the decoding while loop

                Args:
                    step: the decoding step
                    state: the current decoding state
                    outputs: the current decodin outputs
                    score: the score of the decoded sequence
                    reuse: if set to True, the variables in the classifier
                        will be reused
                    initial_state_attention: whether attention has to be applied
                        to the initital state to ge an initial context

                returns:
                    the loop vars'''

                with tf.variable_scope('body'):

                    #put the last output in the correct format
                    prev_output = tf.expand_dims(tf.expand_dims(
                        outputs[-1], 0), 0)

                    #get the output logits
                    with tf.variable_scope(classifier_scope) as scope:
                        if reuse:
                            scope.reuse_variables()
                        logits, state = classifier.decoder(
                            hlfeat=hlfeat,
                            encoder_inputs=prev_output,
                            numlabels=classifier.output_dim,
                            initial_state=state,
                            initial_state_attention=initial_state_attention,
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

                with tf.variable_scope('cond'):
                    return tf.logical_and(
                        tf.logical_not(tf.equal(outputs[-1], 0)),
                        tf.less(step, int(conf['max_steps'])))



            #initialise the loop variables
            score = tf.constant(0, dtype=tf.float32)
            step = tf.constant(0)
            outputs = tf.constant([1])
            state = classifier.decoder.zero_state(1)

            #do the first step because the initial state should not be used
            #to compute a context and reuse should be False
            step, state, outputs, score = body(step, state, outputs, score,
                                               False, False)

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
