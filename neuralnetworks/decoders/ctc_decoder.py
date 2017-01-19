'''@file ctc_decoder.py
contains the CTCDecoder'''

import tensorflow as tf
import decoder
import processing

class CTCDecoder(decoder.Decoder):
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
