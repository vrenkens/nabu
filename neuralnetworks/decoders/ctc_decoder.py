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

        self.conf = conf
        self.max_input_length = max_input_length
        self.expdir = expdir
        self.coder = coder
        self.batch_size = int(conf['batch_size'])

        with tf.variable_scope('ctcdecoder'):

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name='seq_length')

            #create the decoding graph
            with tf.variable_scope(classifier_scope):
                logits, logits_seq_length =\
                    classifier(
                        self.inputs, self.input_seq_length, targets=None,
                        target_seq_length=None, is_training=False)

            #Convert logits to time major
            logits = tf.transpose(logits, [1, 0, 2])

            #do the CTC beam search
            sparse_outputs, logprobs = tf.nn.ctc_greedy_decoder(
                logits, logits_seq_length)
            sparse_outputs = sparse_outputs[0]
            logprobs = tf.unpack(tf.reshape(logprobs, [-1]))

            #split the sparse tensors into the seperate utterances
            output_list = tf.sparse_split(0, self.batch_size, sparse_outputs)
            outputs = [tf.sparse_tensor_to_dense(o) for o in output_list]

            self.outputs = [[(logprobs[i], outputs[i])]
                            for i in range(self.batch_size)]

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        return processing.score.cer(outputs, targets)
