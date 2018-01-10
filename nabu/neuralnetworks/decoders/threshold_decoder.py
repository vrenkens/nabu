'''@file threshold_decoder.py
contains the ThresholdDecoder'''

import os
import tensorflow as tf
import decoder

class ThresholdDecoder(decoder.Decoder):
    '''CTC Decoder'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        super(ThresholdDecoder, self).__init__(conf, model)

        self.threshold = float(self.conf['threshold'])

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a dictionary of [batch_size x time x ...]
                tensors
            input_seq_length: the input sequence lengths as a dictionary of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a dictionary of outputs
        '''

        with tf.name_scope('threshold_decoder'):

            #create the decoding graph
            logits, logits_seq_length =\
                self.model(
                    inputs, input_seq_length, targets=[],
                    target_seq_length=[], is_training=False)

            outputs = {}
            for out in logits:
                labels = tf.greater(
                    tf.nn.sigmoid(tf.squeeze(logits[out], [2])), 
                    self.threshold)
                outputs[out] = (labels, logits_seq_length[out])

        return outputs

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''

        for o in outputs:
            batch_size = outputs[o][0].shape[0]
            with open(os.path.join(directory, o), 'a') as fid:
                for i in range(batch_size):
                    output = outputs[o][0][i, :outputs[o][1][i]]
                    text = ' '.join(map(str, output))
                    fid.write('%s %s\n' % (names[i], text))

    def evaluate(self, outputs, references, reference_seq_length):
        '''evaluate the output of the decoder

        args:
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            the error of the outputs
        '''

        #compute the edit distance
        losses = []
        for o in outputs:
            numerrors = tf.reduce_sum(tf.cast(tf.logical_xor(
                outputs[o][0], references[o]), tf.float32))
            numlabels = tf.cast(
                tf.reduce_sum(reference_seq_length[o]), tf.float32)
            losses.append(numerrors/numlabels)

        loss = tf.reduce_mean(losses)

        return loss
