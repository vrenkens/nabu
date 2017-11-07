'''@file max_decoder.py
contains the MaxDecoder'''

import os
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components import ops

class MaxDecoder(decoder.Decoder):
    '''max Decoder'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        super(MaxDecoder, self).__init__(conf, model)

        self.alphabets = {}
        for o in model.output_names:
            #get the alphabet
            self.alphabets[o] = self.conf['%s_alphabet' % o].split(' ')

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

        with tf.name_scope('max_decoder'):

            #create the decoding graph
            logits, logits_seq_length =\
                self.model(
                    inputs, input_seq_length, targets=[],
                    target_seq_length=[], is_training=False)

            outputs = {}
            for out in logits:
                sm = tf.nn.softmax(logits[out])
                outputs[out] = (
                    tf.cast(tf.argmax(sm, axis=2), tf.int32),
                    logits_seq_length[out])

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
                    text = ' '.join([self.alphabets[o][j] for j in output])
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

        #stack all the logits except the final logits
        stacked_outputs = {
            t:ops.seq2nonseq(outputs[t][0], outputs[t][1])
            for t in outputs}


        #create the stacked targets
        stacked_targets = {
            t:tf.cast(ops.seq2nonseq(references[t],
                                     reference_seq_length[t]), tf.int32)
            for t in references}

        #compute the edit distance
        losses = [
            tf.reduce_mean(tf.reduce_mean(tf.cast(tf.not_equal(
                stacked_outputs[o], stacked_targets[o]), tf.float32)))
            for o in outputs]

        loss = tf.reduce_mean(losses)

        return loss
