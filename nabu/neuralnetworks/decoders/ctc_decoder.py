'''@file ctc_decoder.py
contains the CTCDecoder'''

import os
import numpy as np
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

class CTCDecoder(decoder.Decoder):
    '''CTC Decoder'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        self.alphabets = {}
        for o in model.output_names:
            #get the alphabet
            self.alphabets[o] = conf['%s_alphabet' % o].split(' ')

        super(CTCDecoder, self).__init__(conf, model)

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

        with tf.name_scope('ctc_decoder'):

            #create the decoding graph
            logits, logits_seq_length =\
                self.model(
                    inputs, input_seq_length, targets=[],
                    target_seq_length=[], is_training=False)

            outputs = dict()
            for o in logits:

                #Convert logits to time major
                logits[o] = tf.transpose(logits[o], [1, 0, 2])

                #do the CTC beam search
                out, _ = tf.nn.ctc_greedy_decoder(
                    logits[o], logits_seq_length[o])
                outputs[o] = tf.cast(out[0], tf.int32)

        return outputs

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''

        for o in outputs:
            batch_size = outputs[o].dense_shape[0]
            with open(os.path.join(directory, o), 'a') as fid:
                for i in range(batch_size):
                    indices = np.where(outputs[o].indices[:, 0] == i)[0]
                    output = outputs[o].values[indices]
                    text = ' '.join([self.alphabets[o][i] for i in output])
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

        #compute the edit distance for the decoded sequences
        #convert the representations to sparse Tensors
        sparse_targets = {
            o:dense_sequence_to_sparse(references[o], reference_seq_length[o])
            for o in references}

        #compute the edit distance
        losses = [
            tf.reduce_mean(tf.edit_distance(outputs[o],
                                            sparse_targets[o]))
            for o in outputs]

        loss = tf.reduce_mean(losses)

        return loss
