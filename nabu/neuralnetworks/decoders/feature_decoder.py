'''@file feature_decoder.py
contains the MaxDecoder'''

import os
import numpy as np
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components import ops

class FeatureDecoder(decoder.Decoder):
    '''feature Decoder'''

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

        with tf.name_scope('feature_decoder'):

            #create the decoding graph
            logits, logits_seq_length = self.model(
                inputs, input_seq_length, targets=[],
                target_seq_length=[], is_training=False)

            outputs = {o:(logits[o], logits_seq_length[o]) for o in logits}

        return outputs

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''

        for o in outputs:
            if not os.path.isdir(os.path.join(directory, o)):
                os.makedirs(os.path.join(directory, o))
            batch_size = outputs[o][0].shape[0]
            for i in range(batch_size):
                output = outputs[o][0][i, :outputs[o][1][i]]
                np.save(os.path.join(directory, o, names[i]+'.npy'), output)

    def update_evaluation_loss(self, loss, outputs, references,
                               reference_seq_length):
        '''update the evaluation loss

        args:
            loss: the current evaluation loss
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            an op to update the evalution loss
        '''

        raise Exception('FeatureDecoder can only be used to decode')
