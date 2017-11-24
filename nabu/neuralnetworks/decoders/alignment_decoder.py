'''@file alignment_decoder.py
contains the AlignmentDecoder'''

import os
import struct
import numpy as np
import tensorflow as tf
import decoder

class AlignmentDecoder(decoder.Decoder):
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

        with tf.name_scope('alignment_decoder'):

            #create the decoding graph
            logits, logits_seq_length = self.model(
                inputs, input_seq_length, targets=[],
                target_seq_length=[], is_training=False)

            #compute the log probabilities
            logprobs = tf.log(tf.nn.softmax(logits.values()[0]))

            #read the pd prior
            prior = np.load(self.conf['prior'])

            #compute posterior to pseudo likelihood
            loglikes = logprobs - np.log(prior)

            outputs = {o:(loglikes, logits_seq_length[o]) for o in logits}

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
            scp_file = os.path.join(directory, o, 'feats.scp')
            ark_file = os.path.join(directory, o, 'loglikes.ark')
            for i in range(batch_size):
                name = '-'.join(names[i].split('-')[:-1])
                output = outputs[o][0][i, :outputs[o][1][i]]
                arkwrite(scp_file, ark_file, name, output)

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

        raise Exception('AlignmentDecoder can not be used to validate')

def arkwrite(scp_file, ark_file, name, array):
    '''write the array to the arkfile'''

    scp_fid = open(scp_file, 'a')
    ark_fid = open(ark_file, 'ab')
    rows, cols = array.shape
    ark_fid.write(struct.pack('<%ds'%(len(name)), name))
    pos = ark_fid.tell()
    ark_fid.write(struct.pack('<xcccc', 'B', 'F', 'M', ' '))
    ark_fid.write(struct.pack('<bi', 4, rows))
    ark_fid.write(struct.pack('<bi', 4, cols))
    ark_fid.write(array)
    scp_fid.write('%s %s:%s\n' % (name, ark_file, pos))
    scp_fid.close()
    ark_fid.close()
