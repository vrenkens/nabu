'''@file ctctrainer.py
contains the CTCTrainer'''

import tensorflow as tf
import numpy as np
import trainer
from neuralnetworks import ops

class CTCTrainer(trainer.Trainer):
    '''A trainer that minimises the CTC loss, the output sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the CTC loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length, 1] tensor containing the
                targets
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('CTC_loss'):

            #get the batch size
            batch_size = int(targets.get_shape()[0])

            #convert the targets into a sparse tensor representation
            indices = tf.concat(0, [tf.concat(
                1, [tf.expand_dims(tf.tile([s], [target_seq_length[s]]), 1),
                    tf.expand_dims(tf.range(target_seq_length[s]), 1)])
                                    for s in range(batch_size)])

            values = tf.reshape(
                ops.seq2nonseq(targets, target_seq_length), [-1])

            shape = [batch_size, int(targets.get_shape()[1])]

            sparse_targets = tf.SparseTensor(tf.cast(indices, tf.int64), values,
                                             shape)

            loss = tf.reduce_mean(tf.nn.ctc_loss(logits, sparse_targets,
                                                 logit_seq_length,
                                                 time_major=False))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        decode the validation set with CTC beam search

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a matrix containing the decoded labels with size
            [batch_size, max_decoded_length]
        '''

        #Convert logits to time major
        tm_logits = tf.transpose(logits, [1, 0, 2])

        #do the CTC beam search
        sparse_output, _ = tf.nn.ctc_greedy_decoder(
            tf.pack(tm_logits), logit_seq_length)

        #convert the output to dense tensors with -1 as default values
        dense_output = tf.sparse_tensor_to_dense(sparse_output[0],
                                                 default_value=-1)

        return dense_output


    def validation_metric(self, outputs, targets):
        '''the Label Error Rate for the decoded labels

        Args:
            outputs: the validation output, which is a matrix containing the
                decoded labels of size [batch_size, max_decoded_length]. the
                output sequences are padded with -1
            targets: a list containing the ground truth target labels

        Returns:
            a numpy array containing the loss of all utterances
        '''

        #remove the padding from the outputs
        trimmed_outputs = [o[np.where(o != -1)] for o in outputs]

        ler = np.zeros(len(targets))

        for k, target in enumerate(targets):

            error_matrix = np.zeros([target.size + 1,
                                     trimmed_outputs[k].size + 1])

            error_matrix[:, 0] = np.arange(target.size + 1)
            error_matrix[0, :] = np.arange(trimmed_outputs[k].size + 1)

            for x in range(1, target.size + 1):
                for y in range(1, trimmed_outputs[k].size + 1):
                    error_matrix[x, y] = min([
                        error_matrix[x-1, y] + 1, error_matrix[x, y-1] + 1,
                        error_matrix[x-1, y-1] + (target[x-1] !=
                                                  trimmed_outputs[k][y-1])])

            ler[k] = error_matrix[-1, -1]/target.size

        return ler
