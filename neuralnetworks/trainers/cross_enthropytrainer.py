'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

import tensorflow as tf
import numpy as np
import trainer
from neuralnetworks import ops

class CrossEnthropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):

            #convert to non sequential data
            nonseq_targets = ops.seq2nonseq(targets, target_seq_length)
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #make a vector out of the targets
            nonseq_targets = tf.reshape(nonseq_targets, [-1])

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets,
                                        int(nonseq_logits.get_shape()[1]))

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                nonseq_logits, nonseq_targets))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        apply a softmax to the logits so the cross-enthropy can be computed

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a tensor with the same shape as logits with the label probabilities
        '''

        return tf.nn.softmax(logits)

    def validation_metric(self, outputs, targets):
        '''the cross-enthropy

        Args:
            outputs: the validation output, which is a matrix containing the
                label probabilities of size [batch_size, max_input_length, dim].
            targets: a list containing the ground truth target labels

        Returns:
            a numpy array containing the loss of all utterances
        '''

        loss = np.zeros(outputs.shape[0])

        for utt in range(outputs.shape[0]):
            loss[utt] += np.mean(-np.log(
                outputs[utt, np.arange(targets[utt].size), targets[utt]]))

        return loss
