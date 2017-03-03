'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

class CrossEntropyTrainer(trainer.Trainer):
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
            targets: a [batch_size, max_target_length] tensor containing the
                targets
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                [batch_size] vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):
            output_dim = int(logits.get_shape()[2])

            #put all the tragets on top of each other
            split_targets = tf.unstack(targets)
            for i, target in enumerate(split_targets):
                #only use the real data
                split_targets[i] = target[:target_seq_length[i]]

                #append an end of sequence label
                split_targets[i] = tf.concat(
                    [split_targets[i], [output_dim-1]], 0)

            #concatenate the targets
            nonseq_targets = tf.concat(split_targets, 0)

            #convert the logits to non sequential data
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_logits, labels=nonseq_targets))

        return loss
