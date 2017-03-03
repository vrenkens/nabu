'''@file ctctrainer.py
contains the CTCTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

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

        with tf.name_scope('CTC_loss'):

            #get the batch size
            targets = tf.expand_dims(targets, 2)
            batch_size = int(targets.get_shape()[0])

            #convert the targets into a sparse tensor representation
            indices = tf.concat([tf.concat(
                [tf.expand_dims(tf.tile([s], [target_seq_length[s]]), 1),
                 tf.expand_dims(tf.range(target_seq_length[s]), 1)], 1)
                                 for s in range(batch_size)], 0)

            values = tf.reshape(
                ops.seq2nonseq(targets, target_seq_length), [-1])

            shape = [batch_size, int(targets.get_shape()[1])]

            sparse_targets = tf.SparseTensor(tf.cast(indices, tf.int64), values,
                                             shape)

            loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_targets, logits,
                                                 logit_seq_length,
                                                 time_major=False))

        return loss
