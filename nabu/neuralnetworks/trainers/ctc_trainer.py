'''@file ctctrainer.py
contains the CTCTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks.components import ops

class CTCTrainer(trainer.Trainer):
    '''A trainer that minimises the CTC loss'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame and ads an end of sequence label to the targets

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensor containing
                the logits
            logit_seq_length: a dictionary of [batch_size] vectors containing
                the logit sequence lengths
            target_seq_length: a dictionary of [batch_size] vectors containing
                the target sequence lengths

        Returns:
            a scalar value containing the loss
        '''
        with tf.name_scope('CTC_loss'):

            losses = []

            for t in targets:
                #convert the targets into a sparse tensor representation
                sparse_targets = ops.dense_sequence_to_sparse(
                    targets[t], target_seq_length[t])

                losses.append(tf.reduce_mean(tf.nn.ctc_loss(
                    sparse_targets,
                    logits[t],
                    logit_seq_length[t],
                    time_major=False)))

            loss = tf.reduce_sum(losses)

        return loss

    @property
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

        return 1
