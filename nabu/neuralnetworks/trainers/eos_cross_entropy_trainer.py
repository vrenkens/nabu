'''@file eos_cross_entropy_trainer.py
contains the EosCrossEntropyTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks.components import ops

class EosCrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-entropy loss with an eos label'''

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

        with tf.name_scope('cross_entropy_loss'):
            losses = []

            for t in targets:
                losses.append(ops.cross_entropy_loss_eos(
                    targets[t], logits[t], logit_seq_length[t],
                    target_seq_length[t]
                ))

            loss = tf.reduce_sum(losses)

        return loss

    @property
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

        return 1
