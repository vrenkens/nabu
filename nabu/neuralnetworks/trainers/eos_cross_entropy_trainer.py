'''@file eos_cross_entropy_trainer.py
contains the EosCrossEntropyTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks.components import ops

class EosCrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-entropy loss'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame

        Args:
            targets: a list of [batch_size x ...] tensor containing the
                targets
            logits: a list of [batch_size x ... tensor containing the
                logits
            logit_seq_length: a list of [batch_size] vectors containing the
                logit sequence lengths
            target_seq_length: a list of [batch_size] vectors containing the
                target sequence lengths

        Returns:
            a scalar value containing the loss
        '''

        numtargets = len(targets)

        with tf.name_scope('cross_entropy_loss'):
            losses = []

            for t in range(numtargets):
                losses.append(ops.cross_entropy_loss_eos(
                    targets[t], logits[t], logit_seq_length[t],
                    target_seq_length[t]
                ))

            loss = tf.reduce_sum(losses)

        return loss

    def get_output_dims(self, output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

        return [output_dim + 1 for output_dim in output_dims]
