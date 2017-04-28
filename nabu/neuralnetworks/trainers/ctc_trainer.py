'''@file ctctrainer.py
contains the CTCTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks.components import ops

class CTCTrainer(trainer.Trainer):
    '''A trainer that minimises the CTC loss, the output sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

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

        with tf.name_scope('CTC_loss'):

            losses = []
            numtargets = len(targets)

            for t in range(numtargets):
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

    def get_output_dims(self, output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

        return [output_dim + 1 for output_dim in output_dims]
