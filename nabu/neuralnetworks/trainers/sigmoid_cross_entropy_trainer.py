'''@file sigmoid_cross_entropy_trainer.py
contains the SigmoidCrossEntropyTrainer'''

import tensorflow as tf
from nabu.neuralnetworks.trainers import trainer
from nabu.neuralnetworks.components import ops

class SigmoidCrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-entropy loss

    adds a end of sequence label to each target utterance'''

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

        with tf.name_scope('sigmoid_cross_entropy_loss'):
            losses = []

            for t in targets:
                #stack all the logits except the final logits
                stacked_logits = ops.seq2nonseq(logits[t], logit_seq_length[t])

                #create the stacked targets
                stacked_targets = ops.seq2nonseq(
                    tf.cast(targets[t], tf.float32),
                    target_seq_length[t])

                losses.append(tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=stacked_logits,
                        labels=stacked_targets)))

            loss = tf.reduce_sum(losses)

        return loss

    @property
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

        return 0
