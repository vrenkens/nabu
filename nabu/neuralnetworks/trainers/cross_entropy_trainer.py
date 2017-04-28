'''@file cross_entropy_trainer.py
contains the CrossEntropyTrainer'''

import tensorflow as tf
from nabu.neuralnetworks.trainers import trainer
from nabu.neuralnetworks.components import ops

class CrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-entropy loss

    adds a end of sequence label to each target utterance'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame and ads an end of sequence label to the targets

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
                #stack all the logits except the final logits
                stacked_logits = ops.seq2nonseq(logits[t], logit_seq_length[t])

                #create the stacked targets
                stacked_targets = ops.seq2nonseq(targets[t],
                                                 target_seq_length[t])

                losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=stacked_logits,
                    labels=stacked_targets))

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

        return output_dims
