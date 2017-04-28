'''@file cross_enthropy_evauator.py
contains the CTCEvaluator class'''

import tensorflow as tf
import evaluator
from nabu.neuralnetworks.components import ops

class CTCEvaluator(evaluator.Evaluator):
    '''a evaluator for evaluating the CTC loss'''

    def compute_loss(self, inputs, input_seq_length, targets,
                     target_seq_length):
        '''compute the validation loss for a batch of data

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors
            targets: the targets to the neural network, this is a list of
                [batch_size x max_output_length] tensors.
            target_seq_length: The sequence lengths of the target utterances,
                this is a list of [batch_size] vectors

        Returns:
            the loss as a scalar'''

        #compute logits
        logits, logit_seq_length = self.model(
            inputs=inputs,
            input_seq_length=input_seq_length,
            targets=targets,
            target_seq_length=target_seq_length,
            is_training=False)

        #compte loss
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
