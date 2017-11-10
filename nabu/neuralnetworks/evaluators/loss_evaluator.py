'''@file loss_evaluator.py
contains the LossEvaluator class'''

import tensorflow as tf
import evaluator
from nabu.neuralnetworks.trainers import loss_functions

class LossEvaluator(evaluator.Evaluator):
    '''The Decoder Evaluator is used to evaluate a decoder'''


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

        with tf.name_scope('evaluate_decoder'):
            logits, logit_seq_length = self.model(
                inputs, input_seq_length, targets, target_seq_length, False)

            loss = loss_functions.factory(
                self.conf['loss'])(
                    targets,
                    logits,
                    logit_seq_length,
                    target_seq_length)

        return loss
