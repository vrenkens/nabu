'''@file loss_evaluator.py
contains the LossEvaluator class'''

import tensorflow as tf
import evaluator

class LossEvaluator(evaluator.Evaluator):
    '''The Decoder Evaluator is used to evaluate a decoder'''

    def __init__(self, conf, dataconf, model, loss_function):
        '''Evaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            model: the model to be evaluated
            loss_function: the loss_function tpo be used in compute loss
        '''


        super(LossEvaluator, self).__init__(conf, dataconf, model)
        self.loss_function = loss_function



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

            loss = self.loss_function(
                targets, logits, logit_seq_length, target_seq_length)

        return loss
