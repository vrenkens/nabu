'''@file cross_enthropy_evauator.py
contains the CrossEnthropyEvaluator class'''

import tensorflow as tf
import cross_entropy_evaluator

class PerplexityEvaluator(cross_entropy_evaluator.CrossEntropyEvaluator):
    '''a evaluator for evaluating the perplexity'''

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

        with tf.name_scope('perplexity'):

            cross_entropy = super(PerplexityEvaluator, self).compute_loss(
                inputs, input_seq_length, targets, target_seq_length)

            loss = tf.exp(cross_entropy)

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
