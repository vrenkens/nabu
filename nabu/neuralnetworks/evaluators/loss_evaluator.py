'''@file loss_evaluator.py
contains the LossEvaluator class'''

import tensorflow as tf
import evaluator
from nabu.neuralnetworks.trainers import loss_functions

class LossEvaluator(evaluator.Evaluator):
    '''The Decoder Evaluator is used to evaluate a decoder'''


    def update_loss(self, loss, inputs, input_seq_length, targets,
                    target_seq_length):
        '''update the validation loss for a batch of data

        Args:
            loss: the current loss
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors
            targets: the targets to the neural network, this is a list of
                [batch_size x max_output_length] tensors.
            target_seq_length: The sequence lengths of the target utterances,
                this is a list of [batch_size] vectors

        Returns:
            an operation to update the loss'''

        with tf.name_scope('evaluate_loss'):

            #a variable to hold the total number of utterances
            num_utt = tf.get_variable(
                name='num_val_utterances',
                shape=[],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )

            logits, logit_seq_length = self.model(
                inputs, input_seq_length, targets, target_seq_length, False)

            batch_loss = loss_functions.factory(
                self.conf['loss'])(
                    targets,
                    logits,
                    logit_seq_length,
                    target_seq_length)

            #number of utterances in the batch
            batch_utt = tf.to_float(tf.shape(logits.values()[0])[0])

            new_num_utt = num_utt + batch_utt

            #an operation to update the loss
            update_loss = loss.assign(
                (loss*num_utt + batch_loss*batch_utt)/new_num_utt).op

            #add an operation to update the number of utterances
            with tf.control_dependencies([update_loss]):
                update_loss = num_utt.assign(new_num_utt).op

        return update_loss
