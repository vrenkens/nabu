'''@file max_decoder.py
contains the MaxDecoder'''

import os
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components import ops

class MaxDecoder(decoder.Decoder):
    '''max Decoder takes the maximum probability at each timestep'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        super(MaxDecoder, self).__init__(conf, model)

        self.alphabets = {}
        for o in model.output_names:
            #get the alphabet
            self.alphabets[o] = self.conf['%s_alphabet' % o].split(' ')

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a dictionary of [batch_size x time x ...]
                tensors
            input_seq_length: the input sequence lengths as a dictionary of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a dictionary of outputs
        '''

        with tf.name_scope('max_decoder'):

            #create the decoding graph
            logits, logits_seq_length =\
                self.model(
                    inputs, input_seq_length, targets=[],
                    target_seq_length=[], is_training=False)

            outputs = {}
            for out in logits:
                sm = tf.nn.softmax(logits[out])
                outputs[out] = (
                    tf.cast(tf.argmax(sm, axis=2), tf.int32),
                    logits_seq_length[out])

        return outputs

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''

        for o in outputs:
            batch_size = outputs[o][0].shape[0]
            with open(os.path.join(directory, o), 'a') as fid:
                for i in range(batch_size):
                    output = outputs[o][0][i, :outputs[o][1][i]]
                    text = ' '.join([self.alphabets[o][j] for j in output])
                    fid.write('%s %s\n' % (names[i], text))

    def update_evaluation_loss(self, loss, outputs, references,
                               reference_seq_length):
        '''update the evaluation loss

        args:
            loss: the current evaluation loss
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            an op to update the evalution loss
        '''

        #create a valiable to hold the total number of reference targets
        num_targets = tf.get_variable(
            name='num_targets',
            shape=[],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        #stack all the logits
        stacked_outputs = {
            t:ops.stack_seq(outputs[t][0], outputs[t][1])
            for t in outputs}


        #create the stacked targets
        stacked_targets = {
            t:tf.cast(ops.stack_seq(references[t],
                                    reference_seq_length[t]), tf.int32)
            for t in references}

        #compute the number of errors
        errors = [
            tf.reduce_sum(tf.reduce_sum(tf.cast(tf.not_equal(
                stacked_outputs[o], stacked_targets[o]), tf.float32)))
            for o in outputs]

        errors = tf.reduce_sum(errors)

        #compute the number of targets in this batch
        batch_targets = tf.reduce_sum([
            tf.reduce_sum(lengths)
            for lengths in reference_seq_length.values()])

        new_num_targets = num_targets + tf.cast(batch_targets, tf.float32)

        #an operation to update the loss
        update_loss = loss.assign(
            (loss*num_targets + errors)/new_num_targets).op

        #add an operation to update the number of targets
        with tf.control_dependencies([update_loss]):
            update_loss = num_targets.assign(new_num_targets).op

        return update_loss
