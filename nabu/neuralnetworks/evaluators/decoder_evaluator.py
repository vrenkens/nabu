'''@file decoder_evaluator.py
contains the DecoderEvaluator class'''

import tensorflow as tf
import evaluator
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

class DecoderEvaluator(evaluator.Evaluator):
    '''The Decoder Evaluator is used to evaluate a decoder'''

    def __init__(self, conf, dataconf, model_or_conf):
        '''DecoderEvaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            model_or_conf: eihther a model object or a model
                configuration as a configparser
        '''


        super(DecoderEvaluator, self).__init__(conf, dataconf,
                                               model_or_conf)

        #create a decoder object
        decoderconf = dict(conf.items('decoder'))
        self.decoder = decoder_factory.factory(decoderconf['decoder'])(
            decoderconf, self.model)

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

            #use the decoder to decoder
            sequences, _ = self.decoder(inputs, input_seq_length)

            #compute the edit distance for the decoded sequences
            #convert the representations to sparse Tensors
            sparse_targets = dense_sequence_to_sparse(targets[0],
                                                      target_seq_length[0])

            #compute the edit distance
            loss = tf.reduce_mean(tf.edit_distance(sequences[0],
                                                   sparse_targets))

        return loss

    def get_output_dims(self, output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

        decoderconf = dict(self.conf.items('decoder'))

        return decoder_factory.factory(decoderconf['decoder']).get_output_dims(
            output_dims)
