'''@file lstmlm.py
contains the LstmLm class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier
import lm_decoders

class LstmLm(classifier.Classifier):
    '''a LSTM language model'''

    def __init__(self, conf, output_dim, name=None):
        '''LstnLm constructor

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        #create the speller
        self.decoder = lm_decoders.lstm_decoder.LstmDecoder(
            numlayers=int(conf['numlayers']),
            numunits=int(conf['numunits']),
            dropout=float(conf['dropout']))

        super(LstmLm, self).__init__(conf, output_dim, name)

    def _get_outputs(self, inputs, input_seq_length, targets,
                     target_seq_length, is_training):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        #prepend a sequence border label to the inputs to get the encoder
        #inputs, the label is the last label
        formatted_inputs = tf.cast(inputs, tf.int32)[:, :, 0]
        batch_size = int(inputs.get_shape()[0])
        s_labels = tf.constant(self.output_dim-1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])
        encoder_inputs = tf.concat(1, [s_labels, formatted_inputs])

        #compute the output logits
        logits, _ = self.decoder(
            encoder_inputs=encoder_inputs,
            numlabels=self.output_dim,
            initial_state=None,
            is_training=is_training)

        return logits, input_seq_length + 1
