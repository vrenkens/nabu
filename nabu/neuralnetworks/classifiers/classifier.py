'''@file classifier.py
The abstract class for a neural network classifier'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Classifier(object):
    '''This an abstract class defining a neural net classifier'''
    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim, name=None):
        '''classifier constructor

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        self.conf = conf
        self.output_dim = output_dim

        #increase the output dim with the amount of labels that should be added
        self.output_dim += int(conf['add_labels'])

        #create the variable scope for the classifier
        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, input_seq_length, targets,
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

        with tf.variable_scope(self.scope):
            outputs, output_seq_lengths = self._get_outputs(
                inputs, input_seq_length, targets, target_seq_length,
                is_training)

        #put the reuse flag to true in the scope to make sure the variables are
        #reused in the next call
        self.scope.reuse_variables()

        return outputs, output_seq_lengths

    @abstractmethod
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

        raise NotImplementedError("Abstract method")
