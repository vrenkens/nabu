'''@file classifier.py
The abstract class for a neural network classifier'''

from abc import ABCMeta, abstractmethod

class Classifier(object):
    '''This an abstract class defining a neural net classifier'''
    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim):
        '''classifier constructor'''

        self.conf = conf
        self.output_dim = output_dim
        if conf['blank_label'] == 'True':
            self.output_dim += 1

    @abstractmethod
    def __call__(self, inputs, input_seq_length, targets=None,
                 target_seq_length=None, is_training=False, scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] dimansional vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length x 1] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] dimansional vector
            is_training: whether or not the network is in training mode
            scope: the name scope

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        raise NotImplementedError("Abstract method")
