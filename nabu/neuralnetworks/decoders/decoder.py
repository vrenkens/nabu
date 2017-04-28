'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        self.conf = conf
        self.model = model

    @abstractmethod
    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a list of [batch_size x ...] tensors
            input_seq_length: the input sequence lengths as a list of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a list of length beam_width
                containing [batch_size x ...] sparse tensors, the beam elements
                are sorted from best to worst
            - the sequence scores as a [batch_size x beam_width] tensor
        '''

    @abstractmethod
    def get_output_dims(self, output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)
        WARNING: This should be a static method

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''
