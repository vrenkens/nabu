'''@file targetcoder.py
contains the TargetCoder class'''

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

class TargetCoder(object):
    '''an abstract class for a target coder which can encode and decode target
    sequences'''

    __metaclass__ = ABCMeta

    def __init__(self, target_normalizer=None):
        '''
        TargetCoder constructor

        Args:
            target_normalizer: [optional] a target normalizer function, if None
            is set, the endoded utterance will not be normalized.
        '''

        #save the normalizer
        self.target_normalizer = target_normalizer

        #create an alphabet of possible targets
        alphabet = self.create_alphabet()

        #create a lookup dictionary for fast encoding
        self.lookup = OrderedDict([(character, index) for index, character
                                   in enumerate(alphabet)])

    @abstractmethod
    def create_alphabet(self):
        '''create the alphabet for the coder'''

    def encode(self, targets):
        '''
        encode a target sequence

        Args:
            targets: a string containing the target sequence

        Returns:
            A numpy array containing the encoded targets
        '''

        #normalize the targets
        normalized_targets = self.normalize(targets)

        encoded_targets = []

        for target in normalized_targets.split(' '):
            encoded_targets.append(self.lookup[target])

        return np.array(encoded_targets, dtype=np.uint32)

    def decode(self, encoded_targets):
        '''
        decode an encoded target sequence

        Args:
            encoded_targets: A numpy array containing the encoded targets

        Returns:
            A string containing the decoded target sequence
        '''

        targets = [self.lookup.keys()[encoded_target]
                   for encoded_target in encoded_targets]

        return ' '.join(targets)

    def normalize(self, targets):
        '''
        normalize a target sequence

        Args:
            targets: a string containing the target sequence

        Returns:
            A string containing the normalized targets
        '''
        if self.target_normalizer is not None:
            normalized_targets = self.target_normalizer(targets,
                                                        self.lookup.keys())
        else:
            normalized_targets = targets

        return normalized_targets

    @property
    def num_labels(self):
        '''the number of possible labels'''

        return len(self.lookup)
