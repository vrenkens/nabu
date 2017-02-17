'''@file target_coder.py
contains the TargetCoder class'''

from collections import OrderedDict
import numpy as np

class TargetCoder(object):
    '''an abstract class for a target coder which can encode and decode target
    sequences'''

    def __init__(self, alphabet):
        '''
        TargetCoder constructor

        Args:
            alphabet: the alphabet that is used
        '''

        #create a lookup dictionary for fast encoding
        self.lookup = OrderedDict([(character, index) for index, character
                                   in enumerate(alphabet)])

    def encode(self, targets):
        '''
        encode a target sequence

        Args:
            targets: a string containing the target sequence

        Returns:
            A numpy array containing the encoded targets
        '''

        encoded_targets = []

        for target in targets.split(' '):
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

    @property
    def num_labels(self):
        '''the number of possible labels'''

        return len(self.lookup)
