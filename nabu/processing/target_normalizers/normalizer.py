'''@file normalizer.py
contains the Normalizer class'''

from abc import ABCMeta, abstractmethod

class Normalizer(object):
    '''general class for text normalizers'''

    __metaclass__ = ABCMeta

    def __init__(self):
        '''Normalizer constructor'''

        #create the alphabet
        self.alphabet = self._create_alphabet()

    @abstractmethod
    def __call__(self, transcription):
        '''normalize a transcription

        Args:
            transcription: the transcription to be normalized as a string

        Returns:
            the normalized transcription as a string space seperated per
            character'''

    @abstractmethod
    def _create_alphabet(self):
        '''create the alphabet that is used in the normalizer

        Returns:
            the alphabet as a list of strings'''
