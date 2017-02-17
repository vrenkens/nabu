'''@file timit.py
contains the timit target normalizer'''

import normalizer

class Timit(normalizer.Normalizer):
    '''the timit 39 folded phonemes normalizer'''

    def __call__(self, transcription):
        '''normalize a transcription

        Args:
            transcription: the transcription to be normalized as a string

        Returns:
            the normalized transcription as a string space seperated per
            character'''

        return transcription

    def _create_alphabet(self):
        '''create the alphabet that is used in the normalizer

        Returns:
            the alphabet as a list of strings'''

        alphabet = ['sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b',
                    'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh',
                    'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p',
                    'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']

        return alphabet
