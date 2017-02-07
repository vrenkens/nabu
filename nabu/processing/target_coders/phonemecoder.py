'''@file phonemecoder.py
contain the PhonemeCoder class'''

import targetcoder

class PhonemeCoder(targetcoder.TargetCoder):
    ''' A coder fir the 39 foldet phoneme alphabet.'''

    def create_alphabet(self):
        '''
        Create an alphabet of folded phonemes.

        Returns:
            The coder alphabet
        '''

        alphabet = ['<eos>', '<sos>', 'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b',
                    'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh',
                    'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p',
                    'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']

        return alphabet
