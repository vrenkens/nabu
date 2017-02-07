'''@file textcoder.py
contain the TextCoder class'''

import targetcoder

class TextCoder(targetcoder.TargetCoder):
    '''a coder for text'''

    def create_alphabet(self):
        '''create the alphabet of characters

        Returns:
            The coder alphabet'''

        alphabet = []

        #space
        alphabet.append('<space>')

        #comma
        alphabet.append(',')

        #period
        alphabet.append('.')

        #apostrophy
        alphabet.append('\'')

        #hyphen
        alphabet.append('-')

        #question mark
        alphabet.append('?')

        #unknown character
        alphabet.append('<unk>')

        #letters in the alphabet
        for letter in range(ord('a'), ord('z')+1):
            alphabet.append(chr(letter))

        return alphabet
