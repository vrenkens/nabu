'''@file timit_text.py
contains the timit text target normalizer'''

import normalizer

class TimitText(normalizer.Normalizer):
    '''normalize for the timit text database'''

    def __call__(self, transcription):
        '''normalize a transcription

        Args:
            transcription: the transcription to be normalized as a string

        Returns:
            the normalized transcription as a string space seperated per
            character'''

        #make the transcription lower case and put it into a list
        normalized = list(transcription.lower())

        #replace the spaces with <space>
        normalized = [character if character is not ' ' else '<space>'
                      for character in normalized]

        #replace unknown characters with <unk>
        normalized = [character if character in self.alphabet else '<unk>'
                      for character in normalized]

        return ' '.join(normalized)

    def _create_alphabet(self):
        '''create the alphabet that is used in the normalizer

        Returns:
            the alphabet as a list of strings'''

        alphabet = []

        #space
        alphabet.append('<space>')

        #punctuation
        alphabet.append('.')
        alphabet.append('?')
        alphabet.append('!')
        alphabet.append(',')
        alphabet.append(';')
        alphabet.append('\"')
        alphabet.append('\'')

        #unknown character
        alphabet.append('<unk>')

        #letters in the alphabet
        for letter in range(ord('a'), ord('z')+1):
            alphabet.append(chr(letter))

        return alphabet
