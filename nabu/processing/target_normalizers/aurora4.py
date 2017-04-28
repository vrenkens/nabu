'''@file aurora4.py
contains the aurora4 target normalizer'''

import normalizer

class Aurora4(normalizer.Normalizer):
    '''normalize for the Aurora 4 database'''

    def __call__(self, transcription):
        '''normalize a transcription

        Args:
            transcription: the transcription to be normalized as a string

        Returns:
            the normalized transcription as a string space seperated per
            character'''

        #create a dictionary of words that should be replaced
        replacements = {
            ',COMMA':'COMMA',
            '\"DOUBLE-QUOTE':'DOUBLE-QUOTE',
            '!EXCLAMATION-POINT':'EXCLAMATION-POINT',
            '&AMPERSAND':'AMPERSAND',
            '\'SINGLE-QUOTE':'SINGLE-QUOTE',
            '(LEFT-PAREN':'LEFT-PAREN',
            ')RIGHT-PAREN':'RIGHT-PAREN',
            '-DASH':'DASH',
            '-HYPHEN':'HYPHEN',
            '...ELLIPSIS':'ELLIPSIS',
            '.PERIOD':'PERIOD',
            '/SLASH':'SLASH',
            ':COLON':'COLON',
            ';SEMI-COLON':'SEMI-COLON',
            '<NOISE>': '',
            '?QUESTION-MARK': 'QUESTION-MARK',
            '{LEFT-BRACE': 'LEFT-BRACE',
            '}RIGHT-BRACE': 'RIGHT-BRACE'
            }

        #replace the words in the transcription
        replaced = ' '.join([word if word not in replacements
                             else replacements[word]
                             for word in transcription.split(' ')])

        #make the transcription lower case and put it into a list
        normalized = list(replaced.lower())

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
