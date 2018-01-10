'''@file aurora4.py
contains the global phoneset target normalizer'''

import unicodedata

def normalize(transcription, alphabet):
    '''normalize for the Global Phoneset database

    Args:
        transcription: the transcription to be normalized as a string

    Returns:
        the normalized transcription as a string space seperated per
        character'''

    #remove accents
    normalized = unicodedata.normalize(
        form='NFKD',
        unistr=transcription.decode('utf-8')).encode('ASCII', 'ignore')

    normalized = list(normalized.lower())


    #replace the spaces with <space>
    normalized = [character if character is not ' ' else '<space>'
                  for character in normalized]

    #replace unknown characters with <unk>
    normalized = [character if character in alphabet else '<unk>'
                  for character in normalized]

    return ' '.join(normalized)
