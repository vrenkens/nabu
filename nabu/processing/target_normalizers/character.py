'''@file character.py
contains the character target normalizer'''

def normalize(transcription, alphabet):
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
    normalized = [character if character in alphabet else '<unk>'
                  for character in normalized]

    return ' '.join(normalized)
