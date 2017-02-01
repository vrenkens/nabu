'''@file timit.py
contains the timit target normalizer'''

def timit(transcription, _):
    '''
    normalizer for Timit training transcriptions, this just adds a <eos> token

    Args:
        transcription: the input transcription

    Returns:
        the normalized transcription as a string
    '''

    return transcription + ' <eos>'
