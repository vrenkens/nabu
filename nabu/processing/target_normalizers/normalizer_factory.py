'''@file normalizer_factory.py
Contains the normalizer factory
'''

from . import timit, aurora4, librispeech

def factory(normalizer):
    '''get a normalizer class

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer class'''

    if normalizer == 'aurora4_normalizer':
        return aurora4.Aurora4
    elif normalizer == 'timit_phone_norm':
        return timit.Timit
    elif normalizer == 'librispeech':
        return librispeech.Librispeech
    else:
        raise Exception('Undefined normalizer: %s' % normalizer)
