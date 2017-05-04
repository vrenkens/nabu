'''@file normalizer_factory.py
Contains the normalizer factory
'''

from . import timit, aurora4, librispeech, timit_text

def factory(normalizer):
    '''get a normalizer class

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer class'''

    if normalizer == 'aurora4':
        return aurora4.Aurora4
    elif normalizer == 'timit':
        return timit.Timit
    elif normalizer == 'librispeech':
        return librispeech.Librispeech
    elif normalizer == 'timit_text':
        return timit_text.TimitText
    else:
        raise Exception('Undefined normalizer: %s' % normalizer)
