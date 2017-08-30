'''@file normalizer_factory.py
Contains the normalizer factory
'''

from . import character, aurora4, phones

def factory(normalizer):
    '''get a normalizer class

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer class'''

    if normalizer == 'aurora4':
        return aurora4.normalize
    elif normalizer == 'phones':
        return phones.normalize
    elif normalizer == 'character':
        return character.normalize
    elif normalize == 'gp':
        return gp.normalize
    else:
        raise Exception('Undefined normalizer: %s' % normalizer)
