'''@file normalizer_factory.py
Contains the normalizer factory
'''

def factory(normalizer):
    '''get a normalizer class

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer class'''

    if normalizer == 'aurora4':
        import aurora4
        return aurora4.normalize
    elif normalizer == 'phones':
        import phones
        return phones.normalize
    elif normalizer == 'character':
        import character
        return character.normalize
    elif normalizer == 'gp':
        import gp
        return gp.normalize
    else:
        raise Exception('Undefined normalizer: %s' % normalizer)
