'''@file normalizer_factory.py
Contains the normalizer factory
'''

import aurora4
import timit
import gp

def factory(normalizer_type):
    '''create a normalizer_type

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer function'''

    if normalizer_type == 'aurora4_normalizer':
        return aurora4.Aurora4()
    elif normalizer_type == 'timit_phone_norm':
        return timit.Timit()
    elif normalizer_type == 'gp':
        return gp.Gp()
    else:
        raise Exception('Undefined normalizer: %s' % normalizer_type)
