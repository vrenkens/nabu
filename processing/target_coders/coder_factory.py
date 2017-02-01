'''@file coder_factory.py
contain the coder factory'''

import textcoder
import phonemecoder

def factory(target_normalizer, coder_type):
    '''create a target coder

    Args:
        target_normalizer: a target normalizer function
        coder_type: the type of coder to create

    Returns:
        a TargetCoder object'''

    if coder_type == 'textcoder':
        return textcoder.TextCoder(target_normalizer)
    elif coder_type == 'phonemecoder':
        return phonemecoder.PhonemeCoder(target_normalizer)
    else:
        raise Exception('Undefined coder type: %s' % coder_type)
