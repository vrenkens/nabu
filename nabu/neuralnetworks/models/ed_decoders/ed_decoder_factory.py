'''@file ed_decoder_factory
contains the ed decoder factory'''

from . import speller, linear_decoder, phonology_decoder

def factory(decoder):
    '''gets an ed decoder class

    Args:
        decoder: the decoder type

    Returns:
        An EDDecoder class'''

    if decoder == 'speller':
        return speller.Speller
    elif decoder == 'linear_decoder':
        return linear_decoder.LinearDecoder
    elif decoder == 'phonology_decoder':
        return phonology_decoder.PhonologyDecoder
    else:
        raise Exception('undefined decoder type: %s' % decoder)
