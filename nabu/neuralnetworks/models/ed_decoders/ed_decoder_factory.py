'''@file ed_decoder_factory
contains the ed decoder factory'''

from . import speller, lstm_decoder, linear_decoder

def factory(decoder):
    '''gets an ed decoder class

    Args:
        decoder: the decoder type

    Returns:
        An EDDecoder class'''

    if decoder == 'speller':
        return speller.Speller
    elif decoder == 'lstm_decoder':
        return lstm_decoder.LstmDecoder
    elif decoder == 'linear_decoder':
        return linear_decoder.LinearDecoder
    else:
        raise Exception('undefined decoder type: %s' % decoder)
