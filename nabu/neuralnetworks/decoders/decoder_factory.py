'''@file decoder_factory.py
contains the decoder factory'''

from . import ctc_decoder, beam_search_decoder, attention_visualizer

def factory(decoder):
    '''
    gets a decoder class

    Args:
        decoder: the decoder type

    Returns:
        a decoder class
    '''

    if decoder == 'ctc_decoder':
        return ctc_decoder.CTCDecoder
    elif decoder == 'beam_search_decoder':
        return beam_search_decoder.BeamSearchDecoder
    elif decoder == 'attention_visualizer':
        return attention_visualizer.AttentionVisiualizer
    else:
        raise Exception('Undefined decoder type: %s' % decoder)
