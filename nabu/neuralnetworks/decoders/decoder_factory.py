'''@file decoder_factory.py
contains the decoder factory'''

from . import ctc_decoder, beam_search_decoder, max_decoder, threshold_decoder,\
 feature_decoder, alignment_decoder

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
    elif decoder == 'max_decoder':
        return max_decoder.MaxDecoder
    elif decoder == 'threshold_decoder':
        return threshold_decoder.ThresholdDecoder
    elif decoder == 'feature_decoder':
        return feature_decoder.FeatureDecoder
    elif decoder == 'alignment_decoder':
        return alignment_decoder.AlignmentDecoder
    else:
        raise Exception('Undefined decoder type: %s' % decoder)
