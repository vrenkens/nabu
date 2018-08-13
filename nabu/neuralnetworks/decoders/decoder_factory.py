'''@file decoder_factory.py
contains the decoder factory'''

def factory(decoder):
    '''
    gets a decoder class

    Args:
        decoder: the decoder type

    Returns:
        a decoder class
    '''

    if decoder == 'ctc_decoder':
        import ctc_decoder
        return ctc_decoder.CTCDecoder
    elif decoder == 'beam_search_decoder':
        import beam_search_decoder
        return beam_search_decoder.BeamSearchDecoder
    elif decoder == 'max_decoder':
        import max_decoder
        return max_decoder.MaxDecoder
    elif decoder == 'threshold_decoder':
        import threshold_decoder
        return threshold_decoder.ThresholdDecoder
    elif decoder == 'feature_decoder':
        import feature_decoder
        return feature_decoder.FeatureDecoder
    elif decoder == 'alignment_decoder':
        import alignment_decoder
        return alignment_decoder.AlignmentDecoder
    elif decoder == 'random_decoder':
        import random_decoder
        return random_decoder.RandomDecoder
    else:
        raise Exception('Undefined decoder type: %s' % decoder)
