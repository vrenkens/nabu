'''@file decoder_factory.py
contains the decoder factory'''

import ctc_decoder
import beam_search_decoder
import attention_visualizer
import lm_confidence_decoder

def factory(conf,
            classifier,
            input_dim,
            max_input_length,
            coder,
            expdir):
    '''
    creates a decoder object

    Args:
        conf: the decoder config
        classifier: the classifier that will be used for decoding
        input_dim: the input dimension to the nnnetgraph
        max_input_length: the maximum length of the inputs
        coder: a TargetCoder object
        expdir: the location where the models were saved and the results
            will be written
    '''

    if conf['decoder'] == 'ctcdecoder':
        decoder_class = ctc_decoder.CTCDecoder
    elif conf['decoder'] == 'beamsearchdecoder':
        decoder_class = beam_search_decoder.BeamSearchDecoder
    elif conf['decoder'] == 'attention_visualizer':
        decoder_class = attention_visualizer.AttentionVisiualizer
    elif conf['decoder'] == 'lm_confidence_decoder':
        decoder_class = lm_confidence_decoder.LmConfidenceDecoder
    else:
        raise Exception('Undefined decoder type: %s' % conf['decoder'])

    return decoder_class(conf,
                         classifier,
                         input_dim,
                         max_input_length,
                         coder,
                         expdir)
