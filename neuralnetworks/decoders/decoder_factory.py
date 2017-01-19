'''@file decoder_factory.py
contains the decoder factory'''

import ctc_decoder
import greedy_decoder
import beam_search_decoder

def factory(conf,
            classifier,
            classifier_scope,
            input_dim,
            max_input_length,
            coder,
            expdir,
            decoder_type):
    '''
    creates a decoder object

    Args:
        conf: the decoder config
        classifier: the classifier that will be used for decoding
        classifier_scope: the scope where the classier should be created/loaded
            from
        input_dim: the input dimension to the nnnetgraph
        max_input_length: the maximum length of the inputs
        coder: a TargetCoder object
        expdir: the location where the models were saved and the results
            will be written
        decoder_type: the decoder type
    '''

    if decoder_type == 'ctcdecoder':
        decoder_class = ctc_decoder.CTCDecoder
    elif decoder_type == 'greedydecoder':
        decoder_class = greedy_decoder.GreedyDecoder
    elif decoder_type == 'beamsearchdecoder':
        decoder_class = beam_search_decoder.BeamSearchDecoder
    else:
        raise Exception('Undefined decoder type: %s' % decoder_type)

    return decoder_class(conf,
                         classifier,
                         classifier_scope,
                         input_dim,
                         max_input_length,
                         coder,
                         expdir)
