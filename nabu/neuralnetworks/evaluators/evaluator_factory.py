'''@file evaluator_factory.py
contains the Evaluator factory'''

from . import cross_entropy_evaluator, ctc_evaluator, decoder_evaluator, \
perplexity_evaluator, eos_cross_entropy_evaluator

def factory(evaluator):
    '''
    gets an evaluator class

    Args:
        evaluator: the evaluator type

    Returns:
        an evaluator class
    '''

    if evaluator == 'cross_entropy_evaluator':
        return cross_entropy_evaluator.CrossEntropyEvaluator
    elif evaluator == 'eos_cross_entropy_evaluator':
        return eos_cross_entropy_evaluator.EosCrossEntropyEvaluator
    elif evaluator == 'ctc_evaluator':
        return ctc_evaluator.CTCEvaluator
    elif evaluator == 'perplexity_evaluator':
        return perplexity_evaluator.PerplexityEvaluator
    elif evaluator == 'decoder_evaluator':
        return decoder_evaluator.DecoderEvaluator
    else:
        raise Exception('Undefined evaluator type: %s' % evaluator)
