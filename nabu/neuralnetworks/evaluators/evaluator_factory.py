'''@file evaluator_factory.py
contains the Evaluator factory'''

def factory(evaluator):
    '''
    gets an evaluator class

    Args:
        evaluator: the evaluator type

    Returns:
        an evaluator class
    '''

    if evaluator == 'decoder_evaluator':
        import decoder_evaluator
        return decoder_evaluator.DecoderEvaluator
    elif evaluator == 'loss_evaluator':
        import loss_evaluator
        return loss_evaluator.LossEvaluator
    else:
        raise Exception('Undefined evaluator type: %s' % evaluator)
