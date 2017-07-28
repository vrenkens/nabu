'''@file trainer_factory.py
contains the Trainer factory mehod'''

from . import cross_entropy_trainer, ctc_trainer, eos_cross_entropy_trainer,\
ctc_phonology_trainer, sigmoid_cross_entropy_trainer

def factory(trainer):
    '''gets a Trainer class

    Args:
        trainer: the trainer type

    Returns: a Trainer class
    '''

    if trainer == 'ctc':
        return ctc_trainer.CTCTrainer
    elif trainer == 'cross_entropy':
        return cross_entropy_trainer.CrossEntropyTrainer
    elif trainer == 'eos_cross_entropy':
        return eos_cross_entropy_trainer.EosCrossEntropyTrainer
    elif trainer == 'ctc_phonology':
        return ctc_phonology_trainer.CTCPhonologyTrainer
    elif trainer == 'sigmoid_cross_entropy':
        return sigmoid_cross_entropy_trainer.SigmoidCrossEntropyTrainer
    else:
        raise Exception('Undefined trainer type: %s' % trainer)
