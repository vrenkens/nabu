'''@file trainer_factory.py
contains the Trainer factory mehod'''

from . import cross_entropy_trainer, ctc_trainer, eos_cross_entropy_trainer

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
    else:
        raise Exception('Undefined trainer type: %s' % trainer)
