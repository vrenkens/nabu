'''@file trainer_factory.py
contains the Trainer factory mehod'''

from . import standard_trainer, fisher_trainer

def factory(trainer):
    '''gets a Trainer class

    Args:
        trainer: the trainer type

    Returns: a Trainer class
    '''

    if trainer == 'fisher':
        return fisher_trainer.FisherTrainer
    elif trainer == 'standard':
        return standard_trainer.StandardTrainer
    else:
        raise Exception('Undefined trainer type: %s' % trainer)
