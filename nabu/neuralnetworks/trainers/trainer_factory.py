'''@file trainer_factory.py
contains the Trainer factory mehod'''

def factory(trainer):
    '''gets a Trainer class

    Args:
        trainer: the trainer type

    Returns: a Trainer class
    '''

    if trainer == 'standard':
        import standard_trainer
        return standard_trainer.StandardTrainer
    else:
        raise Exception('Undefined trainer type: %s' % trainer)
