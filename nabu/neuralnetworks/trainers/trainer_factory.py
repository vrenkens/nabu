'''@file trainer_factory.py
contains the Trainer factory mehod'''

import cross_entropytrainer
import ctctrainer

def factory(conf,
            decoder,
            classifier,
            input_dim,
            dispenser,
            val_reader,
            val_targets,
            expdir,
            server,
            task_index):
    '''Create a Trainer object

    Args:
        classifier: the neural net classifier that will be trained
        conf: the trainer config
        decoder: a callable that will create a decoder
        input_dim: the input dimension to the nnnetgraph
        num_steps: the total number of steps that will be taken
        dispenser: a Batchdispenser object
        val_reader: a feature reader for the validation data if None
            validation will not be used
        val_targets: a dictionary containing the targets of the validation set
        logdir: directory where the summaries will be written
        server: optional server to be used for distributed training
        cluster: optional cluster to be used for distributed training
        task_index: optional index of the worker task in the cluster

    Returns: a Trainer object
    '''

    if conf['trainer'] == 'ctc':
        trainer_class = ctctrainer.CTCTrainer
    elif conf['trainer'] == 'cross_entropy':
        trainer_class = cross_entropytrainer.CrossEntropyTrainer
    else:
        raise Exception('Undefined trainer type: %s' % conf['trainer'])

    return trainer_class(conf,
                         decoder,
                         classifier,
                         input_dim,
                         dispenser,
                         val_reader,
                         val_targets,
                         expdir,
                         server,
                         task_index)
