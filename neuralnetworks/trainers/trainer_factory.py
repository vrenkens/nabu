import cross_enthropytrainer
import ctctrainer

def factory(conf,
                    decoder_conf,
                    classifier,
                    input_dim,
                    max_input_length,
                    max_target_length,
                    dispenser,
                    val_reader,
                    val_targets,
                    expdir,
                    server,
                    cluster,
                    task_index,
                    trainer_type):

    '''Create a Trainer object

    Args:
        classifier: the neural net classifier that will be trained
        conf: the trainer config
        decoder_conf: the decoder config used for validation
        input_dim: the input dimension to the nnnetgraph
        max_input_length: the maximal length of the input sequences
        max_target_length: the maximal length of the target sequences
        num_steps: the total number of steps that will be taken
        dispenser: a Batchdispenser object
        cluster: the optional cluster used for distributed training, it
            should contain at least one parmeter server and one worker
        val_reader: a feature reader for the validation data if None
            validation will not be used
        val_targets: a dictionary containing the targets of the validation set
        logdir: directory where the summaries will be written
        server: optional server to be used for distributed training
        cluster: optional cluster to be used for distributed training
        task_index: optional index of the worker task in the cluster
        trainer_type: the trainer type

    Returns: a Trainer object
    '''

    if trainer_type == 'ctc':
        trainer_class = ctctrainer.CTCTrainer
    elif trainer_type == 'cross_enthropy':
        trainer_class = cross_enthropytrainer.CrossEnthropyTrainer
    else:
        raise Exception('Undefined trainer type: %s' % trainer_type)

    return trainer_class(conf,
                         decoder_conf,
                         classifier,
                         input_dim,
                         max_input_length,
                         max_target_length,
                         dispenser,
                         val_reader,
                         val_targets,
                         expdir,
                         server,
                         cluster,
                         task_index)
