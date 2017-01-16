'''@file classifier_factory
contains the classifier factory'''

import classifiers

def classifier_factory(conf, output_dim, classifier_type):
    '''create a classifier

    Args:
        conf: the classifier config as a dictionary
        output_dim: the classifier output dimension
        classifier_type: the classifier type as a string

    Returns:
        A Classifier object'''

    if classifier_type == 'dblstm':
        return classifiers.dblstm.DBLSTM(conf, output_dim)
    elif classifier_type == 'dnn':
        return classifiers.dnn.DNN(conf, output_dim)
    elif classifier_type == 'wavenet':
        return classifiers.wavenet.Wavenet(conf, output_dim)
    elif classifier_type == 'las':
        return classifiers.las.LAS(conf, output_dim)
    else:
        raise Exception('undefined classifier type: %s' % classifier_type)
