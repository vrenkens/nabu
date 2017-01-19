'''@file classifier_factory
contains the classifier factory'''

import dblstm
import dnn
import wavenet
import las

def factory(conf, output_dim, classifier_type):
    '''create a classifier

    Args:
        conf: the classifier config as a dictionary
        output_dim: the classifier output dimension
        classifier_type: the classifier type as a string

    Returns:
        A Classifier object'''

    if classifier_type == 'dblstm':
        return dblstm.DBLSTM(conf, output_dim)
    elif classifier_type == 'dnn':
        return dnn.DNN(conf, output_dim)
    elif classifier_type == 'wavenet':
        return wavenet.Wavenet(conf, output_dim)
    elif classifier_type == 'las':
        return las.LAS(conf, output_dim)
    else:
        raise Exception('undefined classifier type: %s' % classifier_type)
