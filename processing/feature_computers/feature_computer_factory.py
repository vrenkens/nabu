'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

import mfcc
import fbank

def factory(conf):
    '''
    create a FeatureComputer

    Args:
        conf: the feature configuration
    '''

    if conf['feature'] == 'fbank':
        return fbank.Fbank(conf)
    elif conf['feature'] == 'mfcc':
        return mfcc.Mfcc(conf)
    else:
        raise Exception('Undefined feature type: %s' % conf['feature'])
