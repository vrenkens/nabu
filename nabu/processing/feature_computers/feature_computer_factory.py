'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

import mfcc
import fbank

def factory(feature):
    '''
    create a FeatureComputer

    Args:
        feature: the feature computer type
    '''

    if feature == 'fbank':
        return fbank.Fbank
    elif feature == 'mfcc':
        return mfcc.Mfcc
    else:
        raise Exception('Undefined feature type: %s' % feature)
