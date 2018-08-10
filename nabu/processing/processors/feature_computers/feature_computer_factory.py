'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

def factory(feature):
    '''
    create a FeatureComputer

    Args:
        feature: the feature computer type
    '''

    if feature == 'fbank':
        import fbank
        return fbank.Fbank
    elif feature == 'mfcc':
        import mfcc
        return mfcc.Mfcc
    else:
        raise Exception('Undefined feature type: %s' % feature)
