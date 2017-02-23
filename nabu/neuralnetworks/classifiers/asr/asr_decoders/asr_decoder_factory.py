'''@file asr_factory
contains the asr decoder factory'''

from . import speller

def factory(conf, output_dim):
    '''create an asr classifier

    Args:
        conf: the classifier config as a dictionary
        output_dim: the classifier output dimension

    Returns:
        A decoder object'''

    if conf['decoder'] == 'speller':
        return speller.Speller(conf, output_dim)
    else:
        raise Exception('undefined asr decoder type: %s' % conf['decoder'])
