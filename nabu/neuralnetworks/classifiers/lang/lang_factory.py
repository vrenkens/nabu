'''@file lang_factory.py
contains the language model factory'''

from . import lstmlm

def factory(conf, output_dim):
    '''creates a language model classifier

    Args:
        conf: the language model config as a dictionary
        output_dim: the classifier output dimension

    Returns:
        A Classifier object'''

    if conf['lm'] == 'dblstm':
        return lstmlm.LstmLm(conf, output_dim)
    else:
        raise Exception('undefined language model type: %s' % conf['lm'])
