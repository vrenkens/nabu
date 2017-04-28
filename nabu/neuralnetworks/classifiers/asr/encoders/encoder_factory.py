'''@file asr_factory
contains the asr decoder factory'''

from . import listener, ff_listener

def factory(conf, name):
    '''create an asr classifier

    Args:
        conf: the classifier config as a dictionary
        name: the name of the encoder

    Returns:
        An encoder object'''

    if conf['encoder'] == 'listener':
        return listener.Listener(conf, name)
    if conf['encoder'] == 'ff_listener':
        return ff_listener.FfListener(conf, name)
    else:
        raise Exception('undefined asr encoder type: %s' % conf['encoder'])
