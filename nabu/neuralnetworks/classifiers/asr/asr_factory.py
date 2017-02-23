'''@file asr_factory
contains the asr factory'''

from . import dblstm, dnn, wavenet, encoder_decoder

def factory(conf, output_dim):
    '''create an asr classifier

    Args:
        conf: the classifier config as a dictionary
        output_dim: the classifier output dimension

    Returns:
        A Classifier object'''

    if conf['asr'] == 'dblstm':
        return dblstm.DBLSTM(conf, output_dim)
    elif conf['asr'] == 'dnn':
        return dnn.DNN(conf, output_dim)
    elif conf['asr'] == 'wavenet':
        return wavenet.Wavenet(conf, output_dim)
    elif conf['asr'] == 'encoder_decoder':
        return encoder_decoder.EncoderDecoder(conf, output_dim)
    else:
        raise Exception('undefined asr type: %s' % conf['asr'])
