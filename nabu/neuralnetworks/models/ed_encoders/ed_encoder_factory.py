'''@file ed_encoder_factory.py
contains the EDEncoder factory'''

from . import listener, dummy_encoder, dblstm, dnn, listener_ps,\
bottleneck_encoder,hotstart_encoder

def factory(encoder):
    '''get an EDEncoder class

    Args:
        encoder: the encoder type

    Returns:
        an EDEncoder class'''

    if encoder == 'listener':
        return listener.Listener
    if encoder == 'listener_ps':
        return listener_ps.ListenerPS
    elif encoder == 'dummy_encoder':
        return dummy_encoder.DummyEncoder
    elif encoder == 'dblstm':
        return dblstm.DBLSTM
    elif encoder == 'dnn':
        return dnn.DNN
    elif encoder == 'bottleneck_encoder':
        return bottleneck_encoder.BottleneckEncoder
    elif encoder == 'hotstart_encoder':
        return hotstart_encoder.HotstartEncoder
    else:
        raise Exception('undefined encoder type: %s' % encoder)
