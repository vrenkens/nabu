'''@file ed_encoder_factory.py
contains the EDEncoder factory'''

from . import listener, dummy_encoder, dblstm, dnn, \
bottleneck_encoder, bldnn, hotstart_encoder, stack_encoder, parallel_encoder

def factory(encoder):
    '''get an EDEncoder class

    Args:
        encoder: the encoder type

    Returns:
        an EDEncoder class'''

    if encoder == 'listener':
        return listener.Listener
    elif encoder == 'dummy_encoder':
        return dummy_encoder.DummyEncoder
    elif encoder == 'dblstm':
        return dblstm.DBLSTM
    elif encoder == 'dnn':
        return dnn.DNN
    elif encoder == 'bottleneck_encoder':
        return bottleneck_encoder.BottleneckEncoder
    elif encoder == 'bldnn':
        return bldnn.BLDNN
    elif encoder == 'hotstart_encoder':
        return hotstart_encoder.HotstartEncoder
    elif encoder == 'stack_encoder':
        return stack_encoder.StackEncoder
    elif encoder == 'parallel_encoder':
        return parallel_encoder.ParallelEncoder
    else:
        raise Exception('undefined encoder type: %s' % encoder)
